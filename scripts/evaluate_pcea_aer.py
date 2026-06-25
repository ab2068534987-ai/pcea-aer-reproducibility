from __future__ import annotations

import argparse
import csv
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np

from pcos.analysis.reporting import summarize, write_csv
from pcos.baselines.heuristics import heft_policy
from pcos.cli.main import load_config, make_env, provider_from_cfg
from pcos.postprocess.active_window_energy_reallocation import (
    ActiveWindowEnergyReallocator,
    capture_schedule,
)
from pcos.rl.pcea_ppo import PCEAPPOAgent, PCEAConfig


def pcea_config_from_cfg(cfg: Dict) -> PCEAConfig:
    pc = cfg.get("pcea", {})
    ppo = cfg.get("ppo", {})
    return PCEAConfig(
        gamma=ppo.get("gamma", 0.99),
        gae_lambda=ppo.get("gae_lambda", 0.95),
        lr=ppo.get("lr", 3e-4),
        ppo_clip=ppo.get("ppo_clip", 0.2),
        entropy_coef=ppo.get("entropy_coef", 0.01),
        value_coef=ppo.get("value_coef", 0.5),
        ppo_epochs=ppo.get("ppo_epochs", 4),
        minibatch_size=ppo.get("minibatch_size", 32),
        lambda_power_init=pc.get("lambda_power_init", 0.1),
        lambda_deadline_init=pc.get("lambda_deadline_init", 1.0),
        dual_lr_power=pc.get("dual_lr_power", 0.02),
        dual_lr_deadline=pc.get("dual_lr_deadline", 0.05),
        epsilon_power_violation=pc.get("epsilon_power_violation", pc.get("max_power_violation", 0.02)),
        epsilon_deadline_miss=pc.get("epsilon_deadline_miss", pc.get("max_deadline_miss", 0.01)),
        max_power_violation_soft=pc.get("max_power_violation_soft", pc.get("epsilon_power_violation", pc.get("max_power_violation", 0.02))),
        lambda_max=pc.get("lambda_max", 10.0),
        lambda_power_max=pc.get("lambda_power_max", pc.get("lambda_max", 10.0)),
        lambda_deadline_max=pc.get("lambda_deadline_max", pc.get("lambda_max", 10.0)),
        scalar_gae=pc.get("scalar_gae", False),
        fixed_dual=pc.get("fixed_dual", False),
        cpeg_enabled=pc.get("cpeg_enabled", False),
        cpeg_logit_penalty=pc.get("cpeg_logit_penalty", 8.0),
        anti_idle_enabled=pc.get("anti_idle_enabled", False),
        anti_idle_logit_penalty=pc.get("anti_idle_logit_penalty", 4.0),
        energy_guard_enabled=pc.get("energy_guard_enabled", False),
        energy_guard_logit_bonus=pc.get("energy_guard_logit_bonus", 3.0),
        activation_guard_enabled=pc.get("activation_guard_enabled", False),
        activation_logit_penalty=pc.get("activation_logit_penalty", 4.0),
        deterministic_repair_enabled=pc.get("deterministic_repair_enabled", False),
    )


def build_agent(cfg: Dict, checkpoint_path: Path, workflows, device: str) -> PCEAPPOAgent:
    import torch

    probe_env = make_env(cfg, seed=0)
    obs0 = probe_env.reset(workflows[0])
    global_dim = len(obs0["global"])
    pair_dim = obs0["pairs"].shape[1] if obs0["pairs"].size else 10
    agent = PCEAPPOAgent(
        global_dim=global_dim,
        pair_dim=pair_dim,
        config=pcea_config_from_cfg(cfg),
        hidden=cfg.get("ppo", {}).get("hidden", 128),
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    agent.model.load_state_dict(state)
    if isinstance(checkpoint, dict):
        agent.lambda_power = float(checkpoint.get("lambda_power", agent.lambda_power))
        agent.lambda_deadline = float(checkpoint.get("lambda_deadline", agent.lambda_deadline))
    return agent


def num(row: Dict[str, float], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def mean_value(row: Dict[str, float], key: str) -> float:
    if key in row:
        return float(row[key])
    return float(row.get(key + "_mean", 0.0))


def pct_gap(value: float, base: float) -> float:
    return (value - base) / max(abs(base), 1e-9) * 100.0


def fmt(x: float) -> str:
    return f"{x:.2f}"


def fmt_pct(x: float) -> str:
    return f"{x:+.2f}%"


def write_errors(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "workflow_id", "error", "traceback"])
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(heft_rows: List[Dict[str, float]], aer_rows: List[Dict[str, float]], heft_aer_rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    pcea_rows = []
    pcea_aer_rows = []
    for row in aer_rows:
        pcea_rows.append(
            {
                "method": "PCEA",
                "workflow_id": row["workflow_id"],
                "energy_active_total": row["pcea_energy_before"],
                "energy_idle_active": row["energy_idle_before"],
                "makespan": row["makespan_before"],
                "deadline_miss_rate": row["deadline_miss_rate_before"],
                "power_envelope_violation": row["power_envelope_violation_before"],
                "peak_active_power": row["peak_active_power_before"],
                "server_activated_count": row["server_activated_count_before"],
                "server_activated_count_before": row["server_activated_count_before"],
                "server_activated_count_after": row["server_activated_count_before"],
                "server_activated_delta": 0.0,
                "task_machine_ratio": row["task_machine_ratio"],
                "active_server_utilization": row["active_server_utilization_before"],
                "active_server_utilization_before": row["active_server_utilization_before"],
                "active_server_utilization_after": row["active_server_utilization_before"],
                "active_server_utilization_delta": 0.0,
                "defer_count": row["defer_count_before"],
                "migration_count": 0.0,
            }
        )
        pcea_aer_rows.append(
            {
                "method": "PCEA-AER",
                "workflow_id": row["workflow_id"],
                "energy_active_total": row["pcea_energy_after_aer"],
                "energy_idle_active": row["energy_idle_after"],
                "makespan": row["makespan_after"],
                "deadline_miss_rate": row["deadline_miss_rate_after"],
                "power_envelope_violation": row["power_envelope_violation_after"],
                "peak_active_power": row["peak_active_power_after"],
                "server_activated_count": row["server_activated_count_after"],
                "server_activated_count_before": row["server_activated_count_before"],
                "server_activated_count_after": row["server_activated_count_after"],
                "server_activated_delta": row["server_activated_delta"],
                "task_machine_ratio": row["task_machine_ratio"],
                "active_server_utilization": row["active_server_utilization_after"],
                "active_server_utilization_before": row["active_server_utilization_before"],
                "active_server_utilization_after": row["active_server_utilization_after"],
                "active_server_utilization_delta": row["active_server_utilization_delta"],
                "defer_count": row["defer_count_after"],
                "migration_count": row["migration_count"],
            }
        )
    comparison_rows = []
    comparison_rows.extend(heft_rows)
    comparison_rows.extend(pcea_rows)
    comparison_rows.extend(pcea_aer_rows)
    comparison_rows.extend(heft_aer_rows)
    summary_rows = []
    for method in ["HEFT", "PCEA", "PCEA-AER", "HEFT-AER"]:
        xs = [r for r in comparison_rows if r.get("method") == method]
        if xs:
            summary = summarize(xs)
            summary["method"] = method
            summary_rows.append(summary)
    return summary_rows


def table(rows: List[Dict[str, float]], fields: List[str]) -> str:
    out = ["| method | " + " | ".join(fields) + " |", "|---" + "|---:" * len(fields) + "|"]
    for row in rows:
        vals = [str(row.get("method", ""))]
        for field in fields:
            vals.append(fmt(mean_value(row, field)))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def build_report(out_dir: Path, aer_rows: List[Dict[str, float]], summary_rows: List[Dict[str, float]], errors: List[Dict[str, str]]) -> None:
    by_method = {str(r["method"]): r for r in summary_rows}
    heft = by_method["HEFT"]
    pcea = by_method["PCEA"]
    pcea_aer = by_method["PCEA-AER"]
    heft_energy = mean_value(heft, "energy_active_total")
    pcea_energy = mean_value(pcea, "energy_active_total")
    aer_energy = mean_value(pcea_aer, "energy_active_total")
    pcea_idle = mean_value(pcea, "energy_idle_active")
    aer_idle = mean_value(pcea_aer, "energy_idle_active")
    energy_delta = aer_energy - pcea_energy
    idle_delta = aer_idle - pcea_idle
    idle_share = abs(idle_delta) / max(abs(energy_delta), 1e-9) * 100.0
    makespan_delta = mean_value(pcea_aer, "makespan") - mean_value(pcea, "makespan")
    deadline_delta = mean_value(pcea_aer, "deadline_miss_rate") - mean_value(pcea, "deadline_miss_rate")
    guard_events = [r for r in aer_rows if r["constraint_guard_flag"] > 0.5]
    error_count = len(errors)

    fields = [
        "energy_active_total",
        "energy_idle_active",
        "makespan",
        "deadline_miss_rate",
        "power_envelope_violation",
        "peak_active_power",
        "server_activated_count",
        "task_machine_ratio",
        "active_server_utilization",
        "defer_count",
        "migration_count",
    ]

    task_machine_ratio = mean_value(pcea_aer, "task_machine_ratio")
    server_delta = mean_value(pcea_aer, "server_activated_delta")
    util_before = mean_value(pcea_aer, "active_server_utilization_before")
    util_after = mean_value(pcea_aer, "active_server_utilization_after")
    util_delta = mean_value(pcea_aer, "active_server_utilization_delta")

    report = f"""# PCEA-AER Single-Workflow Evaluation

## Summary

- Benchmark workflows evaluated: {len(aer_rows)}.
- HEFT active energy: {fmt(heft_energy)}.
- PCEA-AER active energy: {fmt(aer_energy)}; relative to HEFT {fmt_pct(pct_gap(aer_energy, heft_energy))}.
- PCEA-AER relative to PCEA active energy: {fmt_pct(pct_gap(aer_energy, pcea_energy))}.
- Idle-active energy delta from PCEA to PCEA-AER: {fmt(idle_delta)}; share of active-energy delta: {fmt(idle_share)}%.
- Makespan delta from PCEA to PCEA-AER: {fmt(makespan_delta)}.
- Deadline-miss-rate delta from PCEA to PCEA-AER: {deadline_delta:.8f}.
- Mean migration count: {mean_value(pcea_aer, 'migration_count'):.2f}.
- Constraint guard events: {len(guard_events)}; recorded execution errors: {error_count}.

## Resource Utilization

- Mean `task_machine_ratio`: {task_machine_ratio:.2f}.
- Mean `server_activated_count` delta: {server_delta:.2f}.
- Mean `active_server_utilization`: {util_before:.4f} -> {util_after:.4f}; delta = {util_delta:.4f}.

## Method Comparison

{table(summary_rows, fields)}

## Files

- `{(out_dir / 'aer_metrics.csv').as_posix()}`
- `{(out_dir / 'aer_summary.csv').as_posix()}`
- `{(out_dir / 'aer_report.md').as_posix()}`
"""
    (out_dir / "aer_report.md").write_text(report, encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PCEA-AER on a fixed workflow split.")
    parser.add_argument("--config", default="configs/pcea_medium_cluster15_benchmark.json")
    parser.add_argument("--pcea-config", default="configs/train_pcea_medium_cluster15_benchmark_pcea_anti_idle.json")
    parser.add_argument("--checkpoint", default="models/pcea_aer/best_energy_pcea_ppo.pt")
    parser.add_argument("--split", default="benchmark")
    parser.add_argument("--limit", type=int, default=136)
    parser.add_argument("--output", default="results/eval_pcea_aer_single_workflow_anti_idle_default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--max-passes", type=int, default=2)
    parser.add_argument("--candidate-top-k", type=int, default=0)
    parser.add_argument("--include-heft-aer", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    pcea_cfg = load_config(args.pcea_config)
    workflows = list(provider_from_cfg(base_cfg, split=args.split, seed=args.seed).iter_workflows(limit=args.limit))
    if not workflows:
        raise RuntimeError(f"No workflows found for split={args.split!r}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors: List[Dict[str, str]] = []
    agent = build_agent(pcea_cfg, Path(args.checkpoint), workflows, args.device)

    heft_rows: List[Dict[str, float]] = []
    aer_rows: List[Dict[str, float]] = []
    heft_aer_rows: List[Dict[str, float]] = []
    reallocator = ActiveWindowEnergyReallocator(
        env_factory=lambda: make_env(pcea_cfg, seed=args.seed),
        max_steps=args.max_steps,
        max_passes=args.max_passes,
        candidate_top_k=args.candidate_top_k,
    )

    for workflow in workflows:
        try:
            def heft_action_selector(env, _obs):
                return heft_policy(env)

            heft_schedule = capture_schedule(
                make_env(base_cfg, seed=args.seed),
                workflow,
                "HEFT",
                heft_action_selector,
                max_steps=args.max_steps,
            )
            heft_metrics = heft_schedule.metrics
            heft_metrics["migration_count"] = 0.0
            heft_metrics["task_machine_ratio"] = len(workflow.tasks) / max(1.0, float(len(make_env(base_cfg, seed=args.seed).cluster_specs)))
            heft_metrics["server_activated_count_before"] = heft_metrics["server_activated_count"]
            heft_metrics["server_activated_count_after"] = heft_metrics["server_activated_count"]
            heft_metrics["server_activated_delta"] = 0.0
            heft_metrics["active_server_utilization_before"] = heft_metrics["active_server_utilization"]
            heft_metrics["active_server_utilization_after"] = heft_metrics["active_server_utilization"]
            heft_metrics["active_server_utilization_delta"] = 0.0
            heft_rows.append(heft_metrics)

            def pcea_action_selector(env, obs):
                if len(obs["mask"]) == 0 or float(np.asarray(obs["mask"], dtype=np.float32).sum()) <= 0.0:
                    env._ensure_actionable_or_done()
                    obs = env._observe()
                    if env.done:
                        return 0
                action, _, _ = agent.act(obs, deterministic=True)
                return int(action)

            schedule = capture_schedule(
                make_env(pcea_cfg, seed=args.seed),
                workflow,
                "PCEA",
                pcea_action_selector,
                max_steps=args.max_steps,
            )
            aer = reallocator.apply(workflow, schedule)
            before = aer.before_metrics
            after = aer.after_metrics
            task_machine_ratio = len(workflow.tasks) / max(1.0, float(len(make_env(pcea_cfg, seed=args.seed).cluster_specs)))
            server_activated_delta = before["server_activated_count"] - after["server_activated_count"]
            active_util_before = before.get("active_server_utilization", 0.0)
            active_util_after = after.get("active_server_utilization", 0.0)
            constraint_guard_flag = float(
                after["makespan"] > before["makespan"] + 1e-9
                or after["deadline_miss_rate"] > before["deadline_miss_rate"] + 1e-12
            )
            aer_rows.append(
                {
                    "workflow_id": workflow.workflow_id,
                    "pcea_energy_before": before["energy_active_total"],
                    "pcea_energy_after_aer": after["energy_active_total"],
                    "energy_reduction_ratio": (before["energy_active_total"] - after["energy_active_total"]) / max(before["energy_active_total"], 1e-9),
                    "makespan_before": before["makespan"],
                    "makespan_after": after["makespan"],
                    "energy_idle_before": before["energy_idle_active"],
                    "energy_idle_after": after["energy_idle_active"],
                    "deadline_miss_rate": after["deadline_miss_rate"],
                    "deadline_miss_rate_before": before["deadline_miss_rate"],
                    "deadline_miss_rate_after": after["deadline_miss_rate"],
                    "migration_count": float(aer.migration_count),
                    "task_machine_ratio": task_machine_ratio,
                    "server_activated_count_before": before["server_activated_count"],
                    "server_activated_count_after": after["server_activated_count"],
                    "server_activated_delta": server_activated_delta,
                    "active_server_utilization_before": active_util_before,
                    "active_server_utilization_after": active_util_after,
                    "active_server_utilization_delta": active_util_after - active_util_before,
                    "power_envelope_violation_before": before["power_envelope_violation"],
                    "power_envelope_violation_after": after["power_envelope_violation"],
                    "peak_active_power_before": before["peak_active_power"],
                    "peak_active_power_after": after["peak_active_power"],
                    "defer_count_before": before["defer_count"],
                    "defer_count_after": after["defer_count"],
                    "pcea_aer_energy_gap_vs_heft": after["energy_active_total"] - heft_metrics["energy_active_total"],
                    "pcea_aer_energy_gap_pct_vs_heft": pct_gap(after["energy_active_total"], heft_metrics["energy_active_total"]),
                    "candidates_evaluated": float(aer.candidates_evaluated),
                    "passes_done": float(aer.passes_done),
                    "constraint_replay_count": float(aer.constraint_violations),
                    "constraint_guard_flag": constraint_guard_flag,
                }
            )

            if args.include_heft_aer:
                def heft_action_selector(env, _obs):
                    return heft_policy(env)

                heft_reallocator = ActiveWindowEnergyReallocator(
                    env_factory=lambda: make_env(base_cfg, seed=args.seed),
                    max_steps=args.max_steps,
                    max_passes=args.max_passes,
                    candidate_top_k=args.candidate_top_k,
                )
                heft_aer = heft_reallocator.apply(workflow, heft_schedule)
                hm = heft_aer.after_metrics
                h_server_delta = heft_schedule.metrics["server_activated_count"] - hm["server_activated_count"]
                h_util_before = heft_schedule.metrics.get("active_server_utilization", 0.0)
                h_util_after = hm.get("active_server_utilization", 0.0)
                heft_aer_rows.append(
                    {
                        "method": "HEFT-AER",
                        "workflow_id": workflow.workflow_id,
                        "energy_active_total": hm["energy_active_total"],
                        "energy_idle_active": hm["energy_idle_active"],
                        "makespan": hm["makespan"],
                        "deadline_miss_rate": hm["deadline_miss_rate"],
                        "power_envelope_violation": hm["power_envelope_violation"],
                        "peak_active_power": hm["peak_active_power"],
                        "server_activated_count": hm["server_activated_count"],
                        "server_activated_count_before": heft_schedule.metrics["server_activated_count"],
                        "server_activated_count_after": hm["server_activated_count"],
                        "server_activated_delta": h_server_delta,
                        "task_machine_ratio": heft_metrics["task_machine_ratio"],
                        "active_server_utilization": h_util_after,
                        "active_server_utilization_before": h_util_before,
                        "active_server_utilization_after": h_util_after,
                        "active_server_utilization_delta": h_util_after - h_util_before,
                        "defer_count": hm["defer_count"],
                        "migration_count": float(heft_aer.migration_count),
                    }
                )
        except Exception as exc:
            errors.append({"method": "PCEA-AER", "workflow_id": workflow.workflow_id, "error": str(exc), "traceback": traceback.format_exc()})
        print(f"workflow={workflow.workflow_id} done ({len(aer_rows)}/{len(workflows)})")

    summary_rows = build_summary_rows(heft_rows, aer_rows, heft_aer_rows)
    write_csv(out_dir / "aer_metrics.csv", aer_rows)
    write_csv(out_dir / "aer_summary.csv", summary_rows)
    write_errors(out_dir / "aer_errors.csv", errors)
    if aer_rows:
        build_report(out_dir, aer_rows, summary_rows, errors)
    print(f"evaluated={len(aer_rows)} errors={len(errors)}")
    print(f"wrote {out_dir / 'aer_report.md'}")


if __name__ == "__main__":
    main()
