from __future__ import annotations

import argparse
import csv
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from pcos.analysis.reporting import summarize, write_csv
from pcos.baselines.heuristics import fcfs_policy, heft_policy, min_energy_policy
from pcos.cli.main import load_config, make_env, provider_from_cfg
from pcos.core.entities import Task, Workflow
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


def build_agent(cfg: Dict, checkpoint_path: Path, probe_workflow, device: str) -> PCEAPPOAgent:
    import torch

    probe_env = make_env(cfg, seed=0)
    obs0 = probe_env.reset(probe_workflow)
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


def workflow_target(workflow: Workflow) -> float:
    if workflow.metadata and workflow.metadata.get("makespan_target") is not None:
        try:
            return float(workflow.metadata["makespan_target"])
        except (TypeError, ValueError):
            pass
    if workflow.makespan_target is not None:
        return float(workflow.makespan_target)
    return max((float(t.deadline or 0.0) for t in workflow.tasks), default=1.0)


def make_group_workflow(group_id: str, workflows: List[Workflow]) -> Workflow:
    tasks: List[Task] = []
    edges: List[Tuple[int, int]] = []
    component_ids: Dict[int, str] = {}
    task_offsets: Dict[str, int] = {}
    for workflow in workflows:
        offset = len(tasks)
        task_offsets[workflow.workflow_id] = offset
        old_to_new = {task.id: offset + i for i, task in enumerate(workflow.tasks)}
        for i, task in enumerate(workflow.tasks):
            new_id = offset + i
            component_ids[new_id] = workflow.workflow_id
            tasks.append(
                replace(
                    task,
                    id=new_id,
                    preds=[old_to_new[pred] for pred in task.preds],
                    succs=[old_to_new[succ] for succ in task.succs],
                )
            )
        for src, dst in workflow.edges:
            if src in old_to_new and dst in old_to_new:
                edges.append((old_to_new[src], old_to_new[dst]))
    makespan_target = max([workflow_target(workflow) for workflow in workflows] or [1.0])
    metadata = {
        "workflow_ids": [workflow.workflow_id for workflow in workflows],
        "group_size": len(workflows),
        "submit_time": 0.0,
        "makespan_target": makespan_target,
        "component_task_offsets": task_offsets,
        "component_ids": component_ids,
    }
    return Workflow(
        workflow_id=group_id,
        tasks=tasks,
        edges=edges,
        metadata=metadata,
        makespan_target=makespan_target,
    )


def make_groups(workflows: List[Workflow], group_size: int) -> List[Tuple[str, List[Workflow], Workflow]]:
    usable = len(workflows) // group_size * group_size
    groups = []
    for start in range(0, usable, group_size):
        group_workflows = workflows[start : start + group_size]
        group_id = f"group_{start // group_size:03d}"
        groups.append((group_id, group_workflows, make_group_workflow(group_id, group_workflows)))
    return groups


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


def fmt(value: float) -> str:
    return f"{value:.2f}"


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def method_row(method: str, group_id: str, workflow_ids: List[str], metrics: Dict[str, float], migration_count: float = 0.0) -> Dict[str, float]:
    return {
        "method": method,
        "group_id": group_id,
        "workflow_ids": ",".join(workflow_ids),
        "group_size": float(len(workflow_ids)),
        "num_tasks": float(metrics.get("num_tasks", 0.0)),
        "group_makespan": metrics["makespan"],
        "group_energy_active_total": metrics["energy_active_total"],
        "energy_idle_active": metrics["energy_idle_active"],
        "deadline_miss_rate": metrics["deadline_miss_rate"],
        "peak_active_power": metrics["peak_active_power"],
        "power_envelope_violation": metrics["power_envelope_violation"],
        "server_activated_count": metrics["server_activated_count"],
        "active_server_utilization": metrics.get("active_server_utilization", 0.0),
        "avg_cpu_utilization": metrics.get("avg_cpu_utilization", 0.0),
        "avg_gpu_utilization": metrics.get("avg_gpu_utilization", 0.0),
        "total_task_busy_time": metrics.get("total_task_busy_time", 0.0),
        "total_active_window_time": metrics.get("total_active_window_time", 0.0),
        "total_cpu_busy_capacity_time": metrics.get("total_cpu_busy_capacity_time", 0.0),
        "total_gpu_busy_capacity_time": metrics.get("total_gpu_busy_capacity_time", 0.0),
        "total_cpu_active_capacity_time": metrics.get("total_cpu_active_capacity_time", 0.0),
        "total_gpu_active_capacity_time": metrics.get("total_gpu_active_capacity_time", 0.0),
        "defer_count": metrics.get("defer_count", 0.0),
        "migration_count": migration_count,
    }


def add_num_tasks(metrics: Dict[str, float], workflow: Workflow) -> Dict[str, float]:
    metrics["num_tasks"] = float(len(workflow.tasks))
    return metrics


def write_group_manifest(path: Path, groups: List[Tuple[str, List[Workflow], Workflow]], dropped: List[Workflow]) -> None:
    rows = []
    for group_id, workflows, group_workflow in groups:
        rows.append(
            {
                "group_id": group_id,
                "workflow_ids": ",".join(workflow.workflow_id for workflow in workflows),
                "group_size": len(workflows),
                "num_tasks": len(group_workflow.tasks),
            }
        )
    for workflow in dropped:
        rows.append(
            {
                "group_id": "dropped",
                "workflow_ids": workflow.workflow_id,
                "group_size": 0,
                "num_tasks": len(workflow.tasks),
            }
        )
    write_csv(path, rows)


def read_summary(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return {row["method"]: row for row in csv.DictReader(f)}


def table(rows: List[Dict[str, float]], fields: List[str]) -> str:
    out = ["| method | " + " | ".join(fields) + " |", "|---" + "|---:" * len(fields) + "|"]
    for row in rows:
        vals = [str(row.get("method", ""))]
        for field in fields:
            vals.append(fmt(mean_value(row, field)))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def build_report(out_dir: Path, rows: List[Dict[str, float]], summary_rows: List[Dict[str, float]], dropped_count: int, single_summary_path: Path) -> None:
    by_method = {str(row["method"]): row for row in summary_rows}
    heft = by_method["HEFT"]
    pcea = by_method["PCEA"]
    pcea_aer = by_method["PCEA-AER"]
    heft_energy = mean_value(heft, "group_energy_active_total")
    aer_energy = mean_value(pcea_aer, "group_energy_active_total")
    cpu_util = mean_value(pcea_aer, "avg_cpu_utilization")
    gpu_util = mean_value(pcea_aer, "avg_gpu_utilization")
    single = read_summary(single_summary_path)
    single_lines = []
    for method in ["HEFT", "PCEA", "PCEA-AER"]:
        group_makespan = mean_value(by_method[method], "group_makespan")
        single_row = single.get(method)
        if single_row:
            single_makespan = num(single_row, "makespan_mean")
            single_lines.append(
                f"- {method}: group makespan = {fmt(group_makespan)}, single-workflow makespan = {fmt(single_makespan)}, relative change {fmt_pct(pct_gap(group_makespan, single_makespan))}."
            )
        else:
            single_lines.append(f"- {method}: group makespan = {fmt(group_makespan)}; single-workflow summary not provided.")

    fields = [
        "group_energy_active_total",
        "group_makespan",
        "deadline_miss_rate",
        "peak_active_power",
        "power_envelope_violation",
        "server_activated_count",
        "active_server_utilization",
        "avg_cpu_utilization",
        "avg_gpu_utilization",
        "migration_count",
    ]
    report = f"""# PCEA-AER Group-Size-5 Evaluation

## Summary

- Groups evaluated: {len({row['group_id'] for row in rows})}.
- Workflows per group: 5.
- Benchmark workflows covered: {len({wf for row in rows for wf in str(row['workflow_ids']).split(',') if wf}) + dropped_count}.
- Unused remainder workflows: {dropped_count}.
- PCEA-AER active energy: {fmt(aer_energy)}; HEFT active energy: {fmt(heft_energy)}; relative change {fmt_pct(pct_gap(aer_energy, heft_energy))}.
- PCEA-AER relative to PCEA active energy: {fmt_pct(pct_gap(aer_energy, mean_value(pcea, 'group_energy_active_total')))}.
- PCEA-AER mean CPU utilization: {cpu_util:.4f}.
- PCEA-AER mean GPU utilization: {gpu_util:.4f}.

## Method Comparison

{table(summary_rows, fields)}

## Makespan Context

{chr(10).join(single_lines)}

## Notes

- Each group contains 5 independent workflows submitted at t=0.
- `group_energy_active_total` is integrated over shared cluster resources in the same `SchedulerEnv`; it is not a sum of isolated single-workflow energy values.
- Workflows in a group have no cross-workflow dependencies and share the same cluster capacity.
- `avg_cpu_utilization` and `avg_gpu_utilization` are capacity-time weighted averages over the active window.

## Files

- `{(out_dir / 'group_manifest.csv').as_posix()}`
- `{(out_dir / 'metrics.csv').as_posix()}`
- `{(out_dir / 'summary.csv').as_posix()}`
- `{(out_dir / 'report.md').as_posix()}`
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate group-size-5 curated benchmark for HEFT, PCEA and PCEA-AER.")
    parser.add_argument("--config", default="configs/pcea_medium_cluster15_benchmark.json")
    parser.add_argument("--pcea-config", default="configs/train_pcea_medium_cluster15_benchmark_pcea_anti_idle.json")
    parser.add_argument("--checkpoint", default="models/pcea_aer/best_energy_pcea_ppo.pt")
    parser.add_argument("--split", default="benchmark")
    parser.add_argument("--limit", type=int, default=136)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--output", default="results/01_main_group5_workflow_anti_idle_default")
    parser.add_argument("--single-summary", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--aer-max-passes", type=int, default=1)
    parser.add_argument("--aer-candidate-top-k", type=int, default=4)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    pcea_cfg = load_config(args.pcea_config)
    workflows = list(provider_from_cfg(base_cfg, split=args.split, seed=args.seed).iter_workflows(limit=args.limit))
    groups = make_groups(workflows, args.group_size)
    dropped = workflows[len(groups) * args.group_size :]
    if not groups:
        raise RuntimeError("No complete groups were produced.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_group_manifest(out_dir / "group_manifest.csv", groups, dropped)

    agent = build_agent(pcea_cfg, Path(args.checkpoint), groups[0][2], args.device)
    rows: List[Dict[str, float]] = []
    errors: List[Dict[str, str]] = []
    reallocator = ActiveWindowEnergyReallocator(
        env_factory=lambda: make_env(pcea_cfg, seed=args.seed),
        max_steps=args.max_steps,
        max_passes=args.aer_max_passes,
        candidate_top_k=args.aer_candidate_top_k,
    )

    for group_id, component_workflows, group_workflow in groups:
        workflow_ids = [workflow.workflow_id for workflow in component_workflows]
        try:
            baseline_policies = [
                ("FCFS", fcfs_policy),
                ("HEFT", heft_policy),
                ("MinEnergy", min_energy_policy),
            ]
            for baseline_name, baseline_policy in baseline_policies:
                def baseline_action_selector(env, _obs, policy=baseline_policy):
                    return policy(env)

                baseline_schedule = capture_schedule(
                    make_env(base_cfg, seed=args.seed),
                    group_workflow,
                    baseline_name,
                    baseline_action_selector,
                    max_steps=args.max_steps,
                )
                rows.append(
                    method_row(
                        baseline_name,
                        group_id,
                        workflow_ids,
                        add_num_tasks(baseline_schedule.metrics, group_workflow),
                    )
                )

            def pcea_action_selector(env, obs):
                if len(obs["mask"]) == 0 or float(np.asarray(obs["mask"], dtype=np.float32).sum()) <= 0.0:
                    env._ensure_actionable_or_done()
                    obs = env._observe()
                    if env.done:
                        return 0
                action, _, _ = agent.act(obs, deterministic=True)
                return int(action)

            pcea_schedule = capture_schedule(
                make_env(pcea_cfg, seed=args.seed),
                group_workflow,
                "PCEA",
                pcea_action_selector,
                max_steps=args.max_steps,
            )
            rows.append(method_row("PCEA", group_id, workflow_ids, add_num_tasks(pcea_schedule.metrics, group_workflow)))

            aer = reallocator.apply(group_workflow, pcea_schedule)
            aer_metrics = add_num_tasks(aer.after_metrics, group_workflow)
            rows.append(method_row("PCEA-AER", group_id, workflow_ids, aer_metrics, migration_count=float(aer.migration_count)))
        except Exception as exc:
            errors.append(
                {
                    "group_id": group_id,
                    "workflow_ids": ",".join(workflow_ids),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        print(f"group={group_id} done rows={len(rows)} errors={len(errors)}")

    write_csv(out_dir / "metrics.csv", rows)
    summary_rows: List[Dict[str, float]] = []
    for method in ["FCFS", "HEFT", "MinEnergy", "PCEA", "PCEA-AER"]:
        xs = [row for row in rows if row.get("method") == method]
        if xs:
            summary = summarize(xs)
            summary["method"] = method
            summary_rows.append(summary)
    write_csv(out_dir / "summary.csv", summary_rows)
    if errors:
        write_csv(out_dir / "errors.csv", errors)
    build_report(out_dir, rows, summary_rows, len(dropped), Path(args.single_summary))
    print(f"groups={len(groups)} dropped={len(dropped)} rows={len(rows)} errors={len(errors)}")
    print(f"wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
