from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from pcos.analysis.reporting import summarize, write_csv
from pcos.baselines.heuristics import BASELINE_POLICIES, run_policy
from pcos.core.entities import cluster_specs_from_config
from pcos.data.provider import WorkflowProvider
from pcos.env.power_envelope import PowerEnvelopeProvider
from pcos.env.scheduler_env import EnvConfig, SchedulerEnv


def load_config(path: str | None) -> Dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_env(cfg: Dict, seed: int = 0) -> SchedulerEnv:
    cluster = cluster_specs_from_config(cfg.get("cluster"))
    full = sum(s.full_power() for s in cluster)
    pcfg = cfg.get("power_envelope", {})
    provider = PowerEnvelopeProvider(
        cluster_full_power_w=full,
        scenario=pcfg.get("scenario", "peak_valley"),
        slot_s=pcfg.get("slot_s", 300.0),
        b_min_ratio=pcfg.get("b_min_ratio", 0.55),
        b_max_ratio=pcfg.get("b_max_ratio", 0.90),
        phase_mode=pcfg.get("phase_mode", "random"),
        csv_path=pcfg.get("csv_path"),
        seed=seed,
    )
    ecfg_raw = cfg.get("env", {})
    dcfg = cfg.get("defer", {})
    ecfg = EnvConfig(
        idle_timeout_s=ecfg_raw.get("idle_timeout_s", 300.0),
        bandwidth_MBps=ecfg_raw.get("bandwidth_MBps", 250.0),
        latency_s=ecfg_raw.get("latency_s", 0.01),
        energy_per_MB=ecfg_raw.get("energy_per_MB", 0.02),
        max_defer_s=dcfg.get("max_defer_s", 300.0),
        safe_slack_margin_s=dcfg.get("safe_slack_margin_s", 30.0),
        headroom_margin_ratio=dcfg.get("headroom_margin_ratio", 0.05),
        require_power_pressure=dcfg.get("require_power_pressure", True),
        enable_defer=dcfg.get("enabled", True),
        energy_norm=ecfg_raw.get("energy_norm", 10000.0),
        performance_cost_coef=cfg.get("pcea", {}).get("performance_cost_coef", ecfg_raw.get("performance_cost_coef", 0.20)),
        urgent_slack_ratio=dcfg.get("urgent_slack_ratio", 0.15),
        min_defer_power_pressure_ratio=dcfg.get("min_defer_power_pressure_ratio", 0.0),
        cpeg_enabled=cfg.get("pcea", {}).get("cpeg_enabled", False),
        cpeg_criticality_threshold=cfg.get("pcea", {}).get("cpeg_criticality_threshold", 0.60),
        cpeg_eft_regret_ratio=cfg.get("pcea", {}).get("cpeg_eft_regret_ratio", 0.10),
        cpeg_power_pressure_margin=cfg.get("pcea", {}).get("cpeg_power_pressure_margin", 0.02),
        cpeg_cost_coef=cfg.get("pcea", {}).get("cpeg_cost_coef", 0.50),
        anti_idle_enabled=cfg.get("pcea", {}).get("anti_idle_enabled", False),
        anti_idle_eft_regret_ratio=cfg.get("pcea", {}).get("anti_idle_eft_regret_ratio", 0.02),
        anti_idle_power_pressure_margin=cfg.get("pcea", {}).get("anti_idle_power_pressure_margin", 0.01),
        anti_idle_cost_coef=cfg.get("pcea", {}).get("anti_idle_cost_coef", 0.60),
        anti_idle_idle_cost_coef=cfg.get("pcea", {}).get("anti_idle_idle_cost_coef", 0.50),
        energy_guard_enabled=cfg.get("pcea", {}).get("energy_guard_enabled", False),
        critical_eft_window_ratio=cfg.get("pcea", {}).get("critical_eft_window_ratio", 0.03),
        noncritical_eft_window_ratio=cfg.get("pcea", {}).get("noncritical_eft_window_ratio", 0.08),
        energy_guard_cost_coef=cfg.get("pcea", {}).get("energy_guard_cost_coef", 0.0),
        energy_guard_include_idle_tail=cfg.get("pcea", {}).get("energy_guard_include_idle_tail", False),
        active_energy_cost_coef=cfg.get("pcea", {}).get("active_energy_cost_coef", 1.0),
        idle_cost_coef=cfg.get("pcea", {}).get("idle_cost_coef", 0.0),
        activation_guard_enabled=cfg.get("pcea", {}).get("activation_guard_enabled", False),
        activation_eft_gain_required=cfg.get("pcea", {}).get("activation_eft_gain_required", 0.05),
        activation_guard_max_criticality=cfg.get("pcea", {}).get("activation_guard_max_criticality", 1.0),
        activation_guard_max_ready_width=cfg.get("pcea", {}).get("activation_guard_max_ready_width", 1_000_000),
        activation_guard_cost_coef=cfg.get("pcea", {}).get("activation_guard_cost_coef", 0.0),
    )
    return SchedulerEnv(cluster=cluster, envelope_provider=provider, config=ecfg, seed=seed)


def provider_from_cfg(cfg: Dict, split: str, seed: int = 0) -> WorkflowProvider:
    data_root = cfg.get("data", {}).get("root", "datasets/alibaba_pcea_sample/processed")
    limit = cfg.get("data", {}).get(f"{split}_limit") or cfg.get("data", {}).get("limit")
    return WorkflowProvider(data_root, split=split, seed=seed, limit=limit)


def evaluate_agent(agent, cfg: Dict, split: str, seed: int, limit: int | None, max_steps: int) -> List[Dict[str, float]]:
    env = make_env(cfg, seed=seed)
    provider = provider_from_cfg(cfg, split=split, seed=seed)
    rows: List[Dict[str, float]] = []
    for wf in provider.iter_workflows(limit=limit):
        obs = env.reset(wf)
        steps = 0
        while not env.done and steps < max_steps:
            if len(obs["mask"]) == 0 or float(np.asarray(obs["mask"], dtype=np.float32).sum()) <= 0.0:
                env._ensure_actionable_or_done()
                obs = env._observe()
                if env.done:
                    break
                if len(obs["mask"]) == 0 or float(np.asarray(obs["mask"], dtype=np.float32).sum()) <= 0.0:
                    raise RuntimeError(f"No legal action during validation: {env._diagnostic_action_state()}")
            action, _, _ = agent.act(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            steps += 1
            if done:
                break
        if not env.done:
            raise RuntimeError(f"Validation exceeded max_steps={max_steps}: {env._diagnostic_action_state()}")
        metrics = env._episode_info()
        metrics["steps"] = steps
        metrics["workflow_id"] = wf.workflow_id
        rows.append(metrics)
    return rows


def cmd_benchmark(args) -> None:
    cfg = load_config(args.config)
    out_dir = Path(args.output or cfg.get("output_dir", "outputs/benchmark"))
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = args.methods.split(",") if args.methods else cfg.get("benchmark", {}).get("methods", ["fcfs", "heft", "min_energy"])
    provider = provider_from_cfg(cfg, split=args.split, seed=args.seed)
    rows_all = []
    for method in methods:
        if method not in BASELINE_POLICIES:
            print(f"[skip] unsupported benchmark baseline: {method}")
            continue
        env = make_env(cfg, seed=args.seed)
        rows = []
        for wf in provider.iter_workflows(limit=args.limit):
            metrics = run_policy(env, wf, BASELINE_POLICIES[method])
            metrics["method"] = method
            metrics["workflow_id"] = wf.workflow_id
            rows.append(metrics)
            rows_all.append(metrics)
        write_csv(out_dir / f"{method}_{args.split}_metrics.csv", rows)
        summary = summarize(rows)
        summary["method"] = method
        write_csv(out_dir / f"{method}_{args.split}_summary.csv", [summary])
        print(method, summary)
    write_csv(out_dir / f"all_{args.split}_metrics.csv", rows_all)
    summary_rows = []
    for method in methods:
        xs = [r for r in rows_all if r.get("method") == method]
        if xs:
            s = summarize(xs)
            s["method"] = method
            summary_rows.append(s)
    write_csv(out_dir / f"all_{args.split}_summary.csv", summary_rows)


def cmd_train(args) -> None:
    cfg = load_config(args.config)
    algo = args.algo
    if algo not in {"pcea_ppo", "ppo"}:
        raise ValueError("Clean project supports train --algo pcea_ppo or ppo. Baselines use benchmark.")
    from pcos.rl.pcea_ppo import PCEAPPOAgent, PCEAConfig, RolloutBuffer

    out_dir = Path(args.output or cfg.get("output_dir", "outputs/train"))
    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(cfg, seed=args.seed)
    provider = provider_from_cfg(cfg, split="train", seed=args.seed)
    # Probe dimensions.
    obs0 = env.reset(provider.sample())
    global_dim = len(obs0["global"])
    pair_dim = obs0["pairs"].shape[1] if obs0["pairs"].size else 10
    pc = cfg.get("pcea", {})
    ppo_cfg = PCEAConfig(
        gamma=cfg.get("ppo", {}).get("gamma", 0.99),
        gae_lambda=cfg.get("ppo", {}).get("gae_lambda", 0.95),
        lr=cfg.get("ppo", {}).get("lr", 3e-4),
        ppo_clip=cfg.get("ppo", {}).get("ppo_clip", 0.2),
        entropy_coef=cfg.get("ppo", {}).get("entropy_coef", 0.01),
        value_coef=cfg.get("ppo", {}).get("value_coef", 0.5),
        ppo_epochs=cfg.get("ppo", {}).get("ppo_epochs", 4),
        minibatch_size=cfg.get("ppo", {}).get("minibatch_size", 32),
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
        scalar_gae=(algo == "ppo") or pc.get("scalar_gae", False),
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
    agent = PCEAPPOAgent(global_dim=global_dim, pair_dim=pair_dim, config=ppo_cfg, hidden=cfg.get("ppo", {}).get("hidden", 128), device=args.device)
    buffer = RolloutBuffer()
    train_rows = []
    train_cfg = cfg.get("train", {})
    episodes_per_iter = int(train_cfg.get("episodes_per_iter", 4))
    max_steps = int(train_cfg.get("max_steps", 2000))
    configured_total = train_cfg.get("total_episodes", train_cfg.get("train_episodes", train_cfg.get("episodes")))
    if args.iterations is not None:
        iterations = int(args.iterations)
        total_episodes = iterations * episodes_per_iter
    elif configured_total is not None:
        total_episodes = int(configured_total)
        iterations = int(math.ceil(total_episodes / max(1, episodes_per_iter)))
    else:
        iterations = int(train_cfg.get("iterations", 20))
        total_episodes = iterations * episodes_per_iter
    eval_interval = int(train_cfg.get("eval_interval", cfg.get("eval_interval", 0)) or 0)
    eval_split = str(train_cfg.get("eval_split", "val"))
    eval_limit = (
        train_cfg.get("val_limit")
        or train_cfg.get("eval_limit")
        or cfg.get("data", {}).get(f"{eval_split}_limit")
        or cfg.get("data", {}).get("eval_limit")
    )
    eval_limit = int(eval_limit) if eval_limit is not None else None
    penalty_deadline = float(pc.get("penalty_deadline", 1_000_000.0))
    energy_records = []
    episodes_done = 0
    next_eval_episode = eval_interval if eval_interval > 0 else None

    import torch

    def clone_checkpoint() -> Dict:
        return {
            "model": {k: v.detach().cpu().clone() for k, v in agent.model.state_dict().items()},
            "lambda_power": agent.lambda_power,
            "lambda_deadline": agent.lambda_deadline,
        }

    def rank_energy_records() -> List[Dict]:
        feasible = [r for r in energy_records if r["deadline_miss_rate"] <= ppo_cfg.epsilon_deadline_miss]
        if feasible:
            return sorted(feasible, key=lambda r: (r["energy_active_total"], r["deadline_miss_rate"], r["iteration"]))
        return sorted(energy_records, key=lambda r: (r["fallback_score"], r["energy_active_total"], r["iteration"]))

    def save_energy_topk() -> List[Dict]:
        ranked = rank_energy_records()
        if not ranked:
            return ranked
        torch.save(ranked[0]["checkpoint"], out_dir / "best_energy_pcea_ppo.pt")
        torch.save(ranked[0]["checkpoint"], out_dir / "best_energy_feasible_pcea_ppo.pt")
        for i, record in enumerate(ranked[:10], start=1):
            torch.save(record["checkpoint"], out_dir / f"top{i}_energy_pcea_ppo.pt")
        return ranked

    for it in range(iterations):
        if episodes_done >= total_episodes:
            break
        buffer.clear()
        ep_metrics = []
        episodes_this_iter = min(episodes_per_iter, total_episodes - episodes_done)
        for _ in range(episodes_this_iter):
            wf = provider.sample()
            obs = env.reset(wf)
            for step in range(max_steps):
                if env.done:
                    break
                if len(obs["mask"]) == 0 or float(obs["mask"].sum()) <= 0:
                    env._ensure_actionable_or_done()
                    obs = env._observe()
                    if env.done:
                        break
                    if len(obs["mask"]) == 0 or float(obs["mask"].sum()) <= 0:
                        raise RuntimeError("No legal action after ensure_actionable_or_done")
                action, logp, values = agent.act(obs)
                next_obs, costs, done, info = env.step(action)
                buffer.add(
                    obs=obs,
                    action=action,
                    logp=logp,
                    cost_energy=costs["energy"],
                    cost_power=costs["power"],
                    cost_deadline=costs["deadline"],
                    value_energy=values["energy"],
                    value_power=values["power"],
                    value_deadline=values["deadline"],
                    value_scalar=values["scalar"],
                    done=done,
                )
                obs = next_obs
                if done:
                    break
            ep_metrics.append(env._episode_info())
            episodes_done += 1
        stats = agent.update(buffer)
        mean_power = float(np.mean([m["power_envelope_violation"] for m in ep_metrics]))
        mean_deadline = float(np.mean([m["deadline_miss_rate"] for m in ep_metrics]))
        validation_summary = {}
        should_validate = next_eval_episode is not None and episodes_done >= next_eval_episode
        if should_validate:
            val_rows = evaluate_agent(agent, cfg, split=eval_split, seed=args.seed, limit=eval_limit, max_steps=max_steps)
            val_summary = summarize(val_rows)
            for metric in [
                "energy_active_total",
                "energy_idle_active",
                "makespan",
                "deadline_miss_rate",
                "power_envelope_violation",
                "peak_active_power",
            ]:
                validation_summary[f"validation_{metric}"] = val_summary.get(f"{metric}_mean", 0.0)
            while next_eval_episode is not None and episodes_done >= next_eval_episode:
                next_eval_episode += eval_interval
        dual_power_metric = validation_summary.get("validation_power_envelope_violation", ppo_cfg.max_power_violation_soft)
        dual_deadline_metric = validation_summary.get("validation_deadline_miss_rate", mean_deadline)
        agent.update_dual(dual_power_metric, dual_deadline_metric)
        checkpoint = clone_checkpoint()
        torch.save(checkpoint, out_dir / "last_pcea_ppo.pt")
        row = {
            "iteration": it,
            "episodes_done": episodes_done,
            "lambda_power": agent.lambda_power,
            "lambda_deadline": agent.lambda_deadline,
            "checkpoint_energy_rank": "",
            "is_best_energy": 0,
            **summarize(ep_metrics),
            **validation_summary,
            **stats,
        }
        if validation_summary:
            energy = float(validation_summary["validation_energy_active_total"])
            deadline = float(validation_summary["validation_deadline_miss_rate"])
            record_id = len(energy_records)
            energy_records.append(
                {
                    "id": record_id,
                    "iteration": it,
                    "episodes_done": episodes_done,
                    "energy_active_total": energy,
                    "deadline_miss_rate": deadline,
                    "power_envelope_violation": float(validation_summary["validation_power_envelope_violation"]),
                    "fallback_score": energy + penalty_deadline * max(0.0, deadline - ppo_cfg.epsilon_deadline_miss),
                    "checkpoint": checkpoint,
                }
            )
            ranked = save_energy_topk()
            for idx, record in enumerate(ranked, start=1):
                if record["id"] == record_id:
                    row["checkpoint_energy_rank"] = idx
                    row["is_best_energy"] = 1 if idx == 1 else 0
                    break
        train_rows.append(row)
        print(row)
        write_csv(out_dir / "train_metrics.csv", train_rows)
    checkpoint = clone_checkpoint()
    torch.save(checkpoint, out_dir / "pcea_ppo.pt")
    torch.save(checkpoint, out_dir / "last_pcea_ppo.pt")
    if energy_records:
        save_energy_topk()


def cmd_smoke(args) -> None:
    cfg = load_config(args.config)
    env = make_env(cfg, seed=args.seed)
    provider = provider_from_cfg(cfg, split="train", seed=args.seed, )
    wf = provider.sample()
    rows = []
    for name, policy in BASELINE_POLICIES.items():
        metrics = run_policy(env, wf, policy)
        metrics["method"] = name
        metrics["workflow_id"] = wf.workflow_id
        rows.append(metrics)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Power-Compute Scheduler clean CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("--config", default="configs/pcea_base.json")
    b.add_argument("--split", default="benchmark")
    b.add_argument("--methods", default=None)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--output", default=None)
    b.set_defaults(func=cmd_benchmark)
    t = sub.add_parser("train")
    t.add_argument("--config", default="configs/pcea_base.json")
    t.add_argument("--algo", default="pcea_ppo")
    t.add_argument("--iterations", type=int, default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu")
    t.add_argument("--output", default=None)
    t.set_defaults(func=cmd_train)
    s = sub.add_parser("smoke")
    s.add_argument("--config", default="configs/pcea_base.json")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_smoke)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
