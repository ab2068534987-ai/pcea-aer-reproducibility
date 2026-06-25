from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from pcos.core.entities import estimate_duration
from pcos.env.scheduler_env import SchedulerEnv

PolicyFn = Callable[[SchedulerEnv], int]


def fcfs_policy(env: SchedulerEnv) -> int:
    pairs = env.pair_actions()
    if not pairs:
        return 0
    # pair_actions are ready-order sorted; choose first feasible pair.
    return 0


def heft_policy(env: SchedulerEnv) -> int:
    pairs = env.pair_actions()
    if not pairs:
        return 0
    best_idx = 0
    best_key = None
    for i, (tid, mid) in enumerate(pairs):
        task = env.workflow.tasks[tid]
        start = env._earliest_start_on_machine(tid, mid)
        eft = start + estimate_duration(task, env.machines[mid].spec)
        key = (-task.upward_rank, eft, mid)
        if best_key is None or key < best_key:
            best_key = key
            best_idx = i
    return best_idx


def min_energy_policy(env: SchedulerEnv) -> int:
    pairs = env.pair_actions()
    if not pairs:
        return 0
    best_idx = 0
    best_score = float("inf")
    for i, (tid, mid) in enumerate(pairs):
        task = env.workflow.tasks[tid]
        m = env.machines[mid]
        dur = estimate_duration(task, m.spec)
        # Approximate active power if the task runs on this machine.
        before = m.power_breakdown(env.current_time, env.config.idle_timeout_s).total
        u_cpu = min(1.0, (m.used_cpu + task.cpu) / max(m.spec.cpu, 1e-9))
        u_gpu = min(1.0, (m.used_gpu + task.gpu) / max(m.spec.gpu, 1e-9)) if m.spec.gpu > 0 else 0.0
        from pcos.core.power import coupled_power
        after = coupled_power(m.spec.p_idle, m.spec.p_cpu0, m.spec.p_gpu0, m.spec.k1, m.spec.k2, u_cpu, u_gpu, True).total
        delta_e = max(after, before) * dur
        comm = 0.0
        for p in task.preds:
            pm = env.task_machine.get(p)
            if pm is not None and pm != mid:
                comm += env.workflow.tasks[p].data_size * env.config.energy_per_MB
        activation_penalty = 0.15 * m.spec.p_idle * min(dur, env.config.idle_timeout_s) if not m.activated else 0.0
        score = delta_e + comm + activation_penalty - 0.01 * task.upward_rank
        if score < best_score:
            best_score = score
            best_idx = i
    return best_idx


def run_policy(env: SchedulerEnv, workflow, policy: PolicyFn, max_steps: int = 10000) -> Dict[str, float]:
    env.reset(workflow)
    steps = 0
    info = {}
    while not env.done and steps < max_steps:
        pairs = env.pair_actions()
        if not pairs:
            env._advance_to_next_event()
            continue
        action = policy(env)
        _, _, _, info = env.step(action)
        steps += 1
    info = env._episode_info()
    info["steps"] = steps
    return info


BASELINE_POLICIES: Dict[str, PolicyFn] = {
    "fcfs": fcfs_policy,
    "heft": heft_policy,
    "min_energy": min_energy_policy,
}
