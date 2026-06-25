from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from pcos.core.entities import Task
from pcos.env.scheduler_env import SchedulerEnv


@dataclass
class AERSchedule:
    method: str
    task_order: List[int]
    assignment: Dict[int, int]
    candidate_machines: Dict[int, List[int]]
    metrics: Dict[str, float]
    steps: int


@dataclass
class AERReplayResult:
    metrics: Dict[str, float]
    steps: int


@dataclass
class AERResult:
    method: str
    workflow_id: str
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    migration_count: int
    candidates_evaluated: int
    passes_done: int
    constraint_violations: int
    replay_energy_delta: float
    migration_log: List[Dict[str, object]]


def task_static_fit(task: Task, env: SchedulerEnv, machine_id: int) -> bool:
    spec = env.cluster_specs[machine_id]
    return (
        task.cpu <= spec.cpu + 1e-9
        and task.gpu <= spec.gpu + 1e-9
        and task.mem <= spec.mem + 1e-9
    )


def active_window_utilization_metrics(env: SchedulerEnv) -> Dict[str, float]:
    """Compute schedule-level active-window utilization diagnostics.

    A machine active window is the union of [task_start, task_finish + idle_timeout]
    intervals clipped to the episode makespan. This matches the active-idle
    accounting used by the environment without charging time after workflow
    completion.
    """
    makespan = max(env.task_finish.values(), default=env.current_time)
    intervals_by_machine: Dict[int, List[Tuple[float, float]]] = {}
    total_task_busy_time = 0.0
    total_cpu_busy_capacity_time = 0.0
    total_gpu_busy_capacity_time = 0.0
    for task_id, start in env.task_start.items():
        finish = env.task_finish.get(task_id)
        machine_id = env.task_machine.get(task_id)
        if finish is None or machine_id is None:
            continue
        start = float(start)
        finish = float(finish)
        busy = max(0.0, finish - start)
        total_task_busy_time += busy
        task = env.workflow.tasks[task_id]
        total_cpu_busy_capacity_time += max(0.0, float(task.cpu)) * busy
        total_gpu_busy_capacity_time += max(0.0, float(task.gpu)) * busy
        active_end = min(float(makespan), finish + env.config.idle_timeout_s)
        if active_end > start + 1e-12:
            intervals_by_machine.setdefault(machine_id, []).append((start, active_end))

    total_active_window_time = 0.0
    total_cpu_active_capacity_time = 0.0
    total_gpu_active_capacity_time = 0.0
    for machine_id, intervals in intervals_by_machine.items():
        intervals.sort()
        merged: List[Tuple[float, float]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1] + 1e-12:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        active_window = sum(max(0.0, end - start) for start, end in merged)
        total_active_window_time += active_window
        spec = env.cluster_specs[machine_id]
        total_cpu_active_capacity_time += active_window * max(0.0, float(spec.cpu))
        total_gpu_active_capacity_time += active_window * max(0.0, float(spec.gpu))

    return {
        "total_task_busy_time": total_task_busy_time,
        "total_active_window_time": total_active_window_time,
        "active_server_utilization": total_task_busy_time / max(total_active_window_time, 1e-9),
        "total_cpu_busy_capacity_time": total_cpu_busy_capacity_time,
        "total_gpu_busy_capacity_time": total_gpu_busy_capacity_time,
        "total_cpu_active_capacity_time": total_cpu_active_capacity_time,
        "total_gpu_active_capacity_time": total_gpu_active_capacity_time,
        "avg_cpu_utilization": total_cpu_busy_capacity_time / max(total_cpu_active_capacity_time, 1e-9),
        "avg_gpu_utilization": total_gpu_busy_capacity_time / max(total_gpu_active_capacity_time, 1e-9),
    }


def add_active_window_diagnostics(env: SchedulerEnv, metrics: Dict[str, float]) -> Dict[str, float]:
    metrics.update(active_window_utilization_metrics(env))
    return metrics


def capture_schedule(
    env: SchedulerEnv,
    workflow,
    method: str,
    action_selector: Callable[[SchedulerEnv, Dict[str, np.ndarray]], int],
    max_steps: int = 10000,
) -> AERSchedule:
    obs = env.reset(workflow)
    task_order: List[int] = []
    assignment: Dict[int, int] = {}
    candidate_machines: Dict[int, List[int]] = {}
    steps = 0
    while not env.done and steps < max_steps:
        if len(obs["mask"]) == 0 or float(np.asarray(obs["mask"], dtype=np.float32).sum()) <= 0.0:
            env._ensure_actionable_or_done()
            obs = env._observe()
            if env.done:
                break
        pairs = env.pair_actions()
        action = int(action_selector(env, obs))
        if action < len(pairs):
            task_id, machine_id = pairs[action]
            candidates = []
            for candidate_task, candidate_machine in pairs:
                if candidate_task != task_id:
                    continue
                info = env._pair_energy_estimate(candidate_task, candidate_machine)
                score = float(info["energy_delta"]) + float(info["active_idle_delta"])
                candidates.append((score, float(info["eft"]), candidate_machine))
            candidate_machines[task_id] = [mid for _, _, mid in sorted(candidates)]
            task_order.append(task_id)
            assignment[task_id] = machine_id
        obs, _, done, _ = env.step(action)
        steps += 1
        if done:
            break
    if not env.done:
        raise RuntimeError(f"AER schedule capture exceeded max_steps={max_steps}: {env._diagnostic_action_state()}")
    metrics = env._episode_info()
    add_active_window_diagnostics(env, metrics)
    metrics["method"] = method
    metrics["workflow_id"] = workflow.workflow_id
    metrics["steps"] = steps
    return AERSchedule(
        method=method,
        task_order=task_order,
        assignment=assignment,
        candidate_machines=candidate_machines,
        metrics=metrics,
        steps=steps,
    )


def replay_allocation(
    env: SchedulerEnv,
    workflow,
    task_order: List[int],
    assignment: Dict[int, int],
    max_steps: int = 10000,
) -> AERReplayResult:
    env.reset(workflow)
    order_rank = {task_id: i for i, task_id in enumerate(task_order)}
    steps = 0
    while not env.done and steps < max_steps:
        pairs = env.pair_actions()
        eligible: List[Tuple[int, int, int]] = []
        for action_id, (task_id, machine_id) in enumerate(pairs):
            target_machine = assignment.get(task_id)
            if target_machine is not None and target_machine == machine_id:
                eligible.append((order_rank.get(task_id, 10**9), action_id, task_id))
        if not eligible:
            if env.done:
                break
            if env.running:
                env._advance_to_next_event()
                continue
            raise RuntimeError(f"AER replay found no eligible pair. State: {env._diagnostic_action_state()}")
        _, action_id, _ = min(eligible, key=lambda x: (x[0], x[1]))
        env.step(action_id)
        steps += 1
    if not env.done:
        raise RuntimeError(f"AER replay exceeded max_steps={max_steps}: {env._diagnostic_action_state()}")
    metrics = env._episode_info()
    add_active_window_diagnostics(env, metrics)
    metrics["steps"] = steps
    return AERReplayResult(metrics=metrics, steps=steps)


def aer_accepts(
    candidate: Dict[str, float],
    current: Dict[str, float],
    original: Dict[str, float],
    objective_metric: str,
    energy_tol: float,
    makespan_tol: float,
    deadline_tol: float,
    enforce_makespan: bool = True,
    enforce_deadline: bool = True,
    require_active_energy_improvement: bool = True,
) -> bool:
    if metric_value(candidate, objective_metric) >= metric_value(current, objective_metric) - energy_tol:
        return False
    if require_active_energy_improvement and candidate["energy_active_total"] >= current["energy_active_total"] - energy_tol:
        return False
    if enforce_makespan:
        if candidate["makespan"] > current["makespan"] + makespan_tol:
            return False
        if candidate["makespan"] > original["makespan"] + makespan_tol:
            return False
    if enforce_deadline:
        if candidate["deadline_miss_rate"] > current["deadline_miss_rate"] + deadline_tol:
            return False
        if candidate["deadline_miss_rate"] > original["deadline_miss_rate"] + deadline_tol:
            return False
    return True


def metric_value(metrics: Dict[str, float], name: str) -> float:
    if name == "non_idle_energy":
        return (
            float(metrics.get("energy_cpu", 0.0))
            + float(metrics.get("energy_gpu", 0.0))
            + float(metrics.get("energy_comm", 0.0))
        )
    if name == "idle_energy":
        return float(metrics.get("energy_idle_active", 0.0))
    return float(metrics.get(name, metrics.get("energy_active_total", 0.0)))


def classify_migration(before: Dict[str, float], after: Dict[str, float]) -> str:
    server_delta = float(before.get("server_activated_count", 0.0)) - float(after.get("server_activated_count", 0.0))
    makespan_delta = float(after.get("makespan", 0.0)) - float(before.get("makespan", 0.0))
    idle_delta = float(after.get("energy_idle_active", 0.0)) - float(before.get("energy_idle_active", 0.0))
    active_window_delta = float(after.get("total_active_window_time", 0.0)) - float(before.get("total_active_window_time", 0.0))
    non_idle_before = metric_value(before, "non_idle_energy")
    non_idle_after = metric_value(after, "non_idle_energy")
    non_idle_delta = non_idle_after - non_idle_before

    if server_delta > 0.5:
        return "avoid_activation"
    if makespan_delta < -1e-9:
        return "shorten_active_tail"
    if active_window_delta < -1e-9 and idle_delta < -1e-9:
        return "reduce_idle_gap"
    if idle_delta < -1e-9:
        return "fill_existing_active_window"
    if non_idle_delta < -1e-9:
        return "energy_lower_machine"
    return "other"


class ActiveWindowEnergyReallocator:
    def __init__(
        self,
        env_factory: Callable[[], SchedulerEnv],
        max_steps: int = 10000,
        max_passes: int = 2,
        candidate_top_k: int = 0,
        energy_tol: float = 1e-6,
        makespan_tol: float = 1e-9,
        deadline_tol: float = 1e-12,
        objective_metric: str = "energy_active_total",
        enforce_makespan: bool = True,
        enforce_deadline: bool = True,
        require_active_energy_improvement: bool = True,
    ) -> None:
        self.env_factory = env_factory
        self.max_steps = max_steps
        self.max_passes = max_passes
        self.candidate_top_k = candidate_top_k
        self.energy_tol = energy_tol
        self.makespan_tol = makespan_tol
        self.deadline_tol = deadline_tol
        self.objective_metric = objective_metric
        self.enforce_makespan = enforce_makespan
        self.enforce_deadline = enforce_deadline
        self.require_active_energy_improvement = require_active_energy_improvement

    def apply(self, workflow, schedule: AERSchedule) -> AERResult:
        current_assignment = dict(schedule.assignment)
        original_replay = replay_allocation(
            self.env_factory(),
            workflow,
            schedule.task_order,
            current_assignment,
            max_steps=self.max_steps,
        )
        original_metrics = original_replay.metrics
        current_metrics = dict(original_metrics)
        migration_count = 0
        candidates_evaluated = 0
        constraint_violations = 0
        passes_done = 0
        migration_log: List[Dict[str, object]] = []

        probe_env = self.env_factory()
        for pass_id in range(self.max_passes):
            passes_done = pass_id + 1
            improved = False
            for task_id in schedule.task_order:
                task = workflow.tasks[task_id]
                old_machine = current_assignment[task_id]
                best_machine: Optional[int] = None
                best_metrics: Optional[Dict[str, float]] = None
                if self.candidate_top_k > 0:
                    candidate_machines = list(schedule.candidate_machines.get(task_id, []))[: self.candidate_top_k]
                    if old_machine not in candidate_machines:
                        candidate_machines.append(old_machine)
                else:
                    candidate_machines = list(range(len(probe_env.cluster_specs)))
                for machine_id in candidate_machines:
                    if machine_id == old_machine or not task_static_fit(task, probe_env, machine_id):
                        continue
                    trial_assignment = dict(current_assignment)
                    trial_assignment[task_id] = machine_id
                    candidates_evaluated += 1
                    try:
                        trial = replay_allocation(
                            self.env_factory(),
                            workflow,
                            schedule.task_order,
                            trial_assignment,
                            max_steps=self.max_steps,
                        )
                    except RuntimeError:
                        constraint_violations += 1
                        continue
                    if aer_accepts(
                        trial.metrics,
                        current_metrics,
                        original_metrics,
                        objective_metric=self.objective_metric,
                        energy_tol=self.energy_tol,
                        makespan_tol=self.makespan_tol,
                        deadline_tol=self.deadline_tol,
                        enforce_makespan=self.enforce_makespan,
                        enforce_deadline=self.enforce_deadline,
                        require_active_energy_improvement=self.require_active_energy_improvement,
                    ):
                        if best_metrics is None or metric_value(trial.metrics, self.objective_metric) < metric_value(best_metrics, self.objective_metric):
                            best_machine = machine_id
                            best_metrics = trial.metrics
                if best_machine is not None and best_metrics is not None:
                    task = workflow.tasks[task_id]
                    old_spec = probe_env.cluster_specs[old_machine]
                    new_spec = probe_env.cluster_specs[best_machine]
                    migration_log.append(
                        {
                            "workflow_id": getattr(workflow, "workflow_id", ""),
                            "component_workflow_id": workflow.metadata.get("component_ids", {}).get(task_id, "")
                            if getattr(workflow, "metadata", None)
                            else "",
                            "pass_id": pass_id,
                            "migration_index": migration_count,
                            "task_id": task_id,
                            "profile_name": task.profile_name,
                            "cpu": float(task.cpu),
                            "gpu": float(task.gpu),
                            "mem": float(task.mem),
                            "base_duration": float(task.base_duration),
                            "from_machine": old_machine,
                            "to_machine": best_machine,
                            "from_machine_type": old_spec.name,
                            "to_machine_type": new_spec.name,
                            "from_machine_id": old_spec.machine_id,
                            "to_machine_id": new_spec.machine_id,
                            "energy_before": float(current_metrics.get("energy_active_total", 0.0)),
                            "energy_after": float(best_metrics.get("energy_active_total", 0.0)),
                            "energy_delta": float(best_metrics.get("energy_active_total", 0.0))
                            - float(current_metrics.get("energy_active_total", 0.0)),
                            "idle_energy_before": float(current_metrics.get("energy_idle_active", 0.0)),
                            "idle_energy_after": float(best_metrics.get("energy_idle_active", 0.0)),
                            "idle_energy_delta": float(best_metrics.get("energy_idle_active", 0.0))
                            - float(current_metrics.get("energy_idle_active", 0.0)),
                            "makespan_before": float(current_metrics.get("makespan", 0.0)),
                            "makespan_after": float(best_metrics.get("makespan", 0.0)),
                            "makespan_delta": float(best_metrics.get("makespan", 0.0))
                            - float(current_metrics.get("makespan", 0.0)),
                            "server_activated_before": float(current_metrics.get("server_activated_count", 0.0)),
                            "server_activated_after": float(best_metrics.get("server_activated_count", 0.0)),
                            "active_window_before": float(current_metrics.get("total_active_window_time", 0.0)),
                            "active_window_after": float(best_metrics.get("total_active_window_time", 0.0)),
                            "migration_type": classify_migration(current_metrics, best_metrics),
                        }
                    )
                    current_assignment[task_id] = best_machine
                    current_metrics = best_metrics
                    migration_count += 1
                    improved = True
            if not improved:
                break

        return AERResult(
            method=schedule.method,
            workflow_id=workflow.workflow_id,
            before_metrics=schedule.metrics,
            after_metrics=current_metrics,
            migration_count=migration_count,
            candidates_evaluated=candidates_evaluated,
            passes_done=passes_done,
            constraint_violations=constraint_violations,
            replay_energy_delta=current_metrics["energy_active_total"] - original_metrics["energy_active_total"],
            migration_log=migration_log,
        )
