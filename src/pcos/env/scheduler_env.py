from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pcos.core.entities import (
    MachineSpec,
    MachineState,
    RunningTask,
    Workflow,
    default_cluster_specs,
    estimate_duration,
)
from pcos.env.power_envelope import PowerEnvelopeProvider

DEFER_ACTION = -1


@dataclass
class EnvConfig:
    idle_timeout_s: float = 300.0
    bandwidth_MBps: float = 250.0
    latency_s: float = 0.01
    energy_per_MB: float = 0.02
    max_defer_s: float = 300.0
    safe_slack_margin_s: float = 30.0
    headroom_margin_ratio: float = 0.05
    require_power_pressure: bool = True
    enable_defer: bool = True
    energy_norm: float = 10000.0
    performance_cost_coef: float = 0.20
    urgent_slack_ratio: float = 0.15
    min_defer_power_pressure_ratio: float = 0.0
    cpeg_enabled: bool = False
    cpeg_criticality_threshold: float = 0.60
    cpeg_eft_regret_ratio: float = 0.10
    cpeg_power_pressure_margin: float = 0.02
    cpeg_cost_coef: float = 0.50
    anti_idle_enabled: bool = False
    anti_idle_eft_regret_ratio: float = 0.02
    anti_idle_power_pressure_margin: float = 0.01
    anti_idle_cost_coef: float = 0.60
    anti_idle_idle_cost_coef: float = 0.50
    energy_guard_enabled: bool = False
    critical_eft_window_ratio: float = 0.03
    noncritical_eft_window_ratio: float = 0.08
    energy_guard_cost_coef: float = 0.0
    energy_guard_include_idle_tail: bool = False
    active_energy_cost_coef: float = 1.0
    idle_cost_coef: float = 0.0
    activation_guard_enabled: bool = False
    activation_eft_gain_required: float = 0.05
    activation_guard_max_criticality: float = 1.0
    activation_guard_max_ready_width: int = 1_000_000
    activation_guard_cost_coef: float = 0.0


class SchedulerEnv:
    """Event-driven heterogeneous CPU-GPU DAG scheduler with PCEA extensions.

    Action ids 0..num_pairs-1 correspond to flattened (ready task, machine) pairs.
    If enabled, the last action id is a global DEFER action.
    """

    def __init__(
        self,
        cluster: Optional[Sequence[MachineSpec]] = None,
        envelope_provider: Optional[PowerEnvelopeProvider] = None,
        config: Optional[EnvConfig] = None,
        seed: int = 0,
    ):
        self.cluster_specs = list(cluster or default_cluster_specs())
        self.config = config or EnvConfig()
        self.cluster_full_power = sum(s.full_power() for s in self.cluster_specs)
        self.envelope_provider = envelope_provider or PowerEnvelopeProvider(self.cluster_full_power, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.workflow: Optional[Workflow] = None
        self.machines: List[MachineState] = []
        self.reset_state_vars()

    def reset_state_vars(self) -> None:
        self.current_time = 0.0
        self.phase_s = 0.0
        self.started: set[int] = set()
        self.completed: set[int] = set()
        self.task_start: Dict[int, float] = {}
        self.task_finish: Dict[int, float] = {}
        self.task_machine: Dict[int, int] = {}
        self.running: Dict[int, RunningTask] = {}
        self.ready: List[int] = []
        self.energy_active_total = 0.0
        self.energy_active_compute = 0.0
        self.energy_idle_active = 0.0
        self.energy_cpu = 0.0
        self.energy_gpu = 0.0
        self.energy_interaction_signed = 0.0
        self.energy_comm = 0.0
        self.energy_always_on_total = 0.0
        self.power_violation_integral = 0.0
        self.power_violation_mean = 0.0
        self.peak_active_power = 0.0
        self.defer_count = 0
        self.defer_time_total = 0.0
        self.cross_machine_edge_count = 0
        self.cross_machine_data_mb = 0.0
        self.task_criticality: Dict[int, float] = {}
        self.cpeg_penalized_pair_count = 0
        self.cpeg_slow_action_count = 0
        self.cpeg_cost_total = 0.0
        self.anti_idle_penalized_pair_count = 0
        self.anti_idle_slow_action_count = 0
        self.anti_idle_cost_total = 0.0
        self.energy_guard_penalized_pair_count = 0
        self.energy_guard_selected_count = 0
        self.energy_guard_selected_preferred_count = 0
        self.energy_guard_selected_regret_total = 0.0
        self.energy_guard_cost_total = 0.0
        self.activation_guard_penalized_pair_count = 0
        self.activation_guard_selected_count = 0
        self.activation_guard_selected_penalized_count = 0
        self.activation_guard_selected_gain_total = 0.0
        self.activation_guard_cost_total = 0.0
        self._last_info: Dict[str, float] = {}

    def reset(self, workflow: Workflow, phase_s: Optional[float] = None):
        self.workflow = workflow
        self.machines = [MachineState(spec=s, index=i) for i, s in enumerate(self.cluster_specs)]
        self.reset_state_vars()
        self.workflow = workflow
        submit_time = float(workflow.metadata.get("submit_time", 0.0)) if workflow.metadata else 0.0
        self.phase_s = self.envelope_provider.sample_phase(submit_time) if phase_s is None else float(phase_s)
        self.task_criticality = self._compute_task_criticality()
        self._refresh_ready()
        self._ensure_actionable_or_done()
        return self._observe()

    @property
    def done(self) -> bool:
        return self.workflow is not None and len(self.completed) == len(self.workflow.tasks)

    def pair_actions(self) -> List[Tuple[int, int]]:
        actions: List[Tuple[int, int]] = []
        for tid in self.ready:
            task = self.workflow.tasks[tid]
            for mid, m in enumerate(self.machines):
                if m.can_fit(task):
                    actions.append((tid, mid))
        return actions

    def num_actions(self) -> int:
        return len(self.pair_actions()) + (1 if self._defer_allowed() else 0)

    def legal_action_count(self) -> int:
        return int(len(self.pair_actions()) + (1 if self._defer_allowed() else 0))

    def has_legal_action(self) -> bool:
        return self.legal_action_count() > 0

    def _diagnostic_action_state(self) -> dict:
        workflow_id = self.workflow.workflow_id if self.workflow is not None else None
        tasks = self.workflow.tasks if self.workflow is not None else []
        ready_tasks = []
        for tid in self.ready:
            task = self.workflow.tasks[tid]
            ready_tasks.append(
                {
                    "task_id": tid,
                    "cpu": task.cpu,
                    "gpu": task.gpu,
                    "mem": task.mem,
                    "profile_name": task.profile_name,
                    "profile_id": task.profile_id,
                    "deadline": task.deadline,
                }
            )
        machines = []
        for machine in self.machines:
            machines.append(
                {
                    "machine_id": machine.spec.machine_id or str(machine.index),
                    "name": machine.spec.name,
                    "free_cpu": machine.spec.cpu - machine.used_cpu,
                    "free_gpu": machine.spec.gpu - machine.used_gpu,
                    "free_mem": machine.spec.mem - machine.used_mem,
                    "used_cpu": machine.used_cpu,
                    "used_gpu": machine.used_gpu,
                    "used_mem": machine.used_mem,
                    "running": sorted(machine.running.keys()),
                }
            )
        return {
            "workflow_id": workflow_id,
            "current_time": self.current_time,
            "ready": list(self.ready),
            "running": sorted(self.running.keys()),
            "completed_count": len(self.completed),
            "num_tasks": len(tasks),
            "ready_tasks": ready_tasks,
            "machines": machines,
        }

    def _ensure_actionable_or_done(self) -> None:
        if self.workflow is None:
            return
        guard = len(self.workflow.tasks) + 10
        for _ in range(guard):
            self._complete_finished_tasks()
            self._refresh_ready()
            if self.done:
                return
            if self.legal_action_count() > 0:
                return
            if self.running:
                before = self.current_time
                self._advance_to_next_event()
                if self.current_time <= before + 1e-9 and self.legal_action_count() == 0:
                    raise RuntimeError(f"SchedulerEnv cannot advance to an actionable state: {self._diagnostic_action_state()}")
                continue
            raise RuntimeError(f"SchedulerEnv deadlock: no legal actions, no running tasks, not done. State: {self._diagnostic_action_state()}")
        raise RuntimeError(f"SchedulerEnv actionability guard exceeded. State: {self._diagnostic_action_state()}")

    def action_mask(self) -> np.ndarray:
        pairs = self.pair_actions()
        mask = np.ones(len(pairs) + (1 if self._defer_allowed() else 0), dtype=np.float32)
        return mask

    def step(self, action_id: int):
        if self.workflow is None:
            raise RuntimeError("Call reset() before step().")
        old_time = self.current_time
        prev_energy = self.energy_active_total
        prev_idle = self.energy_idle_active
        prev_power_cost = self.power_violation_integral
        prev_deadline_cost = self._deadline_cost_dense()
        pairs = self.pair_actions()
        is_defer = False
        cpeg_cost = 0.0
        selected_cpeg_slow = 0.0
        anti_idle_cost = 0.0
        selected_anti_idle_slow = 0.0
        energy_guard_cost = 0.0
        selected_energy_guard_preferred = 0.0
        selected_energy_guard_regret = 0.0
        activation_guard_cost = 0.0
        selected_activation_guard_penalized = 0.0
        selected_activation_gain_ratio = 0.0
        if action_id < len(pairs):
            tid, mid = pairs[action_id]
            cpeg_info = self._cpeg_pair_info(tid, mid)
            anti_idle_info = self._anti_idle_pair_info(tid, mid, cpeg_info)
            energy_guard_info = self._energy_guard_pair_info(tid, mid)
            activation_guard_info = self._activation_guard_pair_info(tid, mid, energy_guard_info)
            selected_cpeg_slow = float(cpeg_info["slow_critical"])
            if selected_cpeg_slow:
                cpeg_cost = self.config.cpeg_cost_coef * cpeg_info["criticality"] * cpeg_info["eft_regret_ratio"]
                self.cpeg_slow_action_count += 1
                self.cpeg_cost_total += cpeg_cost
            selected_anti_idle_slow = float(anti_idle_info["slow_finish"])
            if selected_anti_idle_slow:
                anti_idle_cost = self.config.anti_idle_cost_coef * anti_idle_info["eft_regret_ratio"]
                self.anti_idle_slow_action_count += 1
                self.anti_idle_cost_total += anti_idle_cost
            selected_energy_guard_preferred = float(energy_guard_info["preferred"])
            selected_energy_guard_regret = float(energy_guard_info["energy_regret_ratio"])
            if self.config.energy_guard_enabled:
                self.energy_guard_selected_count += 1
                self.energy_guard_selected_preferred_count += int(selected_energy_guard_preferred)
                self.energy_guard_selected_regret_total += selected_energy_guard_regret
                energy_guard_cost = self.config.energy_guard_cost_coef * selected_energy_guard_regret
                self.energy_guard_cost_total += energy_guard_cost
            selected_activation_guard_penalized = float(activation_guard_info["penalized"])
            selected_activation_gain_ratio = float(activation_guard_info["eft_gain_ratio"])
            if self.config.activation_guard_enabled:
                self.activation_guard_selected_count += 1
                self.activation_guard_selected_penalized_count += int(selected_activation_guard_penalized)
                self.activation_guard_selected_gain_total += selected_activation_gain_ratio
                activation_guard_cost = self.config.activation_guard_cost_coef * max(
                    0.0, self.config.activation_eft_gain_required - selected_activation_gain_ratio
                )
                self.activation_guard_cost_total += activation_guard_cost
            self._dispatch(tid, mid)
        elif self._defer_allowed():
            is_defer = True
            self._safe_defer()
        else:
            # Invalid action fallback: if no legal policy action, advance event.
            self._advance_to_next_event()
        self._ensure_actionable_or_done()
        step_time_delta = max(0.0, self.current_time - old_time)
        deadline_risk_cost = max(0.0, self._deadline_cost_dense() - prev_deadline_cost)
        performance_cost = self.config.performance_cost_coef * step_time_delta / self._makespan_target_for_cost()
        idle_delta = max(0.0, self.energy_idle_active - prev_idle)
        active_idle_cost = 0.0
        if self.config.anti_idle_enabled and not self.config.energy_guard_enabled:
            active_idle_cost = self.config.anti_idle_idle_cost_coef * idle_delta / max(self.config.energy_norm, 1e-9)
        energy_delta = max(0.0, self.energy_active_total - prev_energy)
        energy_step_cost = (
            self.config.active_energy_cost_coef * energy_delta / max(self.config.energy_norm, 1e-9)
            + self.config.idle_cost_coef * idle_delta / max(self.config.energy_norm, 1e-9)
            + energy_guard_cost
            + activation_guard_cost
        )
        obs = self._observe()
        done = self.done
        costs = {
            "energy": energy_step_cost,
            "power": (self.power_violation_integral - prev_power_cost),
            "deadline": deadline_risk_cost + performance_cost + cpeg_cost + anti_idle_cost + active_idle_cost,
        }
        info = self._episode_info()
        info.update(costs)
        info["is_defer"] = float(is_defer)
        info["step_time_delta"] = step_time_delta
        info["deadline_risk_cost"] = deadline_risk_cost
        info["performance_cost"] = performance_cost
        info["cpeg_cost"] = cpeg_cost
        info["cpeg_selected_slow_critical"] = selected_cpeg_slow
        info["anti_idle_cost"] = anti_idle_cost
        info["active_idle_cost"] = active_idle_cost
        info["anti_idle_selected_slow_finish"] = selected_anti_idle_slow
        info["energy_guard_cost"] = energy_guard_cost
        info["energy_guard_selected_preferred"] = selected_energy_guard_preferred
        info["energy_guard_selected_regret"] = selected_energy_guard_regret
        info["activation_guard_cost"] = activation_guard_cost
        info["activation_guard_selected_penalized"] = selected_activation_guard_penalized
        info["activation_guard_selected_gain_ratio"] = selected_activation_gain_ratio
        return obs, costs, done, info

    def _dispatch(self, tid: int, mid: int) -> None:
        task = self.workflow.tasks[tid]
        machine = self.machines[mid]
        start = self._earliest_start_on_machine(tid, mid)
        if start > self.current_time:
            self._integrate_until(start)
            self.current_time = start
        # Charge cross-machine communication energy once when the child is dispatched.
        self.energy_comm += self._communication_cost_for_start(tid, mid)
        self.energy_active_total = self.energy_active_compute + self.energy_comm
        duration = estimate_duration(task, machine.spec)
        end = self.current_time + duration
        rt = RunningTask(tid, mid, self.current_time, end, task.cpu, task.gpu, task.mem)
        machine.assign(rt)
        self.running[tid] = rt
        self.started.add(tid)
        self.task_start[tid] = self.current_time
        self.task_machine[tid] = mid
        if tid in self.ready:
            self.ready.remove(tid)

    def _earliest_start_on_machine(self, tid: int, mid: int) -> float:
        t = self.current_time
        task = self.workflow.tasks[tid]
        for p in task.preds:
            if p in self.task_finish:
                parent_finish = self.task_finish[p]
                parent_m = self.task_machine.get(p, mid)
                if parent_m != mid:
                    data_mb = max(0.0, self.workflow.tasks[p].data_size)
                    t = max(t, parent_finish + data_mb / max(self.config.bandwidth_MBps, 1e-9) + self.config.latency_s)
                else:
                    t = max(t, parent_finish)
        return t

    def _safe_defer(self) -> None:
        target = self._safe_defer_target()
        if target <= self.current_time + 1e-9:
            return
        dt = target - self.current_time
        self.defer_count += 1
        self.defer_time_total += dt
        self._integrate_until(target)
        self.current_time = target
        self._complete_finished_tasks()
        self._refresh_ready()

    def _advance_to_next_event(self) -> None:
        times = [rt.end_time for rt in self.running.values() if rt.end_time > self.current_time + 1e-9]
        if not times:
            return
        target = min(times)
        self._integrate_until(target)
        self.current_time = target
        self._complete_finished_tasks()
        self._refresh_ready()

    def _integrate_until(self, target_time: float) -> None:
        while self.current_time < target_time - 1e-9:
            next_env = self.envelope_provider.next_change_after(self.current_time, self.phase_s)
            next_finish = min([rt.end_time for rt in self.running.values() if rt.end_time > self.current_time + 1e-9] or [target_time])
            t1 = min(target_time, next_env, next_finish)
            if t1 <= self.current_time + 1e-9:
                t1 = min(target_time, self.current_time + 1e-6)
            dt = t1 - self.current_time
            self._integrate_interval(dt)
            self.current_time = t1
            self._complete_finished_tasks()

    def _integrate_interval(self, dt: float) -> None:
        if dt <= 0:
            return
        active_power = 0.0
        always_power = 0.0
        idle_e = cpu_e = gpu_e = inter_e = 0.0
        for m in self.machines:
            pb = m.power_breakdown(self.current_time, self.config.idle_timeout_s)
            active_power += pb.total
            always_pb = m.spec.full_power() if False else m.spec.p_idle  # sensitivity: idle-only always-on baseline
            always_power += always_pb
            idle_e += pb.idle * dt
            cpu_e += pb.cpu * dt
            gpu_e += pb.gpu * dt
            inter_e += pb.interaction_signed * dt
        self.energy_idle_active += idle_e
        self.energy_cpu += cpu_e
        self.energy_gpu += gpu_e
        self.energy_interaction_signed += inter_e
        self.energy_active_compute += idle_e + cpu_e + gpu_e + inter_e
        self.energy_active_total = self.energy_active_compute + self.energy_comm
        self.energy_always_on_total += always_power * dt
        envelope = self.envelope_provider.value(self.current_time, self.phase_s)
        violation = max(0.0, (active_power - envelope) / max(self.cluster_full_power, 1e-9))
        self.power_violation_integral += (violation**2) * dt / max(self.config.max_defer_s, 1.0)
        self.peak_active_power = max(self.peak_active_power, active_power)

    def _complete_finished_tasks(self) -> None:
        finished = [tid for tid, rt in self.running.items() if rt.end_time <= self.current_time + 1e-9]
        for tid in finished:
            rt = self.running.pop(tid)
            self.machines[rt.machine_id].release(tid)
            self.completed.add(tid)
            self.task_finish[tid] = rt.end_time
            # communication energy is charged when child becomes ready/starts across machines; charge here for outgoing edges whose child already has placement too.
        # Charge communication energy at child dispatch time through _communication_cost_for_start.

    def _refresh_ready(self) -> None:
        self.ready = []
        if self.workflow is None:
            return
        for t in self.workflow.tasks:
            if t.id in self.started or t.id in self.completed:
                continue
            if all(p in self.completed for p in t.preds):
                self.ready.append(t.id)
        self.ready.sort(key=lambda i: (-self.workflow.tasks[i].upward_rank, i))

    def _communication_cost_for_start(self, tid: int, mid: int) -> float:
        cost = 0.0
        task = self.workflow.tasks[tid]
        for p in task.preds:
            pm = self.task_machine.get(p)
            if pm is not None and pm != mid:
                data_mb = max(0.0, self.workflow.tasks[p].data_size)
                cost += data_mb * self.config.energy_per_MB
                self.cross_machine_edge_count += 1
                self.cross_machine_data_mb += data_mb
        return cost

    def _defer_allowed(self) -> bool:
        if not self.config.enable_defer or self.workflow is None or self.done:
            return False
        if not self.pair_actions():
            return False
        target = self._safe_defer_target()
        if target <= self.current_time + 1e-6:
            return False
        if target - self.current_time > self.config.max_defer_s + 1e-9:
            return False
        latest_start = self._latest_safe_start_without_margin()
        if latest_start is None or latest_start - self.current_time < self.config.safe_slack_margin_s - 1e-9:
            return False
        if self._has_urgent_ready_task():
            return False
        if not self.config.require_power_pressure:
            return True
        active = self._current_active_power()
        envelope = self.envelope_provider.value(self.current_time, self.phase_s)
        if (
            self.config.cpeg_enabled
            and self._max_ready_criticality() >= self.config.cpeg_criticality_threshold
            and active < envelope
        ):
            return False
        headroom_margin = self.config.headroom_margin_ratio * self.cluster_full_power
        if active < envelope - headroom_margin:
            return False
        future = self.envelope_provider.value(min(target, self.current_time + self.config.max_defer_s), self.phase_s)
        future_widening = future > envelope + headroom_margin
        current_violation = max(0.0, (active - envelope) / max(self.cluster_full_power, 1e-9))
        power_pressure = max(0.0, (active - (envelope - headroom_margin)) / max(self.cluster_full_power, 1e-9))
        if current_violation <= 1e-12 and power_pressure <= self.config.min_defer_power_pressure_ratio:
            return False
        if not self.running and not future_widening:
            return False
        return True

    def _safe_defer_target(self) -> float:
        if self.workflow is None:
            return self.current_time
        candidates = [self.current_time + self.config.max_defer_s]
        if self.running:
            candidates.append(min(rt.end_time for rt in self.running.values() if rt.end_time > self.current_time + 1e-9))
        candidates.append(self.envelope_provider.next_change_after(self.current_time, self.phase_s))
        latest = self._latest_safe_start_boundary()
        if latest is not None:
            candidates.append(latest)
        return max(self.current_time, min(candidates))

    def _latest_safe_start_boundary(self) -> Optional[float]:
        latest = self._latest_safe_start_without_margin()
        if latest is None:
            return None
        return latest - self.config.safe_slack_margin_s

    def _latest_safe_start_without_margin(self) -> Optional[float]:
        if self.workflow is None or not self.ready:
            return None
        bounds = []
        for tid in self.ready:
            task = self.workflow.tasks[tid]
            feasible_specs = [m.spec for m in self.machines if m.can_fit(task)]
            if not feasible_specs:
                continue
            best = min(estimate_duration(task, s) for s in feasible_specs)
            deadline = float(task.deadline or self.workflow.makespan_target or (self.current_time + best))
            bounds.append(deadline - best)
        return min(bounds) if bounds else None

    def _has_urgent_ready_task(self) -> bool:
        if self.workflow is None or not self.ready:
            return False
        target = self._makespan_target_for_cost()
        for tid in self.ready:
            task = self.workflow.tasks[tid]
            feasible_specs = [m.spec for m in self.machines if m.can_fit(task)]
            if not feasible_specs:
                continue
            best = min(estimate_duration(task, s) for s in feasible_specs)
            deadline = float(task.deadline or self.workflow.makespan_target or (self.current_time + best))
            slack_ratio = (deadline - self.current_time - best) / target
            if slack_ratio < self.config.urgent_slack_ratio:
                return True
        return False

    def _max_ready_criticality(self) -> float:
        if not self.ready:
            return 0.0
        return max(self.task_criticality.get(tid, 0.0) for tid in self.ready)

    def _compute_task_criticality(self) -> Dict[int, float]:
        if self.workflow is None:
            return {}
        ranks = {t.id: float(t.upward_rank or 0.0) for t in self.workflow.tasks}
        if not any(v > 0.0 for v in ranks.values()):
            memo: Dict[int, float] = {}

            def duration_floor(tid: int) -> float:
                task = self.workflow.tasks[tid]
                feasible = [s for s in self.cluster_specs if task.gpu <= s.gpu + 1e-9]
                specs = feasible or self.cluster_specs
                return min(estimate_duration(task, s) for s in specs)

            def rank_to_exit(tid: int) -> float:
                if tid in memo:
                    return memo[tid]
                task = self.workflow.tasks[tid]
                if not task.succs:
                    memo[tid] = duration_floor(tid)
                else:
                    memo[tid] = duration_floor(tid) + max(rank_to_exit(sid) for sid in task.succs)
                return memo[tid]

            ranks = {t.id: rank_to_exit(t.id) for t in self.workflow.tasks}
        max_rank = max(ranks.values(), default=1.0)
        if max_rank <= 0.0:
            return {t.id: 0.0 for t in self.workflow.tasks}
        return {tid: max(0.0, min(1.0, rank / max_rank)) for tid, rank in ranks.items()}

    def _cpeg_pair_info(self, tid: int, mid: int) -> Dict[str, float]:
        task = self.workflow.tasks[tid]
        criticality = self.task_criticality.get(tid, 0.0)
        feasible = [(i, m) for i, m in enumerate(self.machines) if m.can_fit(task)]
        if not feasible:
            return {
                "criticality": criticality,
                "eft": 0.0,
                "best_eft": 0.0,
                "eft_regret_ratio": 0.0,
                "slow_critical": 0.0,
                "feasible_machine_count": 0.0,
                "best_machine": -1.0,
            }
        best_mid = -1
        best_eft = math.inf
        chosen_eft = 0.0
        for candidate_mid, machine in feasible:
            start = self._earliest_start_on_machine(tid, candidate_mid)
            eft = start + estimate_duration(task, machine.spec)
            if candidate_mid == mid:
                chosen_eft = eft
            if eft < best_eft:
                best_eft = eft
                best_mid = candidate_mid
        regret_ratio = max(0.0, chosen_eft - best_eft) / max(best_eft, 1e-6)
        active = self._current_active_power()
        envelope = self.envelope_provider.value(self.current_time, self.phase_s)
        low_power_pressure = active < envelope - self.config.cpeg_power_pressure_margin * self.cluster_full_power
        slow_critical = (
            self.config.cpeg_enabled
            and criticality >= self.config.cpeg_criticality_threshold
            and regret_ratio >= self.config.cpeg_eft_regret_ratio
            and low_power_pressure
        )
        return {
            "criticality": criticality,
            "eft": chosen_eft,
            "best_eft": best_eft,
            "eft_regret_ratio": regret_ratio,
            "slow_critical": float(slow_critical),
            "feasible_machine_count": float(len(feasible)),
            "best_machine": float(best_mid),
        }

    def _anti_idle_pair_info(self, tid: int, mid: int, cpeg_info: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        info = cpeg_info or self._cpeg_pair_info(tid, mid)
        active = self._current_active_power()
        envelope = self.envelope_provider.value(self.current_time, self.phase_s)
        low_power_pressure = active < envelope - self.config.anti_idle_power_pressure_margin * self.cluster_full_power
        slow_finish = (
            self.config.anti_idle_enabled
            and low_power_pressure
            and info["eft_regret_ratio"] >= self.config.anti_idle_eft_regret_ratio
        )
        return {
            "eft_regret_ratio": info["eft_regret_ratio"],
            "slow_finish": float(slow_finish),
        }

    def _pair_energy_estimate(self, tid: int, mid: int) -> Dict[str, float]:
        task = self.workflow.tasks[tid]
        machine = self.machines[mid]
        duration = estimate_duration(task, machine.spec)
        start = self._earliest_start_on_machine(tid, mid)
        eft = start + duration
        was_active = machine.is_active_window(self.current_time, self.config.idle_timeout_s)
        before_power = machine.power_breakdown(self.current_time, self.config.idle_timeout_s).total
        after_cpu = min(1.0, (machine.used_cpu + task.cpu) / max(machine.spec.cpu, 1e-9))
        after_gpu = min(1.0, (machine.used_gpu + task.gpu) / max(machine.spec.gpu, 1e-9)) if machine.spec.gpu > 0 else 0.0
        from pcos.core.power import coupled_power

        after_power = coupled_power(
            machine.spec.p_idle,
            machine.spec.p_cpu0,
            machine.spec.p_gpu0,
            machine.spec.k1,
            machine.spec.k2,
            after_cpu,
            after_gpu,
            True,
        ).total
        comm = 0.0
        for p in task.preds:
            pm = self.task_machine.get(p)
            if pm is not None and pm != mid:
                comm += self.workflow.tasks[p].data_size * self.config.energy_per_MB
        activation_flag = 0.0 if was_active else 1.0
        if was_active:
            active_idle_delta = machine.spec.p_idle * max(0.0, eft - machine.last_active_time)
        else:
            active_idle_delta = machine.spec.p_idle * self.config.idle_timeout_s
        estimated_energy_delta = max(0.0, after_power - before_power) * duration + comm
        if self.config.energy_guard_include_idle_tail:
            estimated_energy_delta += active_idle_delta
        return {
            "duration": duration,
            "energy_delta": estimated_energy_delta,
            "eft": eft,
            "active_idle_delta": active_idle_delta,
            "machine_active_power": after_power,
            "machine_idle_power": machine.spec.p_idle,
            "activation_flag": activation_flag,
        }

    def _activation_guard_pair_info(
        self,
        tid: int,
        mid: int,
        energy_guard_info: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        task = self.workflow.tasks[tid]
        feasible = [(i, self._pair_energy_estimate(tid, i)) for i, m in enumerate(self.machines) if m.can_fit(task)]
        chosen = energy_guard_info or self._pair_energy_estimate(tid, mid)
        active_feasible = [(i, info) for i, info in feasible if info["activation_flag"] <= 0.5]
        if not active_feasible:
            return {
                "activation_flag": chosen["activation_flag"],
                "best_active_eft": 0.0,
                "eft_gain_ratio": 1.0,
                "penalized": 0.0,
            }
        best_active_eft = min(info["eft"] for _, info in active_feasible)
        eft_gain_ratio = max(0.0, best_active_eft - chosen["eft"]) / max(best_active_eft, 1e-6)
        criticality = self.task_criticality.get(tid, 0.0)
        ready_width = len(self.ready)
        penalized = (
            self.config.activation_guard_enabled
            and chosen["activation_flag"] > 0.5
            and eft_gain_ratio < self.config.activation_eft_gain_required
            and criticality < self.config.activation_guard_max_criticality
            and ready_width <= self.config.activation_guard_max_ready_width
        )
        return {
            "activation_flag": chosen["activation_flag"],
            "best_active_eft": best_active_eft,
            "eft_gain_ratio": eft_gain_ratio,
            "criticality": criticality,
            "ready_width": float(ready_width),
            "penalized": float(penalized),
        }

    def _energy_guard_pair_info(self, tid: int, mid: int) -> Dict[str, float]:
        task = self.workflow.tasks[tid]
        criticality = self.task_criticality.get(tid, 0.0)
        feasible = [(i, self._pair_energy_estimate(tid, i)) for i, m in enumerate(self.machines) if m.can_fit(task)]
        if not feasible:
            return {
                "duration": 0.0,
                "energy_delta": 0.0,
                "eft": 0.0,
                "best_eft": 0.0,
                "eft_regret_ratio": 0.0,
                "energy_regret_ratio": 0.0,
                "active_idle_delta": 0.0,
                "machine_active_power": 0.0,
                "machine_idle_power": 0.0,
                "activation_flag": 0.0,
                "preferred": 0.0,
                "penalized": 0.0,
            }
        chosen = next((info for candidate_mid, info in feasible if candidate_mid == mid), feasible[0][1])
        best_eft = min(info["eft"] for _, info in feasible)
        window_ratio = (
            self.config.critical_eft_window_ratio
            if criticality >= self.config.cpeg_criticality_threshold
            else self.config.noncritical_eft_window_ratio
        )
        window_eft = best_eft * (1.0 + max(0.0, window_ratio))
        candidates_in_window = [info for _, info in feasible if info["eft"] <= window_eft + 1e-9]
        if not candidates_in_window:
            candidates_in_window = [min((info for _, info in feasible), key=lambda x: x["eft"])]
        best_energy = min(info["energy_delta"] for info in candidates_in_window)
        eft_regret_ratio = max(0.0, chosen["eft"] - best_eft) / max(best_eft, 1e-6)
        energy_regret_ratio = max(0.0, chosen["energy_delta"] - best_energy) / max(best_energy, 1e-6)
        preferred = (
            self.config.energy_guard_enabled
            and chosen["eft"] <= window_eft + 1e-9
            and chosen["energy_delta"] <= best_energy * (1.0 + 1e-6) + 1e-9
        )
        penalized = self.config.energy_guard_enabled and not preferred
        return {
            "duration": chosen["duration"],
            "energy_delta": chosen["energy_delta"],
            "eft": chosen["eft"],
            "best_eft": best_eft,
            "eft_regret_ratio": eft_regret_ratio,
            "energy_regret_ratio": energy_regret_ratio,
            "active_idle_delta": chosen["active_idle_delta"],
            "machine_active_power": chosen["machine_active_power"],
            "machine_idle_power": chosen["machine_idle_power"],
            "activation_flag": chosen["activation_flag"],
            "preferred": float(preferred),
            "penalized": float(penalized),
        }

    def _current_active_power(self) -> float:
        return sum(m.power_breakdown(self.current_time, self.config.idle_timeout_s).total for m in self.machines)

    def _deadline_cost_dense(self) -> float:
        if self.workflow is None:
            return 0.0
        latest = self._latest_safe_start_boundary()
        if latest is None:
            return 0.0
        return max(0.0, (self.current_time - latest) / self._makespan_target_for_cost())

    def _makespan_target_for_cost(self) -> float:
        if self.workflow is None:
            return 1.0
        raw = self.workflow.metadata.get("makespan_target") if self.workflow.metadata else None
        if raw is not None:
            try:
                return max(1.0, float(raw))
            except (TypeError, ValueError):
                pass
        critical_path_estimate = max((t.upward_rank for t in self.workflow.tasks), default=1.0)
        return max(1.0, float(critical_path_estimate) * 1.5)

    def _episode_deadline_miss(self) -> float:
        if self.workflow is None or not self.done:
            return 0.0
        misses = 0
        for t in self.workflow.tasks:
            finish = self.task_finish.get(t.id, math.inf)
            if finish > float(t.deadline or self.workflow.makespan_target or math.inf) + 1e-9:
                misses += 1
        return misses / max(1, len(self.workflow.tasks))

    def _observe(self) -> Dict[str, np.ndarray]:
        pairs = self.pair_actions()
        pair_features = []
        cpeg_slow = []
        cpeg_regret = []
        cpeg_criticality = []
        anti_idle_slow = []
        anti_idle_regret = []
        energy_guard_preferred = []
        energy_guard_penalized = []
        energy_guard_regret = []
        activation_guard_penalized = []
        activation_guard_gain_ratio = []
        for tid, mid in pairs:
            info = self._cpeg_pair_info(tid, mid)
            anti_idle_info = self._anti_idle_pair_info(tid, mid, info)
            energy_guard_info = self._energy_guard_pair_info(tid, mid)
            activation_guard_info = self._activation_guard_pair_info(tid, mid, energy_guard_info)
            pair_features.append(self._pair_features(tid, mid, energy_guard_info))
            cpeg_slow.append(info["slow_critical"])
            cpeg_regret.append(info["eft_regret_ratio"])
            cpeg_criticality.append(info["criticality"])
            anti_idle_slow.append(anti_idle_info["slow_finish"])
            anti_idle_regret.append(anti_idle_info["eft_regret_ratio"])
            energy_guard_preferred.append(energy_guard_info["preferred"])
            energy_guard_penalized.append(energy_guard_info["penalized"])
            energy_guard_regret.append(energy_guard_info["energy_regret_ratio"])
            activation_guard_penalized.append(activation_guard_info["penalized"])
            activation_guard_gain_ratio.append(activation_guard_info["eft_gain_ratio"])
        if pair_features:
            pair_arr = np.asarray(pair_features, dtype=np.float32)
        else:
            pair_arr = np.zeros((0, 20 if self.config.energy_guard_enabled else 10), dtype=np.float32)
        cpeg_slow_arr = np.asarray(cpeg_slow, dtype=np.float32)
        anti_idle_slow_arr = np.asarray(anti_idle_slow, dtype=np.float32)
        energy_guard_preferred_arr = np.asarray(energy_guard_preferred, dtype=np.float32)
        energy_guard_penalized_arr = np.asarray(energy_guard_penalized, dtype=np.float32)
        activation_guard_penalized_arr = np.asarray(activation_guard_penalized, dtype=np.float32)
        self.cpeg_penalized_pair_count += int(cpeg_slow_arr.sum())
        self.anti_idle_penalized_pair_count += int(anti_idle_slow_arr.sum())
        self.energy_guard_penalized_pair_count += int(energy_guard_penalized_arr.sum())
        self.activation_guard_penalized_pair_count += int(activation_guard_penalized_arr.sum())
        global_arr = np.asarray(self._global_features(), dtype=np.float32)
        mask = np.ones(pair_arr.shape[0] + (1 if self._defer_allowed() else 0), dtype=np.float32)
        return {
            "global": global_arr,
            "pairs": pair_arr,
            "mask": mask,
            "defer_allowed": np.asarray([1.0 if self._defer_allowed() else 0.0], dtype=np.float32),
            "pair_cpeg_slow_critical": cpeg_slow_arr,
            "pair_cpeg_eft_regret_ratio": np.asarray(cpeg_regret, dtype=np.float32),
            "pair_cpeg_criticality": np.asarray(cpeg_criticality, dtype=np.float32),
            "pair_anti_idle_slow_finish": anti_idle_slow_arr,
            "pair_anti_idle_eft_regret_ratio": np.asarray(anti_idle_regret, dtype=np.float32),
            "pair_is_energy_guard_preferred": energy_guard_preferred_arr,
            "pair_energy_guard_penalized": energy_guard_penalized_arr,
            "pair_energy_guard_energy_regret_ratio": np.asarray(energy_guard_regret, dtype=np.float32),
            "pair_activation_guard_penalized": activation_guard_penalized_arr,
            "pair_activation_guard_eft_gain_ratio": np.asarray(activation_guard_gain_ratio, dtype=np.float32),
        }

    def _pair_features(self, tid: int, mid: int, energy_guard_info: Optional[Dict[str, float]] = None) -> List[float]:
        task = self.workflow.tasks[tid]
        m = self.machines[mid]
        dur = estimate_duration(task, m.spec)
        active_power = self._current_active_power()
        before = m.power_breakdown(self.current_time, self.config.idle_timeout_s).total
        # Approximate after power by temporarily adding utilization fractions.
        after_cpu = min(1.0, (m.used_cpu + task.cpu) / max(m.spec.cpu, 1e-9))
        after_gpu = min(1.0, (m.used_gpu + task.gpu) / max(m.spec.gpu, 1e-9)) if m.spec.gpu > 0 else 0.0
        from pcos.core.power import coupled_power
        after = coupled_power(m.spec.p_idle, m.spec.p_cpu0, m.spec.p_gpu0, m.spec.k1, m.spec.k2, after_cpu, after_gpu, True).total
        delta_power = max(0.0, after - before)
        envelope = self.envelope_provider.value(self.current_time, self.phase_s)
        projected_violation = max(0.0, (active_power + delta_power - envelope) / max(self.cluster_full_power, 1e-9))
        comm = 0.0
        for p in task.preds:
            pm = self.task_machine.get(p)
            if pm is not None and pm != mid:
                comm += self.workflow.tasks[p].data_size * self.config.energy_per_MB
        deadline = float(task.deadline or self.workflow.makespan_target or (self.current_time + dur))
        slack_after = (deadline - self.current_time - dur) / max(self.workflow.makespan_target or 1.0, 1.0)
        activation = 0.0 if m.activated else 1.0
        fit_cpu = (m.spec.cpu - m.used_cpu) / max(task.cpu, 1e-9)
        fit_gpu = (m.spec.gpu - m.used_gpu) / max(task.gpu, 1e-9) if task.gpu > 0 else 10.0
        features = [
            task.cpu / 64.0,
            task.gpu / 8.0,
            task.mem / 768.0,
            dur / 1000.0,
            task.gpu_intensity,
            delta_power / max(self.cluster_full_power, 1e-9),
            projected_violation,
            comm / 100.0,
            slack_after,
            activation,
        ]
        if self.config.energy_guard_enabled:
            eg = energy_guard_info or self._energy_guard_pair_info(tid, mid)
            target = self._makespan_target_for_cost()
            features.extend(
                [
                    eg["duration"] / 1000.0,
                    eg["energy_delta"] / max(self.config.energy_norm, 1e-9),
                    eg["eft"] / max(target, 1.0),
                    eg["best_eft"] / max(target, 1.0),
                    eg["eft_regret_ratio"],
                    eg["energy_regret_ratio"],
                    eg["active_idle_delta"] / max(self.config.energy_norm, 1e-9),
                    eg["machine_active_power"] / max(self.cluster_full_power, 1e-9),
                    eg["machine_idle_power"] / max(self.cluster_full_power, 1e-9),
                    eg["activation_flag"],
                ]
            )
        return features

    def _global_features(self) -> List[float]:
        n = max(1, len(self.workflow.tasks) if self.workflow else 1)
        envelope = self.envelope_provider.value(self.current_time, self.phase_s)
        active_power = self._current_active_power()
        next_t = self.envelope_provider.next_change_after(self.current_time, self.phase_s)
        next_env = self.envelope_provider.value(next_t, self.phase_s)
        return [
            len(self.ready) / n,
            len(self.running) / n,
            len(self.completed) / n,
            self.current_time / max(self.workflow.makespan_target or 1.0, 1.0),
            self.energy_active_total / max(self.config.energy_norm, 1.0),
            envelope / max(self.cluster_full_power, 1e-9),
            active_power / max(self.cluster_full_power, 1e-9),
            (envelope - active_power) / max(self.cluster_full_power, 1e-9),
            (next_t - self.current_time) / max(self.config.max_defer_s, 1.0),
            (next_env - envelope) / max(self.cluster_full_power, 1e-9),
            1.0 if self._defer_allowed() else 0.0,
        ]

    def _episode_info(self) -> Dict[str, float]:
        makespan = max(self.task_finish.values(), default=self.current_time)
        activated = sum(1 for m in self.machines if m.activated)
        total_wait = 0.0
        for tid, st in self.task_start.items():
            total_wait += max(0.0, st)  # workflows are episode-relative; ready-time approximation.
        avg_wait = total_wait / max(1, len(self.task_start))
        cost = max(0.0, (makespan - float(self.workflow.makespan_target or makespan)) / max(float(self.workflow.makespan_target or makespan), 1e-9))
        return {
            "time": self.current_time,
            "makespan": makespan,
            "energy_active_total": self.energy_active_total,
            "energy_active_compute": self.energy_active_compute,
            "energy_idle_active": self.energy_idle_active,
            "energy_cpu": self.energy_cpu,
            "energy_gpu": self.energy_gpu,
            "energy_interaction_signed": self.energy_interaction_signed,
            "energy_comm": self.energy_comm,
            "energy_always_on_total": self.energy_always_on_total,
            "power_envelope_violation": self.power_violation_integral,
            "peak_active_power": self.peak_active_power,
            "peak_active_power_ratio": self.peak_active_power / max(self.cluster_full_power, 1e-9),
            "deadline_miss_rate": self._episode_deadline_miss(),
            "cost": cost,
            "avg_waiting_time": avg_wait,
            "server_activated_count": activated,
            "cross_machine_edge_count": self.cross_machine_edge_count,
            "cross_machine_data_mb": self.cross_machine_data_mb,
            "defer_count": float(self.defer_count),
            "defer_time_total": self.defer_time_total,
            "cpeg_penalized_pair_count": float(self.cpeg_penalized_pair_count),
            "cpeg_slow_action_count": float(self.cpeg_slow_action_count),
            "cpeg_cost_total": self.cpeg_cost_total,
            "anti_idle_penalized_pair_count": float(self.anti_idle_penalized_pair_count),
            "anti_idle_slow_action_count": float(self.anti_idle_slow_action_count),
            "anti_idle_cost_total": self.anti_idle_cost_total,
            "energy_guard_penalized_pair_count": float(self.energy_guard_penalized_pair_count),
            "energy_guard_selected_preferred_rate": (
                float(self.energy_guard_selected_preferred_count) / max(1.0, float(self.energy_guard_selected_count))
            ),
            "energy_guard_selected_regret_mean": (
                self.energy_guard_selected_regret_total / max(1.0, float(self.energy_guard_selected_count))
            ),
            "energy_guard_cost_total": self.energy_guard_cost_total,
            "activation_guard_penalized_pair_count": float(self.activation_guard_penalized_pair_count),
            "activation_guard_selected_penalized_rate": (
                float(self.activation_guard_selected_penalized_count) / max(1.0, float(self.activation_guard_selected_count))
            ),
            "activation_guard_selected_gain_mean": (
                self.activation_guard_selected_gain_total / max(1.0, float(self.activation_guard_selected_count))
            ),
            "activation_guard_cost_total": self.activation_guard_cost_total,
        }
