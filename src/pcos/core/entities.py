from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Set, Tuple

from .power import PowerBreakdown, coupled_power


@dataclass
class Task:
    id: int
    cpu: float
    gpu: float
    mem: float
    base_duration: float
    gpu_intensity: float = 0.0
    data_size: float = 0.0
    profile_name: str = "generic"
    profile_id: int = 0
    preferred_type: str = ""
    affinity_bonus: float = 0.0
    parallelism: float = 1.0
    deadline: Optional[float] = None
    upward_rank: float = 0.0
    downward_rank: float = 0.0
    preds: List[int] = field(default_factory=list)
    succs: List[int] = field(default_factory=list)


@dataclass
class Workflow:
    workflow_id: str
    tasks: List[Task]
    edges: List[Tuple[int, int]]
    metadata: Dict[str, object] = field(default_factory=dict)
    makespan_target: Optional[float] = None

    def task_by_id(self, tid: int) -> Task:
        return self.tasks[tid]


@dataclass(frozen=True)
class MachineSpec:
    name: str
    cpu: float
    gpu: float
    mem: float
    cpu_perf: float
    gpu_perf: float
    mem_perf: float
    p_idle: float
    p_cpu0: float
    p_gpu0: float
    k1: float
    k2: float
    speed_profiles: Dict[str, float] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    machine_id: str = ""

    def full_power(self) -> float:
        return coupled_power(self.p_idle, self.p_cpu0, self.p_gpu0, self.k1, self.k2, 1.0, 1.0, True).total


@dataclass
class RunningTask:
    task_id: int
    machine_id: int
    start_time: float
    end_time: float
    cpu: float
    gpu: float
    mem: float


@dataclass
class MachineState:
    spec: MachineSpec
    index: int
    used_cpu: float = 0.0
    used_gpu: float = 0.0
    used_mem: float = 0.0
    running: Dict[int, RunningTask] = field(default_factory=dict)
    activated: bool = False
    last_active_time: float = 0.0

    def can_fit(self, task: Task) -> bool:
        return (
            self.used_cpu + task.cpu <= self.spec.cpu + 1e-9
            and self.used_gpu + task.gpu <= self.spec.gpu + 1e-9
            and self.used_mem + task.mem <= self.spec.mem + 1e-9
        )

    def assign(self, rt: RunningTask) -> None:
        self.running[rt.task_id] = rt
        self.used_cpu += rt.cpu
        self.used_gpu += rt.gpu
        self.used_mem += rt.mem
        self.activated = True
        self.last_active_time = max(self.last_active_time, rt.end_time)

    def release(self, task_id: int) -> None:
        rt = self.running.pop(task_id)
        self.used_cpu -= rt.cpu
        self.used_gpu -= rt.gpu
        self.used_mem -= rt.mem

    def utilization(self) -> Tuple[float, float]:
        u_cpu = self.used_cpu / max(self.spec.cpu, 1e-9)
        u_gpu = self.used_gpu / max(self.spec.gpu, 1e-9) if self.spec.gpu > 0 else 0.0
        return u_cpu, u_gpu

    def is_active_window(self, now: float, idle_timeout: float) -> bool:
        if not self.activated:
            return False
        if self.running:
            return True
        return now <= self.last_active_time + idle_timeout + 1e-9

    def power_breakdown(self, now: float, idle_timeout: float) -> PowerBreakdown:
        active = self.is_active_window(now, idle_timeout)
        u_cpu, u_gpu = self.utilization()
        return coupled_power(
            self.spec.p_idle,
            self.spec.p_cpu0,
            self.spec.p_gpu0,
            self.spec.k1,
            self.spec.k2,
            u_cpu,
            u_gpu,
            active=active,
        )


def estimate_duration(task: Task, spec: MachineSpec) -> float:
    """Task-machine duration estimate used by environment and heuristics.

    CPU-only tasks are forced to have zero effective GPU intensity. This keeps
    the duration model consistent with the active-power model: if task.gpu == 0,
    the task does not consume GPU resources, does not receive GPU acceleration,
    and does not create GPU or CPU-GPU interaction energy.
    """
    gpu_intensity = 0.0 if task.gpu <= 0 else max(0.0, min(1.0, task.gpu_intensity))
    cpu_mix = 1.0 - gpu_intensity
    gpu_mix = gpu_intensity
    cpu_units = max(task.cpu / 16.0, 0.25)
    gpu_units = max(task.gpu, 0.0)
    cpu_comp = spec.cpu_perf * max(cpu_units**0.30, 0.35)
    if task.gpu > 0 and spec.gpu > 0:
        gpu_comp = spec.gpu_perf * max(gpu_units**0.25, 1.0) * min(1.0, 0.70 + 0.10 * spec.gpu)
    elif task.gpu > 0:
        gpu_comp = 0.05
    else:
        gpu_comp = 1.0
    profile_factor = spec.speed_profiles.get(task.profile_name, spec.speed_profiles.get("generic", 1.0))
    affinity = 1.0
    if task.preferred_type and task.preferred_type == spec.name:
        affinity += max(0.0, task.affinity_bonus)
    elif task.preferred_type and task.preferred_type in spec.tags:
        affinity += max(0.0, task.affinity_bonus) * 0.5
    mem_pressure = task.mem / max(spec.mem, 1e-9)
    mem_penalty = 1.0 + max(0.0, mem_pressure - 0.65) * 0.75
    effective = (cpu_mix * cpu_comp + gpu_mix * gpu_comp) * profile_factor * affinity
    return max(0.01, float(task.base_duration) / max(effective, 0.10) * mem_penalty)


def default_cluster_specs() -> List[MachineSpec]:
    """Five-node heterogeneous CPU-GPU cluster used by the clean PCEA project."""
    return [
        MachineSpec(
            name="REAL_T4_4GPU",
            cpu=64,
            gpu=4,
            mem=512,
            cpu_perf=1.00,
            gpu_perf=1.00,
            mem_perf=1.00,
            p_idle=384,
            p_cpu0=145,
            p_gpu0=240,
            k1=0.00053641,
            k2=-0.00076630,
            speed_profiles={"cpu_preproc": 1.00, "gpu_train": 0.85, "gpu_infer": 1.00, "hybrid_analytics": 1.00, "generic": 1.0},
            tags={"t4", "infer", "balanced_gpu", "REAL_T4_4GPU"},
        ),
        MachineSpec(
            name="REAL_4090_2GPU",
            cpu=48,
            gpu=2,
            mem=512,
            cpu_perf=1.18,
            gpu_perf=3.00,
            mem_perf=1.05,
            p_idle=400,
            p_cpu0=97,
            p_gpu0=730,
            k1=0.00114206,
            k2=-0.00148859,
            speed_profiles={"cpu_preproc": 1.05, "gpu_train": 1.50, "gpu_infer": 1.45, "hybrid_analytics": 1.25, "generic": 1.0},
            tags={"4090", "train", "high_gpu", "REAL_4090_2GPU"},
        ),
        MachineSpec(
            name="SYN_T4_2GPU",
            cpu=64,
            gpu=2,
            mem=384,
            cpu_perf=0.96,
            gpu_perf=0.92,
            mem_perf=0.95,
            p_idle=360,
            p_cpu0=145,
            p_gpu0=120,
            k1=0.00050,
            k2=-0.00060,
            speed_profiles={"cpu_preproc": 0.95, "gpu_train": 0.75, "gpu_infer": 0.90, "hybrid_analytics": 0.90, "generic": 0.95},
            tags={"t4", "small_gpu", "SYN_T4_2GPU"},
        ),
        MachineSpec(
            name="SYN_T4_8GPU",
            cpu=64,
            gpu=8,
            mem=768,
            cpu_perf=1.02,
            gpu_perf=1.06,
            mem_perf=1.08,
            p_idle=420,
            p_cpu0=145,
            p_gpu0=480,
            k1=0.00042,
            k2=-0.00050,
            speed_profiles={"cpu_preproc": 1.00, "gpu_train": 1.15, "gpu_infer": 1.05, "hybrid_analytics": 1.10, "generic": 1.0},
            tags={"t4", "multi_gpu", "high_mem", "SYN_T4_8GPU"},
        ),
        MachineSpec(
            name="SYN_4090_4GPU",
            cpu=48,
            gpu=4,
            mem=768,
            cpu_perf=1.20,
            gpu_perf=3.08,
            mem_perf=1.10,
            p_idle=520,
            p_cpu0=97,
            p_gpu0=1460,
            k1=0.00050,
            k2=-0.00050,
            speed_profiles={"cpu_preproc": 1.06, "gpu_train": 1.65, "gpu_infer": 1.55, "hybrid_analytics": 1.30, "generic": 1.05},
            tags={"4090", "train", "multi_gpu", "high_gpu", "SYN_4090_4GPU"},
        ),
    ]


def cluster_specs_from_preset(preset: str) -> List[MachineSpec]:
    """Build a cluster preset without changing per-machine type parameters."""
    if preset in {"default", "default_5", "cluster_5_default"}:
        return default_cluster_specs()
    templates = {spec.name: spec for spec in default_cluster_specs()}
    if preset != "cluster_15_realistic":
        raise ValueError(f"Unknown cluster preset: {preset!r}")
    layout = [
        ("SYN_T4_2GPU", 4, "syn_t4_2g"),
        ("REAL_T4_4GPU", 3, "real_t4_4g"),
        ("SYN_T4_8GPU", 3, "syn_t4_8g"),
        ("REAL_4090_2GPU", 2, "real_4090_2g"),
        ("SYN_4090_4GPU", 3, "syn_4090_4g"),
    ]
    cluster: List[MachineSpec] = []
    for type_name, count, prefix in layout:
        template = templates[type_name]
        for i in range(count):
            cluster.append(replace(template, machine_id=f"{prefix}_{i}"))
    return cluster


def cluster_specs_from_config(cluster_cfg: object | None) -> List[MachineSpec]:
    if cluster_cfg is None:
        return default_cluster_specs()
    if isinstance(cluster_cfg, str):
        return cluster_specs_from_preset(cluster_cfg)
    if isinstance(cluster_cfg, dict):
        return cluster_specs_from_preset(str(cluster_cfg.get("preset", "default_5")))
    raise ValueError(f"Invalid cluster config: {cluster_cfg!r}")
