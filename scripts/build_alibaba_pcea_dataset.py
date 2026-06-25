#!/usr/bin/env python3
"""
Build a PCEA-PPO native Alibaba-enhanced DAG dataset.

This script converts Alibaba 2018 batch_task.csv + batch_instance.csv into the
clean `pcos` workflow format used by the Power-Compute Scheduler project.

Raw input placement expected by default:
    datasets/alibaba_raw/batch_task.csv
    datasets/alibaba_raw/batch_instance.csv

Default processed output:
    datasets/alibaba_pcea/processed/

Key design choices:
  * Alibaba provides compute-side workload realism.
  * GPU/resource/profile/output fields are rule-augmented because the original
    trace has no GPU columns.
  * DAG edges are recovered from strict Alibaba M-style task names when present,
    otherwise inferred sparsely from task runtime intervals.
  * Communication `data_size` is exported in MB, matching the PCEA environment.
  * train/val/test/benchmark are independent time-ordered splits.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

TASK_COLS = [
    "create_time",
    "modify_time",
    "job_name",
    "task_name",
    "instance_num",
    "status",
    "plan_cpu",
    "plan_mem",
]
INSTANCE_COLS_MIN = [
    "start_time",
    "end_time",
    "job_name",
    "task_name",
    "machine",
    "status",
]
TERMINAL_TASK_STATUSES = {"Terminated"}
TERMINAL_INSTANCE_STATUSES = {"Terminated"}
EXPLICIT_DAG_RE = re.compile(r"^M\d+(?:_\d+)*$")

PROFILE_NAME_MAP = {
    "cpu_only": "cpu_preproc",
    "gpu_infer": "gpu_infer",
    "hybrid_analytics": "hybrid_analytics",
    "gpu_train": "gpu_train",
}
PROFILE_ID_MAP = {
    "cpu_preproc": 0,
    "gpu_train": 1,
    "gpu_infer": 2,
    "hybrid_analytics": 3,
    "memory_shuffle": 4,
    "generic": 5,
}
GPU_INTENSITY_MAP = {
    # CPU-only tasks must not receive GPU speedup or GPU dynamic energy.
    "cpu_only": 0.00,
    "gpu_infer": 0.70,
    "hybrid_analytics": 0.55,
    "gpu_train": 0.90,
}
AFFINITY_BONUS_MAP = {
    "cpu_only": 0.00,
    "gpu_infer": 0.08,
    "hybrid_analytics": 0.12,
    "gpu_train": 0.18,
}

# Lightweight power proxy used only for dataset diagnostics. The actual training
# environment still uses src/pcos/core/power.py and the full machine catalog.
POWER_PROXY_SPECS = {
    "REAL_T4_4GPU": {"cpu": 64.0, "gpu": 4.0, "p_idle": 384.0, "p_cpu0": 145.0, "p_gpu0": 240.0, "k1": 0.00053641, "k2": -0.00076630},
    "REAL_4090_2GPU": {"cpu": 48.0, "gpu": 2.0, "p_idle": 400.0, "p_cpu0": 97.0, "p_gpu0": 730.0, "k1": 0.00114206, "k2": -0.00148859},
    "SYN_T4_2GPU": {"cpu": 64.0, "gpu": 2.0, "p_idle": 360.0, "p_cpu0": 145.0, "p_gpu0": 120.0, "k1": 0.00050, "k2": -0.00060},
    "SYN_T4_8GPU": {"cpu": 64.0, "gpu": 8.0, "p_idle": 420.0, "p_cpu0": 145.0, "p_gpu0": 480.0, "k1": 0.00042, "k2": -0.00050},
    "SYN_4090_4GPU": {"cpu": 48.0, "gpu": 4.0, "p_idle": 520.0, "p_cpu0": 97.0, "p_gpu0": 1460.0, "k1": 0.00050, "k2": -0.00050},
}


@dataclass
class Reservoir:
    size: int = 64
    values: List[float] = field(default_factory=list)
    seen: int = 0

    def add(self, x: float, rng: random.Random) -> None:
        self.seen += 1
        if len(self.values) < self.size:
            self.values.append(float(x))
            return
        j = rng.randint(1, self.seen)
        if j <= self.size:
            self.values[j - 1] = float(x)

    def median(self) -> Optional[float]:
        if not self.values:
            return None
        return float(np.median(np.asarray(self.values, dtype=float)))


@dataclass
class RuntimeAgg:
    earliest_start: float = math.inf
    latest_end: float = -math.inf
    count: int = 0
    machine_set: Set[str] = field(default_factory=set)
    duration_sample: Reservoir = field(default_factory=lambda: Reservoir(size=64))

    def add(self, start: float, end: float, machine: str, rng: random.Random) -> None:
        if np.isfinite(start):
            self.earliest_start = min(self.earliest_start, float(start))
        if np.isfinite(end):
            self.latest_end = max(self.latest_end, float(end))
        if np.isfinite(start) and np.isfinite(end) and end > start:
            self.duration_sample.add(end - start, rng)
        if machine:
            self.machine_set.add(machine)
        self.count += 1

    @property
    def duration_p50(self) -> Optional[float]:
        return self.duration_sample.median()


def safe_float(x: object, default: float = np.nan) -> float:
    if x is None:
        return default
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return default
    try:
        return float(s)
    except Exception:
        return default


def safe_int(x: object, default: int = 0) -> int:
    v = safe_float(x)
    if np.isnan(v):
        return default
    return int(v)


def normalize_str(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s


def sanitize_token(s: str) -> str:
    s = normalize_str(s)
    if not s:
        return "unknown"
    s = re.sub(r"[^0-9A-Za-z_\-.]+", "_", s)
    return s.strip("_") or "unknown"


def sanitize_node_id(s: str) -> str:
    s = sanitize_token(s).replace("-", "_").replace(".", "_")
    if not s:
        return "node_unknown"
    if s[0].isdigit():
        s = f"n{s}"
    return s


def detect_num_columns(csv_path: str | Path) -> int:
    with Path(csv_path).open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                return len(row)
    raise ValueError(f"Empty CSV: {csv_path}")


def parse_task_name(task_name: str) -> Tuple[str, List[str], bool]:
    """Parse strict Alibaba M-style DAG encoding only.

    Valid explicit examples:
      M1      -> node n1, no parents
      M2_1    -> node n2 depends on n1
      M5_3_4  -> node n5 depends on n3 and n4

    Non-matching task names are treated as raw node ids and handled by the
    fallback time-inferred DAG rule.
    """
    raw = normalize_str(task_name)
    if not raw:
        return "node_unknown", [], False
    if EXPLICIT_DAG_RE.match(raw):
        nums = re.findall(r"\d+", raw)
        cur = f"n{int(nums[0])}"
        parents = [f"n{int(x)}" for x in nums[1:]]
        return cur, parents, len(parents) > 0
    return sanitize_node_id(raw), [], False


def reachable(src: str, dst: str, succ: Dict[str, Set[str]]) -> bool:
    q = deque([src])
    visited = {src}
    while q:
        u = q.popleft()
        for v in succ.get(u, ()):  # pragma: no branch
            if v == dst:
                return True
            if v not in visited:
                visited.add(v)
                q.append(v)
    return False


def is_dag(node_ids: Sequence[str], edges: Iterable[Tuple[str, str]]) -> bool:
    indeg = {n: 0 for n in node_ids}
    succ = {n: [] for n in node_ids}
    for u, v in edges:
        if u not in indeg or v not in indeg or u == v:
            continue
        succ[u].append(v)
        indeg[v] += 1
    q = deque([n for n in node_ids if indeg[n] == 0])
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return seen == len(node_ids)


def topological_order(node_ids: Sequence[str], edges: Iterable[Tuple[str, str]]) -> List[str]:
    indeg = {n: 0 for n in node_ids}
    succ = {n: [] for n in node_ids}
    for u, v in edges:
        if u in indeg and v in indeg and u != v:
            succ[u].append(v)
            indeg[v] += 1
    q = deque(sorted([n for n in node_ids if indeg[n] == 0]))
    out: List[str] = []
    while q:
        u = q.popleft()
        out.append(u)
        for v in sorted(succ[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(out) != len(node_ids):
        seen = set(out)
        out.extend([n for n in node_ids if n not in seen])
    return out


def transitive_reduction(node_ids: Sequence[str], edges: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    succ = {n: set() for n in node_ids}
    for u, v in edges:
        if u in succ and v in succ and u != v:
            succ[u].add(v)
    reduced = set(edges)
    for u, v in list(edges):
        if u not in succ:
            continue
        succ[u].discard(v)
        if reachable(u, v, succ):
            reduced.discard((u, v))
        succ[u].add(v)
    return reduced


def infer_edges_from_time(task_rows: List[dict], parent_window: float = 60.0) -> Set[Tuple[str, str]]:
    rows = [r for r in task_rows if np.isfinite(r["earliest_start"]) and np.isfinite(r["latest_end"])]
    rows.sort(key=lambda x: (x["earliest_start"], x["latest_end"], x["node_id"]))
    edges: Set[Tuple[str, str]] = set()
    for i, cur in enumerate(rows):
        preds = [
            r for r in rows[:i]
            if r["latest_end"] <= cur["earliest_start"] and r["node_id"] != cur["node_id"]
        ]
        if not preds:
            continue
        max_end = max(r["latest_end"] for r in preds)
        keep = [r for r in preds if r["latest_end"] >= max_end - parent_window]
        for p in keep:
            edges.add((p["node_id"], cur["node_id"]))
    return edges


def approx_quantile(vals: Sequence[float], q: float, default: float) -> float:
    arr = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return default
    return float(np.quantile(arr, q))


def bucket_by_quantiles(value: float, quantiles: Sequence[float], buckets: Sequence[int], default_idx: int = 1) -> int:
    if not np.isfinite(value):
        return int(buckets[min(default_idx, len(buckets) - 1)])
    idx = 0
    while idx < len(quantiles) and value > quantiles[idx]:
        idx += 1
    idx = min(idx, len(buckets) - 1)
    return int(buckets[idx])


def choose_from_levels(levels: Sequence[int], duration: float, q40: float, q80: float) -> int:
    xs = list(levels)
    if len(xs) == 1:
        return int(xs[0])
    if duration <= q40:
        return int(xs[0])
    if duration >= q80:
        return int(xs[-1])
    return int(xs[len(xs) // 2])


def load_batch_task(batch_task_path: str | Path, max_jobs: Optional[int]) -> pd.DataFrame:
    ncols = detect_num_columns(batch_task_path)
    if ncols < 8:
        raise ValueError(f"batch_task.csv should have at least 8 columns, got {ncols}")
    df = pd.read_csv(
        batch_task_path,
        header=None,
        names=TASK_COLS,
        usecols=list(range(8)),
        dtype=str,
        na_filter=False,
        low_memory=False,
    )
    for col in ["create_time", "modify_time", "instance_num", "plan_cpu", "plan_mem"]:
        df[col] = df[col].map(safe_float)
    df["job_name"] = df["job_name"].map(normalize_str)
    df["task_name"] = df["task_name"].map(normalize_str)
    df["status"] = df["status"].map(normalize_str)
    df = df[(df["job_name"] != "") & (df["task_name"] != "")]
    df = df[df["status"].isin(TERMINAL_TASK_STATUSES)]
    df = df[df["instance_num"].fillna(0) > 0].copy()
    if max_jobs is not None and max_jobs > 0:
        job_order = df.groupby("job_name")["create_time"].min().sort_values(kind="mergesort").index[:max_jobs]
        df = df[df["job_name"].isin(set(job_order))].copy()
    return df


def scan_batch_instance(
    batch_instance_path: str | Path,
    selected_pairs: Set[Tuple[str, str]],
    selected_jobs: Set[str],
    chunk_size: int,
    rng: random.Random,
) -> Dict[Tuple[str, str], RuntimeAgg]:
    ncols = detect_num_columns(batch_instance_path)
    if ncols < 6:
        raise ValueError(f"batch_instance.csv should have at least 6 columns, got {ncols}")
    stats: Dict[Tuple[str, str], RuntimeAgg] = defaultdict(RuntimeAgg)
    reader = pd.read_csv(
        batch_instance_path,
        header=None,
        names=INSTANCE_COLS_MIN,
        usecols=list(range(6)),
        dtype=str,
        na_filter=False,
        low_memory=False,
        chunksize=chunk_size,
    )
    for chunk in reader:
        chunk["job_name"] = chunk["job_name"].map(normalize_str)
        chunk["task_name"] = chunk["task_name"].map(normalize_str)
        chunk["status"] = chunk["status"].map(normalize_str)
        chunk["start_time"] = chunk["start_time"].map(safe_float)
        chunk["end_time"] = chunk["end_time"].map(safe_float)
        chunk["machine"] = chunk["machine"].map(normalize_str)
        chunk = chunk[
            (chunk["job_name"] != "")
            & (chunk["task_name"] != "")
            & (chunk["status"].isin(TERMINAL_INSTANCE_STATUSES))
            & (chunk["job_name"].isin(selected_jobs))
        ].copy()
        if chunk.empty:
            continue
        chunk["pair"] = list(zip(chunk["job_name"], chunk["task_name"]))
        chunk = chunk[chunk["pair"].isin(selected_pairs)]
        if chunk.empty:
            continue
        for row in chunk.itertuples(index=False):
            start = safe_float(row.start_time)
            end = safe_float(row.end_time)
            if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                continue
            stats[(row.job_name, row.task_name)].add(start, end, row.machine, rng)
    return stats


def build_workflows(
    task_df: pd.DataFrame,
    runtime_stats: Dict[Tuple[str, str], RuntimeAgg],
    min_nodes: int,
    max_nodes: int,
    parent_window: float,
    require_runtime: bool = True,
) -> List[dict]:
    task_df = task_df.copy()
    for col, default in [
        ("earliest_start", np.nan),
        ("latest_end", np.nan),
        ("base_duration", np.nan),
        ("observed_instances", 0),
        ("machine_count", 0),
    ]:
        task_df[col] = default
    for idx, row in task_df.iterrows():
        agg = runtime_stats.get((row["job_name"], row["task_name"]))
        if agg is None:
            continue
        task_df.at[idx, "earliest_start"] = agg.earliest_start if agg.earliest_start < math.inf else np.nan
        task_df.at[idx, "latest_end"] = agg.latest_end if agg.latest_end > -math.inf else np.nan
        task_df.at[idx, "base_duration"] = agg.duration_p50
        task_df.at[idx, "observed_instances"] = agg.count
        task_df.at[idx, "machine_count"] = len(agg.machine_set)
    if require_runtime:
        task_df = task_df[np.isfinite(task_df["base_duration"])].copy()

    workflows: List[dict] = []
    for job_name, g in task_df.groupby("job_name", sort=False):
        rows: List[dict] = []
        explicit_edges: Set[Tuple[str, str]] = set()
        any_explicit = False
        used_ids: Dict[str, int] = defaultdict(int)
        parsed: List[Tuple[object, str, List[str], bool]] = []
        for r in g.itertuples(index=False):
            node_id, parents, explicit = parse_task_name(r.task_name)
            used_ids[node_id] += 1
            if used_ids[node_id] > 1:
                node_id = f"{node_id}_{used_ids[node_id]}"
            parsed.append((r, node_id, parents, explicit))
            any_explicit = any_explicit or explicit
        node_set = {node_id for _, node_id, _, _ in parsed}
        for r, node_id, parents, _explicit in parsed:
            rows.append({
                "raw_task_name": r.task_name,
                "node_id": node_id,
                "parents_raw": parents,
                "create_time": safe_float(r.create_time),
                "modify_time": safe_float(r.modify_time),
                "instance_num": safe_int(r.instance_num, 0),
                "plan_cpu": safe_float(r.plan_cpu),
                "plan_mem": safe_float(r.plan_mem),
                "earliest_start": safe_float(r.earliest_start),
                "latest_end": safe_float(r.latest_end),
                "base_duration": safe_float(r.base_duration),
                "observed_instances": safe_int(r.observed_instances, 0),
                "machine_count": safe_int(r.machine_count, 0),
            })
        if any_explicit:
            for r in rows:
                for p in r["parents_raw"]:
                    if p in node_set and p != r["node_id"]:
                        explicit_edges.add((p, r["node_id"]))
        edges = set(explicit_edges)
        if not edges:
            edges = infer_edges_from_time(rows, parent_window=parent_window)
        node_ids = [r["node_id"] for r in rows]
        if edges and is_dag(node_ids, edges):
            edges = transitive_reduction(node_ids, edges)
        if len(rows) < min_nodes or len(rows) > max_nodes:
            continue
        if edges and not is_dag(node_ids, edges):
            continue

        parent_map: Dict[str, List[str]] = defaultdict(list)
        succ_map: Dict[str, List[str]] = defaultdict(list)
        in_degree: Dict[str, int] = defaultdict(int)
        out_degree: Dict[str, int] = defaultdict(int)
        for u, v in edges:
            parent_map[v].append(u)
            succ_map[u].append(v)
            out_degree[u] += 1
            in_degree[v] += 1
        submit_candidates = [r["earliest_start"] for r in rows if np.isfinite(r["earliest_start"])] + [r["create_time"] for r in rows if np.isfinite(r["create_time"])]
        submit_time = float(min(submit_candidates)) if submit_candidates else 0.0
        nodes = []
        for r in rows:
            if not np.isfinite(r["base_duration"]) or r["base_duration"] <= 0:
                continue
            nodes.append({
                "id": r["node_id"],
                "raw_task_name": r["raw_task_name"],
                "parents": sorted(parent_map.get(r["node_id"], [])),
                "children": sorted(succ_map.get(r["node_id"], [])),
                "base_duration": float(r["base_duration"]),
                "parallelism": int(max(r["instance_num"], r["observed_instances"], 1)),
                "plan_cpu": None if np.isnan(r["plan_cpu"]) else float(r["plan_cpu"]),
                "plan_mem": None if np.isnan(r["plan_mem"]) else float(r["plan_mem"]),
                "earliest_start": None if np.isnan(r["earliest_start"]) else float(r["earliest_start"]),
                "latest_end": None if np.isnan(r["latest_end"]) else float(r["latest_end"]),
                "machine_count": int(r["machine_count"]),
                "in_degree": int(in_degree.get(r["node_id"], 0)),
                "out_degree": int(out_degree.get(r["node_id"], 0)),
            })
        if len(nodes) < min_nodes or len(nodes) > max_nodes:
            continue
        workflows.append({
            "workflow_id": f"job_{sanitize_token(str(job_name))}",
            "job_name": str(job_name),
            "submit_time": submit_time,
            "nodes": sorted(nodes, key=lambda x: (x["earliest_start"] is None, x["earliest_start"] if x["earliest_start"] is not None else math.inf, x["id"])),
            "edges_node": sorted(edges),
            "metadata": {
                "source": "AlibabaClusterTrace2018",
                "dataset_version": "alibaba_pcea_v2_balanced_profiles",
                "dag_mode": "explicit_task_name" if explicit_edges else "time_inferred",
                "num_nodes": len(nodes),
                "num_edges": int(sum(len(n["parents"]) for n in nodes)),
            },
        })
    return workflows


def choose_profile(node: dict, stats: dict, policy: str = "balanced_gpu") -> str:
    """Choose a rule-augmented compute profile.

    The original Alibaba trace does not contain GPU labels. The default
    balanced_gpu policy reserves CPU-only for short/light tasks and promotes
    medium/long or structurally important tasks to GPU/hybrid profiles. This
    avoids a benchmark dominated by whole-server idle power from CPU-only work.
    """
    duration = safe_float(node.get("base_duration"), stats["q40"])
    plan_mem = safe_float(node.get("plan_mem"), np.nan)
    degree = safe_int(node.get("in_degree"), 0) + safe_int(node.get("out_degree"), 0)

    q20 = stats.get("q20", stats["q40"])
    q25 = stats.get("q25", stats["q40"])
    q35 = stats.get("q35", stats["q40"])
    q40 = stats["q40"]
    q50 = stats.get("q50", stats["q60"])
    q60 = stats["q60"]
    q70 = stats.get("q70", stats["q80"])
    q75 = stats.get("q75", stats["q80"])
    q80 = stats["q80"]
    mem40 = stats.get("mem40", stats.get("mem50", 0.0))
    mem50 = stats.get("mem50", 0.0)
    mem60 = stats.get("mem60", mem50)
    mem75 = stats.get("mem75", mem60)
    has_mem = np.isfinite(plan_mem)

    if policy == "conservative":
        if duration >= q80 and has_mem and plan_mem >= mem50:
            return "gpu_train"
        if duration >= q60 and degree >= 3:
            return "hybrid_analytics"
        if duration >= q40:
            return "gpu_infer"
        return "cpu_only"

    if policy == "gpu_intensive":
        if duration >= q70 and (not has_mem or plan_mem >= mem40):
            return "gpu_train"
        if (duration >= q40 and degree >= 2) or (has_mem and plan_mem >= mem60 and duration >= q35):
            return "hybrid_analytics"
        if duration >= q20 or (has_mem and plan_mem >= mem50):
            return "gpu_infer"
        return "cpu_only"

    # Default balanced policy. CPU-only is limited to genuinely short/light
    # tasks; medium duration tasks become GPU inference candidates.
    if duration >= q75 and (not has_mem or plan_mem >= mem40):
        return "gpu_train"
    if (duration >= q50 and degree >= 2) or (has_mem and plan_mem >= mem75 and duration >= q40):
        return "hybrid_analytics"
    if duration >= q25 or (has_mem and plan_mem >= mem60 and duration >= q35) or (degree >= 2 and duration >= q35):
        return "gpu_infer"
    return "cpu_only"


def gpu_suitability_score(node: dict, stats: dict) -> float:
    duration = safe_float(node.get("base_duration"), stats["q40"])
    plan_mem = safe_float(node.get("plan_mem"), np.nan)
    degree = safe_int(node.get("in_degree"), 0) + safe_int(node.get("out_degree"), 0)
    dur_score = (duration - stats.get("q20", stats["q40"])) / max(stats.get("q80", 1.0) - stats.get("q20", 0.0), 1e-9)
    mem_score = 0.0
    if np.isfinite(plan_mem):
        mem_score = (plan_mem - stats.get("mem40", 0.0)) / max(stats.get("mem80", stats.get("mem75", 1.0)) - stats.get("mem40", 0.0), 1e-9)
    degree_score = min(1.0, degree / 4.0)
    return float(0.55 * max(0.0, dur_score) + 0.30 * max(0.0, mem_score) + 0.15 * degree_score)


def assign_profile_and_resources(
    node: dict,
    stats: dict,
    rng: random.Random,
    profile_override: Optional[str] = None,
    profile_policy: str = "balanced_gpu",
) -> dict:
    duration = safe_float(node.get("base_duration"), stats["q40"])
    plan_cpu = safe_float(node.get("plan_cpu"), np.nan)
    plan_mem = safe_float(node.get("plan_mem"), np.nan)
    q40, q80 = stats["q40"], stats["q80"]

    profile = profile_override or choose_profile(node, stats, policy=profile_policy)

    # CPU and memory primarily follow Alibaba plan_cpu/plan_mem statistics.
    cpu_buckets = {
        "cpu_only": [4, 8, 16, 24],
        "gpu_infer": [4, 8, 16],
        "hybrid_analytics": [8, 16, 24, 32],
        "gpu_train": [16, 24, 32, 48],
    }
    mem_buckets = {
        "cpu_only": [8, 16, 32, 64],
        "gpu_infer": [16, 32, 64],
        "hybrid_analytics": [32, 64, 128],
        "gpu_train": [64, 128, 256],
    }
    gpu_buckets = {
        "cpu_only": [0],
        "gpu_infer": [1],
        "hybrid_analytics": [1, 2],
        "gpu_train": [1, 2, 4],
    }
    cpu = bucket_by_quantiles(plan_cpu, stats["cpu_quantiles"], cpu_buckets[profile], default_idx=1)
    mem = bucket_by_quantiles(plan_mem, stats["mem_quantiles"], mem_buckets[profile], default_idx=1)
    gpu = choose_from_levels(gpu_buckets[profile], duration, q40, q80)

    profile_to_pref = {
        # CPU-only tasks should not prefer high-power 4090 nodes. They may still
        # be scheduled there if the policy decides to reuse an already-active
        # server, but the data-level affinity no longer encourages it.
        "cpu_only": ["SYN_T4_2GPU", "REAL_T4_4GPU", "SYN_T4_8GPU"],
        "gpu_infer": ["REAL_T4_4GPU", "SYN_T4_2GPU", "SYN_T4_8GPU"],
        "hybrid_analytics": ["REAL_4090_2GPU", "SYN_4090_4GPU", "SYN_T4_8GPU", "REAL_T4_4GPU"],
        "gpu_train": ["REAL_4090_2GPU", "SYN_4090_4GPU", "SYN_T4_8GPU"],
    }
    base_output_mb = {
        "cpu_only": 128.0,
        "gpu_infer": 512.0,
        "hybrid_analytics": 1024.0,
        "gpu_train": 2048.0,
    }
    output_mb = int(np.clip(
        base_output_mb[profile] * math.sqrt(max(duration, 1.0) / max(q40, 1.0)) * rng.uniform(0.7, 1.3),
        64,
        8192,
    ))
    gpu_intensity = GPU_INTENSITY_MAP.get(profile, 0.0)
    if int(gpu) <= 0:
        gpu_intensity = 0.0
    node.update({
        "profile": profile,
        "profile_name": PROFILE_NAME_MAP.get(profile, "generic"),
        "profile_id": PROFILE_ID_MAP.get(PROFILE_NAME_MAP.get(profile, "generic"), PROFILE_ID_MAP["generic"]),
        "cpu": int(cpu),
        "gpu": int(gpu),
        "mem": int(mem),
        "gpu_intensity": float(gpu_intensity),
        "output_mb": float(output_mb),
        "data_size": float(output_mb),  # MB semantics for PCEA communication energy.
        "preferred_types": profile_to_pref[profile],
        "preferred_type": profile_to_pref[profile][0],
        "affinity_bonus": AFFINITY_BONUS_MAP.get(profile, 0.0),
    })
    return node


def compute_global_stats(workflows: List[dict]) -> dict:
    durations, cpus, mems = [], [], []
    for wf in workflows:
        for n in wf["nodes"]:
            durations.append(safe_float(n.get("base_duration"), np.nan))
            cpus.append(safe_float(n.get("plan_cpu"), np.nan))
            mems.append(safe_float(n.get("plan_mem"), np.nan))
    return {
        "q20": approx_quantile(durations, 0.20, 5.0),
        "q25": approx_quantile(durations, 0.25, 8.0),
        "q35": approx_quantile(durations, 0.35, 10.0),
        "q40": approx_quantile(durations, 0.40, 10.0),
        "q50": approx_quantile(durations, 0.50, 15.0),
        "q60": approx_quantile(durations, 0.60, 20.0),
        "q70": approx_quantile(durations, 0.70, 30.0),
        "q75": approx_quantile(durations, 0.75, 35.0),
        "q80": approx_quantile(durations, 0.80, 40.0),
        "q85": approx_quantile(durations, 0.85, 50.0),
        "mem40": approx_quantile(mems, 0.40, 0.0),
        "mem50": approx_quantile(mems, 0.50, 0.0),
        "mem60": approx_quantile(mems, 0.60, 0.0),
        "mem70": approx_quantile(mems, 0.70, 0.0),
        "mem75": approx_quantile(mems, 0.75, 0.0),
        "mem80": approx_quantile(mems, 0.80, 0.0),
        "cpu_quantiles": [approx_quantile(cpus, q, np.nan) for q in (0.25, 0.50, 0.75)],
        "mem_quantiles": [approx_quantile(mems, q, np.nan) for q in (0.25, 0.50, 0.75)],
    }


def augment_workflows(
    workflows: List[dict],
    seed: int,
    profile_policy: str = "balanced_gpu",
    cpu_only_target: float = 0.25,
) -> List[dict]:
    rng = random.Random(seed)
    stats = compute_global_stats(workflows)

    flat_nodes: List[dict] = [n for wf in workflows for n in wf["nodes"]]
    selected_profiles: Dict[int, str] = {}
    cpu_candidates: List[Tuple[float, dict]] = []
    for n in flat_nodes:
        profile = choose_profile(n, stats, policy=profile_policy)
        selected_profiles[id(n)] = profile
        if profile == "cpu_only":
            cpu_candidates.append((gpu_suitability_score(n, stats), n))

    # Deterministic global cap for CPU-only profile. This addresses the observed
    # issue that too many CPU-only tasks can make active energy dominated by
    # whole-server idle power, reducing the value of CPU-GPU scheduling.
    if 0.0 <= cpu_only_target < 1.0 and flat_nodes:
        target_count = int(round(len(flat_nodes) * cpu_only_target))
        current_count = len(cpu_candidates)
        if current_count > target_count:
            promote_n = current_count - target_count
            cpu_candidates.sort(key=lambda x: (-x[0], str(x[1].get("id", ""))))
            for _score, node in cpu_candidates[:promote_n]:
                degree = safe_int(node.get("in_degree"), 0) + safe_int(node.get("out_degree"), 0)
                duration = safe_float(node.get("base_duration"), stats["q40"])
                if degree >= 2 and duration >= stats.get("q35", stats["q40"]):
                    selected_profiles[id(node)] = "hybrid_analytics"
                else:
                    selected_profiles[id(node)] = "gpu_infer"

    for wf in workflows:
        for n in wf["nodes"]:
            assign_profile_and_resources(
                n,
                stats,
                rng,
                profile_override=selected_profiles.get(id(n)),
                profile_policy=profile_policy,
            )
        wf["metadata"]["augmentation"] = {
            "version": "alibaba_pcea_v2_balanced_profiles",
            "seed": seed,
            "profile_policy": profile_policy,
            "cpu_only_target_share": cpu_only_target,
            "gpu_rule": "duration_mem_degree_profile_with_cpu_only_cap",
            "cpu_mem_rule": "plan_cpu_plan_mem_quantile_bucket",
            "output_mb_rule": "profile_duration_scaled_randomized",
            "cpu_only_gpu_intensity": 0.0,
            "cpu_only_preferred_types": ["SYN_T4_2GPU", "REAL_T4_4GPU", "SYN_T4_8GPU"],
            "data_size_unit": "MB",
        }
    return workflows


def dag_stats_and_deadline(workflow: dict, deadline_multiplier: float) -> dict:
    nodes = workflow["nodes"]
    node_ids = [n["id"] for n in nodes]
    node_map = {n["id"]: n for n in nodes}
    edges = set(tuple(e) for e in workflow.get("edges_node", []))
    order = topological_order(node_ids, edges)
    depth = {n: 0 for n in node_ids}
    cp = {n: float(node_map[n]["base_duration"]) for n in node_ids}
    preds = defaultdict(list)
    succs = defaultdict(list)
    for u, v in edges:
        succs[u].append(v)
        preds[v].append(u)
    for v in order:
        if preds[v]:
            depth[v] = max(depth[p] + 1 for p in preds[v])
            cp[v] = max(cp[p] + float(node_map[v]["base_duration"]) for p in preds[v])
    critical_path = max(cp.values(), default=1.0)
    total_work = sum(float(n["base_duration"]) for n in nodes)
    width_levels = defaultdict(int)
    for n, d in depth.items():
        width_levels[d] += 1
    width = max(width_levels.values(), default=len(nodes))
    makespan_target = max(deadline_multiplier * critical_path, critical_path + 10.0, 0.35 * total_work)
    return {
        "critical_path_estimate": float(critical_path),
        "total_work_estimate": float(total_work),
        "makespan_target": float(makespan_target),
        "depth": int(max(depth.values(), default=0) + 1),
        "width_estimate": int(width),
        "edge_density": float(len(edges) / max(1, len(nodes) * max(1, len(nodes) - 1))),
    }


def workflow_to_pcos_json(workflow: dict, deadline_multiplier: float) -> dict:
    nodes = sorted(workflow["nodes"], key=lambda item: (item.get("earliest_start") is None, item.get("earliest_start") if item.get("earliest_start") is not None else math.inf, str(item.get("id"))))
    id_map = {str(node["id"]): idx for idx, node in enumerate(nodes)}
    stats = dag_stats_and_deadline(workflow, deadline_multiplier)
    tasks = []
    for idx, node in enumerate(nodes):
        gpu = max(0, safe_int(node.get("gpu"), 0))
        gpu_intensity = 0.0 if gpu <= 0 else float(node.get("gpu_intensity", 0.0))
        preferred_type = str(node.get("preferred_type", ""))
        profile_name = str(node.get("profile_name", "generic"))
        if gpu <= 0 and (preferred_type.startswith("REAL_4090") or preferred_type.startswith("SYN_4090")):
            preferred_type = "SYN_T4_2GPU"
        tasks.append({
            "id": idx,
            "cpu": max(1, safe_int(node.get("cpu"), 8)),
            "gpu": gpu,
            "mem": max(1, safe_int(node.get("mem"), 32)),
            "base_duration": max(safe_float(node.get("base_duration"), 1.0), 1e-6),
            "gpu_intensity": gpu_intensity,
            "data_size": float(node.get("data_size", node.get("output_mb", 256.0))),
            "output_mb": float(node.get("output_mb", node.get("data_size", 256.0))),
            "profile_name": profile_name,
            "profile_id": int(node.get("profile_id", PROFILE_ID_MAP["generic"])),
            "preferred_type": preferred_type,
            "affinity_bonus": float(node.get("affinity_bonus", 0.0)),
            "parallelism": max(1, safe_int(node.get("parallelism"), 1)),
            "deadline": stats["makespan_target"],
            "raw_node_id": str(node.get("id")),
            "raw_task_name": str(node.get("raw_task_name") or node.get("id")),
            "plan_cpu": node.get("plan_cpu"),
            "plan_mem": node.get("plan_mem"),
        })
    edges = []
    for u, v in sorted(set(tuple(e) for e in workflow.get("edges_node", []))):
        if u in id_map and v in id_map:
            parent_node = nodes[id_map[u]]
            edges.append({
                "src": id_map[u],
                "dst": id_map[v],
                "data_mb": float(parent_node.get("output_mb", parent_node.get("data_size", 256.0))),
            })
    metadata = dict(workflow.get("metadata", {}))
    metadata.update(stats)
    metadata["submit_time"] = float(workflow.get("submit_time", 0.0))
    metadata["job_name"] = workflow.get("job_name")
    return {
        "workflow_id": str(workflow.get("workflow_id", "workflow_unknown")),
        "submit_time": float(workflow.get("submit_time", 0.0)),
        "tasks": tasks,
        "edges": edges,
        "makespan_target": stats["makespan_target"],
        "metadata": metadata,
    }


def split_workflows(workflows: List[dict], train_ratio: float, val_ratio: float, test_ratio: float, benchmark_ratio: float) -> Dict[str, List[dict]]:
    total = train_ratio + val_ratio + test_ratio + benchmark_ratio
    if total <= 0:
        raise ValueError("Split ratios must have positive sum.")
    # Normalize in case the user provided ratios that sum close to but not exactly 1.
    train_ratio, val_ratio, test_ratio, benchmark_ratio = [x / total for x in (train_ratio, val_ratio, test_ratio, benchmark_ratio)]
    ordered = sorted(workflows, key=lambda x: (x.get("submit_time", 0.0), x["workflow_id"]))
    n = len(ordered)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    # Ensure benchmark is non-empty when possible.
    train = ordered[:n_train]
    val = ordered[n_train:n_train + n_val]
    test = ordered[n_train + n_val:n_train + n_val + n_test]
    benchmark = ordered[n_train + n_val + n_test:]
    if n >= 4:
        for name, xs in [("train", train), ("val", val), ("test", test), ("benchmark", benchmark)]:
            if not xs:
                raise ValueError(f"Split {name} is empty; reduce min_nodes/max_jobs or adjust ratios.")
    return {"train": train, "val": val, "test": test, "benchmark": benchmark}


def write_jsonl(items: List[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_manifest(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["workflow_path", "format", "split", "repeat", "weight", "workflow_id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_bundle(
    workflows: List[dict],
    export_dir: str | Path,
    ratios: Tuple[float, float, float, float],
    deadline_multiplier: float,
    profile_policy: str,
    cpu_only_target: float,
) -> dict:
    export_root = Path(export_dir).resolve()
    export_root.mkdir(parents=True, exist_ok=True)
    splits = split_workflows(workflows, *ratios)
    all_rows: List[dict] = []
    split_rows: Dict[str, List[dict]] = {k: [] for k in splits}
    for split, items in splits.items():
        split_dir = export_root / "workflows" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for wf in items:
            payload = workflow_to_pcos_json(wf, deadline_multiplier)
            file_name = f"{sanitize_token(payload['workflow_id'])}.json"
            out_path = split_dir / file_name
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            row = {
                "workflow_path": str(out_path.relative_to(export_root)),
                "format": "json",
                "split": split,
                "repeat": 1,
                "weight": 1.0,
                "workflow_id": payload["workflow_id"],
            }
            all_rows.append(dict(row))
            split_rows[split].append(dict(row))
    write_manifest(export_root / "manifest.csv", all_rows)
    for split, rows in split_rows.items():
        write_manifest(export_root / f"{split}_manifest.csv", rows)
    build_config = {
        "dataset_version": "alibaba_pcea_v2_balanced_profiles",
        "splits": {k: len(v) for k, v in split_rows.items()},
        "data_size_unit": "MB",
        "deadline_multiplier": deadline_multiplier,
        "profile_policy": profile_policy,
        "cpu_only_target_share": cpu_only_target,
        "cpu_only_gpu_intensity": 0.0,
        "cpu_only_preferred_types": ["SYN_T4_2GPU", "REAL_T4_4GPU", "SYN_T4_8GPU"],
    }
    (export_root / "build_config.json").write_text(json.dumps(build_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"export_dir": str(export_root), "splits": {k: len(v) for k, v in split_rows.items()}}


def nominal_energy_proxy(node: dict) -> dict:
    preferred = str(node.get("preferred_type") or "SYN_T4_2GPU")
    spec = POWER_PROXY_SPECS.get(preferred, POWER_PROXY_SPECS["SYN_T4_2GPU"])
    duration = max(0.0, safe_float(node.get("base_duration"), 0.0))
    cpu = max(0.0, safe_float(node.get("cpu"), 0.0))
    gpu = max(0.0, safe_float(node.get("gpu"), 0.0))
    u_cpu = min(1.0, cpu / max(spec["cpu"], 1e-9))
    u_gpu = 0.0 if gpu <= 0 or spec["gpu"] <= 0 else min(1.0, gpu / max(spec["gpu"], 1e-9))
    cross = u_cpu * u_gpu
    idle = spec["p_idle"] * duration
    cpu_e = spec["p_cpu0"] * u_cpu * duration
    gpu_e = spec["p_gpu0"] * u_gpu * duration
    inter = (spec["k1"] * spec["p_cpu0"] * spec["p_gpu0"] * cross + spec["k2"] * spec["p_cpu0"] * spec["p_gpu0"] * (cross**2)) * duration
    return {"idle": idle, "cpu": cpu_e, "gpu": gpu_e, "interaction_signed": inter, "total": idle + cpu_e + gpu_e + inter}


def summarize(workflows: List[dict]) -> dict:
    dag_modes = defaultdict(int)
    node_counts, edge_counts = [], []
    profile_counts = defaultdict(int)
    profile_stats = defaultdict(lambda: {
        "count": 0,
        "duration_sum": 0.0,
        "cpu_work": 0.0,
        "gpu_work": 0.0,
        "mem_work": 0.0,
        "output_mb_sum": 0.0,
        "nominal_energy_proxy_j": 0.0,
        "nominal_idle_proxy_j": 0.0,
        "nominal_cpu_proxy_j": 0.0,
        "nominal_gpu_proxy_j": 0.0,
        "nominal_interaction_proxy_j": 0.0,
    })
    for wf in workflows:
        dag_modes[wf["metadata"].get("dag_mode", "unknown")] += 1
        node_counts.append(len(wf["nodes"]))
        edge_counts.append(len(wf.get("edges_node", [])))
        for n in wf["nodes"]:
            profile = n.get("profile", "unknown")
            profile_counts[profile] += 1
            duration = max(0.0, safe_float(n.get("base_duration"), 0.0))
            st = profile_stats[profile]
            st["count"] += 1
            st["duration_sum"] += duration
            st["cpu_work"] += max(0.0, safe_float(n.get("cpu"), 0.0)) * duration
            st["gpu_work"] += max(0.0, safe_float(n.get("gpu"), 0.0)) * duration
            st["mem_work"] += max(0.0, safe_float(n.get("mem"), 0.0)) * duration
            st["output_mb_sum"] += max(0.0, safe_float(n.get("output_mb"), 0.0))
            e = nominal_energy_proxy(n)
            st["nominal_energy_proxy_j"] += e["total"]
            st["nominal_idle_proxy_j"] += e["idle"]
            st["nominal_cpu_proxy_j"] += e["cpu"]
            st["nominal_gpu_proxy_j"] += e["gpu"]
            st["nominal_interaction_proxy_j"] += e["interaction_signed"]

    total_tasks = sum(profile_counts.values())
    total_duration = sum(v["duration_sum"] for v in profile_stats.values())
    total_energy_proxy = sum(v["nominal_energy_proxy_j"] for v in profile_stats.values())
    for _profile, st in profile_stats.items():
        st["count_share"] = st["count"] / max(total_tasks, 1)
        st["duration_share"] = st["duration_sum"] / max(total_duration, 1e-9)
        st["nominal_energy_proxy_share"] = st["nominal_energy_proxy_j"] / max(total_energy_proxy, 1e-9)

    def basic(xs: List[int | float]) -> dict:
        if not xs:
            return {"count": 0}
        arr = np.asarray(xs, dtype=float)
        return {
            "count": int(arr.size),
            "min": float(arr.min()),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }
    return {
        "num_workflows": len(workflows),
        "dag_mode_counts": dict(dag_modes),
        "node_count_stats": basic(node_counts),
        "edge_count_stats": basic(edge_counts),
        "profile_counts": dict(profile_counts),
        "profile_stats": {k: dict(v) for k, v in sorted(profile_stats.items())},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pcos-native Alibaba PCEA dataset.")
    p.add_argument("--batch-task", default="datasets/alibaba_raw/batch_task.csv", help="Path to Alibaba batch_task.csv")
    p.add_argument("--batch-instance", default="datasets/alibaba_raw/batch_instance.csv", help="Path to Alibaba batch_instance.csv")
    p.add_argument("--export-dir", default="datasets/alibaba_pcea/processed", help="Output directory for pcos workflow JSON + manifests")
    p.add_argument("--output-jsonl", default="", help="Optional raw augmented workflow JSONL path")
    p.add_argument("--stats-json", default="", help="Optional stats JSON path; defaults to <export-dir>/stats.json")
    p.add_argument("--max-jobs", type=int, default=0, help="Optional cap on jobs for local testing")
    p.add_argument("--chunk-size", type=int, default=2_000_000, help="Chunk size for streaming batch_instance")
    p.add_argument("--min-nodes", type=int, default=5)
    p.add_argument("--max-nodes", type=int, default=150)
    p.add_argument("--parent-window", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--benchmark-ratio", type=float, default=0.05)
    p.add_argument("--deadline-multiplier", type=float, default=1.50)
    p.add_argument(
        "--profile-policy",
        choices=["conservative", "balanced_gpu", "gpu_intensive"],
        default="balanced_gpu",
        help="Rule set for GPU/profile augmentation. balanced_gpu reduces CPU-only dominance while remaining deterministic.",
    )
    p.add_argument(
        "--cpu-only-target",
        type=float,
        default=0.25,
        help="Maximum target share for CPU-only tasks after augmentation. Set negative to disable the cap.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    batch_task = Path(args.batch_task)
    batch_instance = Path(args.batch_instance)
    if not batch_task.exists():
        raise FileNotFoundError(f"Missing batch_task.csv: {batch_task}")
    if not batch_instance.exists():
        raise FileNotFoundError(f"Missing batch_instance.csv: {batch_instance}")
    rng = random.Random(args.seed)
    task_df = load_batch_task(batch_task, max_jobs=args.max_jobs if args.max_jobs > 0 else None)
    if task_df.empty:
        raise RuntimeError("No usable rows remained after loading batch_task.csv")
    selected_pairs = set(zip(task_df["job_name"].astype(str), task_df["task_name"].astype(str)))
    selected_jobs = set(task_df["job_name"].astype(str).unique().tolist())
    runtime_stats = scan_batch_instance(batch_instance, selected_pairs, selected_jobs, args.chunk_size, rng)
    workflows = build_workflows(task_df, runtime_stats, args.min_nodes, args.max_nodes, args.parent_window, require_runtime=True)
    if not workflows:
        raise RuntimeError("No workflows were built. Check raw files, status filtering, and node thresholds.")
    workflows = augment_workflows(
        workflows,
        seed=args.seed,
        profile_policy=args.profile_policy,
        cpu_only_target=args.cpu_only_target,
    )
    export_info = export_bundle(
        workflows,
        args.export_dir,
        (args.train_ratio, args.val_ratio, args.test_ratio, args.benchmark_ratio),
        args.deadline_multiplier,
        profile_policy=args.profile_policy,
        cpu_only_target=args.cpu_only_target,
    )
    stats = summarize(workflows)
    stats.update(export_info)
    stats["raw_inputs"] = {"batch_task": str(batch_task), "batch_instance": str(batch_instance)}
    stats["augmentation_config"] = {
        "profile_policy": args.profile_policy,
        "cpu_only_target_share": args.cpu_only_target,
        "cpu_only_gpu_intensity": 0.0,
        "cpu_only_preferred_types": ["SYN_T4_2GPU", "REAL_T4_4GPU", "SYN_T4_8GPU"],
        "data_size_unit": "MB",
    }
    stats_json = Path(args.stats_json) if args.stats_json else Path(args.export_dir) / "stats.json"
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_jsonl:
        write_jsonl(workflows, args.output_jsonl)
    print(json.dumps({
        "export_dir": export_info["export_dir"],
        "stats_json": str(stats_json),
        "num_workflows": stats["num_workflows"],
        "splits": export_info["splits"],
        "dag_mode_counts": stats["dag_mode_counts"],
        "profile_counts": stats["profile_counts"],
        "profile_count_shares": {k: round(v.get("count_share", 0.0), 4) for k, v in stats.get("profile_stats", {}).items()},
        "profile_duration_shares": {k: round(v.get("duration_share", 0.0), 4) for k, v in stats.get("profile_stats", {}).items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
