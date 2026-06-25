from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from pcos.core.entities import Task, Workflow, estimate_duration, default_cluster_specs


def _edge_tuple(edge) -> Tuple[int, int]:
    if isinstance(edge, dict):
        return int(edge.get("src", edge.get("source"))), int(edge.get("dst", edge.get("target")))
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        return int(edge[0]), int(edge[1])
    raise ValueError(f"Unsupported edge format: {edge!r}")


def load_workflow_json(path: str | Path) -> Workflow:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks: List[Task] = []
    for item in data.get("tasks", []):
        tasks.append(
            Task(
                id=int(item["id"]),
                cpu=float(item.get("cpu", 1.0)),
                gpu=float(item.get("gpu", 0.0)),
                mem=float(item.get("mem", 1.0)),
                base_duration=float(item.get("base_duration", item.get("duration", 1.0))),
                gpu_intensity=float(item.get("gpu_intensity", 0.0)),
                data_size=float(item.get("data_size", item.get("output_mb", 0.0))),
                profile_name=str(item.get("profile_name", item.get("profile", "generic"))),
                profile_id=int(item.get("profile_id", 0)),
                preferred_type=str(item.get("preferred_type", "")),
                affinity_bonus=float(item.get("affinity_bonus", 0.0)),
                parallelism=float(item.get("parallelism", 1.0)),
                deadline=item.get("deadline"),
            )
        )
    tasks.sort(key=lambda t: t.id)
    if tasks and any(t.id != i for i, t in enumerate(tasks)):
        id_to_new = {t.id: i for i, t in enumerate(tasks)}
        for i, t in enumerate(tasks):
            t.id = i
        edges = [(id_to_new[a], id_to_new[b]) for a, b in (_edge_tuple(e) for e in data.get("edges", []))]
    else:
        edges = [_edge_tuple(e) for e in data.get("edges", [])]
    wf = Workflow(
        workflow_id=str(data.get("workflow_id", path.stem)),
        tasks=tasks,
        edges=edges,
        metadata=data.get("metadata", {}),
        makespan_target=data.get("makespan_target") or data.get("metadata", {}).get("makespan_target"),
    )
    enrich_workflow(wf)
    return wf


def enrich_workflow(wf: Workflow) -> Workflow:
    n = len(wf.tasks)
    preds = [[] for _ in range(n)]
    succs = [[] for _ in range(n)]
    for a, b in wf.edges:
        if 0 <= a < n and 0 <= b < n and a != b:
            preds[b].append(a)
            succs[a].append(b)
    for i, t in enumerate(wf.tasks):
        t.preds = sorted(set(preds[i]))
        t.succs = sorted(set(succs[i]))
    # Compute rough upward/downward ranks using average duration over the default cluster.
    specs = default_cluster_specs()
    avg_dur = [sum(estimate_duration(t, s) for s in specs if t.gpu <= s.gpu + 1e-9) / max(1, sum(1 for s in specs if t.gpu <= s.gpu + 1e-9)) for t in wf.tasks]
    up = [None] * n
    def rank_u(i: int) -> float:
        if up[i] is not None:
            return up[i]
        if not wf.tasks[i].succs:
            up[i] = avg_dur[i]
        else:
            up[i] = avg_dur[i] + max(rank_u(j) for j in wf.tasks[i].succs)
        return up[i]
    for i in reversed(range(n)):
        wf.tasks[i].upward_rank = rank_u(i)
    down = [0.0] * n
    order = topological_order(wf)
    for i in order:
        for j in wf.tasks[i].succs:
            down[j] = max(down[j], down[i] + avg_dur[i])
    for i, t in enumerate(wf.tasks):
        t.downward_rank = down[i]
    cp = max((t.upward_rank for t in wf.tasks), default=1.0)
    if wf.makespan_target is None:
        wf.makespan_target = float(max(cp * 1.35, cp + 10.0))
    for t in wf.tasks:
        if t.deadline is None:
            # Conservative per-task latest finish proxy.
            t.deadline = float(wf.makespan_target)
    return wf


def topological_order(wf: Workflow) -> List[int]:
    indeg = {t.id: len(t.preds) for t in wf.tasks}
    ready = [t.id for t in wf.tasks if indeg[t.id] == 0]
    order: List[int] = []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for j in wf.tasks[i].succs:
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
    if len(order) != len(wf.tasks):
        # Fallback for malformed DAGs: keep missing nodes in id order.
        seen = set(order)
        order.extend([t.id for t in wf.tasks if t.id not in seen])
    return order
