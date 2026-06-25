from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for r in rows for k in r.keys() if isinstance(r.get(k), (int, float))})
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if isinstance(r.get(k), (int, float))]
        if vals:
            out[k + "_mean"] = mean(vals)
            out[k + "_std"] = pstdev(vals) if len(vals) > 1 else 0.0
    return out


def write_csv(path: str | Path, rows: List[Dict[str, float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
