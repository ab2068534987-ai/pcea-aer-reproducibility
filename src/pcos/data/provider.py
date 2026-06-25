from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Optional

from .workflow_io import load_workflow_json


class WorkflowProvider:
    """Loads Alibaba-enhanced workflow JSON files from a manifest or a directory."""

    def __init__(self, root: str | Path, split: str = "train", seed: int = 0, limit: Optional[int] = None):
        self.root = Path(root)
        self.split = split
        self.rng = random.Random(seed)
        self.paths = self._discover_paths(split)
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"No workflow JSON files found under {self.root!s} for split={split!r}")

    def _discover_paths(self, split: str) -> List[Path]:
        manifest_candidates = [
            self.root / f"{split}_manifest.csv",
            self.root / "manifest.csv",
        ]
        paths: List[Path] = []
        for mf in manifest_candidates:
            if mf.exists():
                with mf.open(newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("split") and row["split"] != split and mf.name == "manifest.csv":
                            continue
                        raw = row.get("path") or row.get("workflow_path") or row.get("file") or row.get("json_path")
                        if raw:
                            p = Path(raw)
                            if not p.is_absolute():
                                p = self.root / p
                            if p.exists():
                                paths.append(p)
                        elif row.get("workflow_id"):
                            p = self.root / "workflows" / split / f"{row['workflow_id']}.json"
                            if p.exists():
                                paths.append(p)
                if paths:
                    return sorted(set(paths))
        return sorted((self.root / "workflows" / split).glob("*.json"))

    def sample(self):
        return load_workflow_json(self.rng.choice(self.paths))

    def iter_workflows(self, limit: Optional[int] = None):
        xs = self.paths if limit is None else self.paths[:limit]
        for p in xs:
            yield load_workflow_json(p)

    def __len__(self) -> int:
        return len(self.paths)
