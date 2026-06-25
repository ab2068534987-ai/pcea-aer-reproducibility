from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class EnvelopePoint:
    time_s: float
    power_w: float


class PowerEnvelopeProvider:
    """External electricity-side power envelope B(t).

    The envelope is a scenario, not a claim that grid data and Alibaba workloads
    are temporally/geographically co-located. Synthetic envelopes are the default.
    """

    def __init__(
        self,
        cluster_full_power_w: float,
        scenario: str = "peak_valley",
        slot_s: float = 300.0,
        b_min_ratio: float = 0.55,
        b_max_ratio: float = 0.90,
        period_s: float = 24 * 3600.0,
        phase_mode: str = "random",
        csv_path: Optional[str] = None,
        seed: int = 0,
    ):
        self.cluster_full_power_w = float(cluster_full_power_w)
        self.scenario = scenario
        self.slot_s = float(slot_s)
        self.b_min = float(b_min_ratio) * self.cluster_full_power_w
        self.b_max = float(b_max_ratio) * self.cluster_full_power_w
        self.period_s = float(period_s)
        self.phase_mode = phase_mode
        self.rng = random.Random(seed)
        self.points: List[EnvelopePoint] = []
        if csv_path:
            self._load_csv(csv_path)

    def _load_csv(self, csv_path: str) -> None:
        path = Path(csv_path)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        raw = []
        for r in rows:
            t = float(r.get("time_s", len(raw) * self.slot_s))
            if "power_envelope_w" in r:
                p = float(r["power_envelope_w"])
            elif "value" in r:
                p = float(r["value"])
            else:
                raise ValueError("CSV must include power_envelope_w or value")
            raw.append((t, p))
        if not raw:
            return
        vals = [p for _, p in raw]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            self.points = [EnvelopePoint(t, self.b_max) for t, _ in raw]
        else:
            self.points = [EnvelopePoint(t, self.b_min + (p - lo) / (hi - lo) * (self.b_max - self.b_min)) for t, p in raw]
        self.period_s = max(t for t, _ in raw) + self.slot_s

    def sample_phase(self, submit_time: Optional[float] = None) -> float:
        if self.phase_mode == "submit_time" and submit_time is not None:
            return float(submit_time) % self.period_s
        if self.phase_mode == "zero":
            return 0.0
        return self.rng.uniform(0.0, self.period_s)

    def value(self, episode_time_s: float, phase_s: float = 0.0) -> float:
        t = (phase_s + episode_time_s) % self.period_s
        if self.points:
            return self._value_from_points(t)
        ratio = self._synthetic_ratio(t)
        return self.b_min + ratio * (self.b_max - self.b_min)

    def next_change_after(self, episode_time_s: float, phase_s: float = 0.0) -> float:
        t_abs = phase_s + episode_time_s
        if self.points:
            next_slot = math.floor(t_abs / self.slot_s + 1.0) * self.slot_s
        else:
            next_slot = math.floor(t_abs / self.slot_s + 1.0) * self.slot_s
        return max(episode_time_s + 1e-6, next_slot - phase_s)

    def _value_from_points(self, t: float) -> float:
        if not self.points:
            return self.b_max
        idx = int(t // self.slot_s) % len(self.points)
        return self.points[idx].power_w

    def _synthetic_ratio(self, t: float) -> float:
        h = (t % self.period_s) / 3600.0
        if self.scenario == "flat":
            return 0.72
        if self.scenario == "renewable_window":
            # Solar bump around noon and mild wind bump at night.
            solar = max(0.0, math.sin((h - 6.0) / 12.0 * math.pi))
            wind = 0.5 + 0.5 * math.cos((h - 2.0) / 24.0 * 2.0 * math.pi)
            return min(1.0, max(0.0, 0.25 + 0.55 * solar + 0.20 * wind))
        # peak_valley default: lower envelope during grid stress windows.
        if 0 <= h < 7:
            r = 0.90
        elif 7 <= h < 10:
            r = 0.75
        elif 10 <= h < 14:
            r = 0.60
        elif 14 <= h < 18:
            r = 0.75
        elif 18 <= h < 22:
            r = 0.55
        else:
            r = 0.85
        return (r - 0.55) / (0.90 - 0.55)
