"""
Temporal gating for CV facts — require sustained cues before asserting observations.
"""

from __future__ import annotations

import time


class FactTemporalGate:
    """Confirm a fact only after `min_seconds` of continuous raw detection."""

    def __init__(self, min_seconds: float):
        self.min_seconds = max(0.05, min_seconds)
        self._active_since: float | None = None
        self._confirmed = False

    def reset(self) -> None:
        self._active_since = None
        self._confirmed = False

    def update(self, raw: bool, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        if raw:
            if self._active_since is None:
                self._active_since = now
            elif now - self._active_since >= self.min_seconds:
                self._confirmed = True
        else:
            self._active_since = None
            self._confirmed = False
        return self._confirmed

    def progress(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        if self._active_since is None:
            return 0.0
        return min(1.0, (now - self._active_since) / self.min_seconds)


class FactSessionAccumulator:
    """Tracks total frames and seconds each confirmed fact was active in a session."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._active: dict[str, bool] = {}
        self.stats: dict[str, dict[str, float | int]] = {
            "eyes_closed": {"frames": 0, "seconds": 0.0, "periods": 0},
            "looking_away": {"frames": 0, "seconds": 0.0, "periods": 0},
            "head_down": {"frames": 0, "seconds": 0.0, "periods": 0},
            "phone_suspected": {"frames": 0, "seconds": 0.0, "periods": 0},
            "no_face": {"frames": 0, "seconds": 0.0, "periods": 0},
        }

    def tick(self, facts: dict[str, bool], dt: float) -> None:
        dt = max(dt, 0.05)
        for key in self.stats:
            active = bool(facts.get(key))
            was_active = self._active.get(key, False)
            if active:
                self.stats[key]["frames"] = int(self.stats[key]["frames"]) + 1
                self.stats[key]["seconds"] = float(self.stats[key]["seconds"]) + dt
                if not was_active:
                    self.stats[key]["periods"] = int(self.stats[key]["periods"]) + 1
            self._active[key] = active

    def get_summary(self) -> dict[str, dict[str, float | int]]:
        return {k: dict(v) for k, v in self.stats.items()}
