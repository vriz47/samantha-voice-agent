"""Stage timing for the voice pipeline, so latency work targets the real bottleneck."""

from __future__ import annotations

import time
from contextlib import contextmanager


class Profiler:
    def __init__(self, label: str = "turn", t0: float | None = None) -> None:
        self.label = label
        self.t0 = t0 if t0 is not None else time.monotonic()
        self.events: list[tuple[float, str]] = []
        self.durations: dict[str, float] = {}

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def point(self, name: str) -> float:
        self.events.append((time.monotonic() - self.t0, name))
        return time.monotonic() - self.t0

    @contextmanager
    def span(self, name: str):
        self.point(f"{name} start")
        start = time.monotonic()
        try:
            yield
        finally:
            d = time.monotonic() - start
            self.durations[name] = self.durations.get(name, 0.0) + d
            self.point(f"{name} end ({d:.3f}s)")

    def report(self) -> str:
        lines = [f"== {self.label} total {self.elapsed:.3f}s =="]
        prev = 0.0
        for t, name in self.events:
            lines.append(f"  +{t:7.3f}s  (+{t - prev:6.3f})  {name}")
            prev = t
        lines.append("  -- stage totals --")
        for name, d in sorted(self.durations.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:<22} {d:7.3f}s")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "total_s": round(self.elapsed, 4),
            "durations_s": {k: round(v, 4) for k, v in self.durations.items()},
            "timeline": [[round(t, 4), n] for t, n in self.events],
        }