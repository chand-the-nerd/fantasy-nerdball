"""How much memory a run costs, and how many would fit.

The question "how many concurrent users" reduces, once runs can be
parallelised at all, to "how many optimisations fit in the container at
once". That needs two numbers nobody can guess: what a run actually
costs on top of an idle process, and how much the container is allowed.
Both are read here, from /proc and from the cgroup, so the answer comes
from the deployment rather than from an estimate.

Everything degrades to None rather than raising. These are diagnostics;
a container that reports its memory differently is not a reason for a
run to fail.
"""

from __future__ import annotations

import threading
import time

PAGE_SIZE = 4096


def rss_mb() -> float | None:
    """Resident memory of this process, in MB."""
    try:
        with open("/proc/self/statm") as handle:
            resident = int(handle.read().split()[1])
        return round(resident * PAGE_SIZE / 1024 / 1024, 1)
    except Exception:
        return None


def memory_limit_mb() -> float | None:
    """What the container is allowed, in MB.

    cgroup v2 first, then v1. "max" means unlimited, which on Railway
    means the plan's limit isn't expressed here — treated as unknown
    rather than as infinite headroom.
    """
    candidates = (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    for path in candidates:
        try:
            with open(path) as handle:
                raw = handle.read().strip()
        except Exception:
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        # cgroup v1 reports an enormous sentinel for "no limit".
        if value > 1 << 50:
            return None
        return round(value / 1024 / 1024, 1)
    return None


class MemorySampler:
    """Watches this process's memory for the length of a run.

    Peak matters rather than the value at the end: the optimiser builds
    its dataframes, solves, and lets most of it go, so a reading taken
    afterwards would suggest a run is far cheaper than it is to hold.
    """

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.baseline = rss_mb()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "MemorySampler":
        if self.baseline is None:
            return self
        self._thread = threading.Thread(
            target=self._loop, name="memory-sampler", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            current = rss_mb()
            if current is not None and (
                self.peak is None or current > self.peak
            ):
                self.peak = current
            # Sleeping on the event means stopping is immediate rather
            # than up to a full interval late.
            self._stop.wait(self.interval)
            time.sleep(0)

    @property
    def cost_mb(self) -> float | None:
        """What the run added on top of an already-warm process."""
        if self.peak is None or self.baseline is None:
            return None
        return round(max(0.0, self.peak - self.baseline), 1)


def parallel_capacity(
    run_cost_mb: float | None, reserve_fraction: float = 0.2
) -> int | None:
    """How many runs would fit at once, on memory alone.

    A fifth of the container is held back by default. Running a box to
    its stated limit is how you find out what the OOM killer does to a
    queue, and the web layer still needs room to answer requests while
    the runs are going.

    Memory is only one of the constraints — CPU and the single worker
    thread are the others — so this is a ceiling, not a target.
    """
    limit = memory_limit_mb()
    baseline = rss_mb()
    if not limit or not run_cost_mb or baseline is None or run_cost_mb <= 0:
        return None
    usable = (limit * (1 - reserve_fraction)) - baseline
    if usable <= 0:
        return 0
    return max(0, int(usable // run_cost_mb))
