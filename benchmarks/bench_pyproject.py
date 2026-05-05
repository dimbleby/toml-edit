#!/usr/bin/env python3

"""Parse-throughput benchmark for a pyproject.toml-shaped input.

Times `tomlrt.loads` over the repository's own `pyproject.toml`,
which is broadly representative of the workload `tomlrt` is expected
to handle in real projects.

Usage:

    uv run python benchmarks/bench_pyproject.py
"""

from __future__ import annotations

import gc
import statistics
import sys
import time
from pathlib import Path

import tomlrt

REPO_ROOT = Path(__file__).resolve().parent.parent


def _time(text: str, *, repeats: int) -> tuple[float, float]:
    timings: list[float] = []
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            tomlrt.loads(text)
            t1 = time.perf_counter()
        finally:
            gc.enable()
        timings.append(t1 - t0)
    return min(timings), statistics.median(timings)


def main() -> None:
    """Run the pyproject.toml parse benchmark."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    nbytes = len(text.encode("utf-8"))

    print(f"tomlrt : {tomlrt.__file__}")
    print(f"python : {sys.version.split()[0]}")
    print(f"input  : pyproject.toml ({nbytes / 1024:.1f} KiB)")
    print()

    best, median = _time(text, repeats=2000)
    mb_per_s = (nbytes / best) / (1024 * 1024)
    print(
        f"  parse  best {best * 1e6:7.1f} us   "
        f"median {median * 1e6:7.1f} us   "
        f"{mb_per_s:6.2f} MiB/s",
    )


if __name__ == "__main__":
    main()
