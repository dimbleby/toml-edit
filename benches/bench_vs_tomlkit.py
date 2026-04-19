#!/usr/bin/env python3
"""Quick benchmark: toml_edit vs tomlkit on parse/dump round-trips.

Run as a script: ``python benches/bench_vs_tomlkit.py``.
Not part of the test suite; not required to be installed in CI.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

import tomlkit

import toml_edit

if TYPE_CHECKING:
    from collections.abc import Callable

_CORPUS = """
title = "Benchmarking corpus"

[server]
host = "example.com"
port = 8080
keepalive = true
timeout = 12.5

[server.tls]
cert = "/etc/ssl/cert.pem"
key  = "/etc/ssl/key.pem"
ciphers = [
    "TLS_AES_256_GCM_SHA384",
    "TLS_CHACHA20_POLY1305_SHA256",
    "TLS_AES_128_GCM_SHA256",
]

[[users]]
name = "alice"
roles = ["admin", "ops"]

[[users]]
name = "bob"
roles = ["dev"]

[[users]]
name = "carol"
roles = ["dev", "ops"]

[deeply.nested.section.with.many.parts]
key = 42
"""


def _bench(name: str, fn: Callable[[], object], *, iters: int = 200) -> tuple[str, float]:
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    median_us = statistics.median(samples) * 1e6
    return name, median_us


def main() -> None:
    src = _CORPUS.strip() + "\n"
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    pyproject_src = pyproject.read_text()

    print(f"Corpus size: synthetic={len(src):,} bytes, pyproject={len(pyproject_src):,} bytes")
    print()
    print(f"{'Workload':<35} {'toml_edit (us)':>12} {'tomlkit (us)':>14} {'speedup':>10}")
    print("-" * 72)

    for label, payload in [("synthetic", src), ("pyproject.toml", pyproject_src)]:
        for op_name, te_fn, tk_fn in [
            ("parse", lambda p=payload: toml_edit.parse(p),
                     lambda p=payload: tomlkit.parse(p)),
            ("parse+dump", lambda p=payload: toml_edit.dumps(toml_edit.parse(p)),
                          lambda p=payload: tomlkit.dumps(tomlkit.parse(p))),
        ]:
            _, t_us = _bench("toml_edit", te_fn)
            _, k_us = _bench("tomlkit", tk_fn)
            speedup = k_us / t_us if t_us > 0 else float("inf")
            print(f"{label:<20}{op_name:<15} {t_us:>12.1f} {k_us:>14.1f} {speedup:>9.2f}x")


if __name__ == "__main__":
    main()
