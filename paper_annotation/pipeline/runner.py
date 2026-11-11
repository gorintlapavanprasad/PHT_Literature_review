"""Parallel job runner for (paper, model) annotation jobs.

Different providers are independent services, so they run side by side; inside one provider the
number of simultaneous jobs is capped (rate limits). A failing job is logged and reported at the
end - it never aborts the rest of the run, and finished jobs are never redone (results on disk are
skipped by the caller).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable, TypeVar

J = TypeVar("J")


@dataclass
class RunSummary:
    succeeded: list = field(default_factory=list)
    failed: list = field(default_factory=list)   # (job, error string)
    seconds: float = 0.0
    peak_concurrency: dict = field(default_factory=dict)  # provider -> max simultaneous jobs observed


def run_jobs(
    jobs: Iterable[J],
    work: Callable[[J], object],
    provider_of: Callable[[J], Hashable],
    per_provider_limit: int = 3,
    max_workers: int | None = None,
) -> RunSummary:
    jobs = list(jobs)
    logger = logging.getLogger(__name__)
    providers = {provider_of(j) for j in jobs}
    sems = {p: threading.Semaphore(max(1, per_provider_limit)) for p in providers}
    running: dict = defaultdict(int)
    peak: dict = defaultdict(int)
    lock = threading.Lock()
    summary = RunSummary()
    t0 = time.perf_counter()

    def _run(job: J):
        prov = provider_of(job)
        with sems[prov]:
            with lock:
                running[prov] += 1
                peak[prov] = max(peak[prov], running[prov])
            try:
                return work(job)
            finally:
                with lock:
                    running[prov] -= 1

    workers = max_workers or max(1, per_provider_limit * max(1, len(providers)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, j): j for j in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                fut.result()
                summary.succeeded.append(job)
            except Exception as exc:  # noqa: BLE001 - one bad job must not stop the run
                logger.error("Job failed: %s: %s", job, exc)
                summary.failed.append((job, f"{type(exc).__name__}: {exc}"))
    summary.seconds = time.perf_counter() - t0
    summary.peak_concurrency = dict(peak)
    return summary
