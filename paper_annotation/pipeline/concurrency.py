"""Concurrency utilities."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import threading
from typing import Callable, Iterable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_bounded(
    items: Iterable[T],
    worker: Callable[[T], R],
    max_workers: int,
    log_semaphore: bool = True,
) -> List[R]:
    """Run worker over items with bounded concurrency."""
    logger = logging.getLogger(__name__)
    semaphore = threading.Semaphore(max_workers)

    def _wrapped(item: T) -> R:
        if log_semaphore:
            logger.info("Waiting for semaphore")
        semaphore.acquire()
        if log_semaphore:
            logger.info("Acquired semaphore")
        try:
            return worker(item)
        finally:
            semaphore.release()
            if log_semaphore:
                logger.info("Released semaphore")

    results: List[R] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_wrapped, item) for item in items]
        for future in as_completed(futures):
            results.append(future.result())
    return results


if __name__ == "__main__":
    data = [1, 2, 3]
    output = run_bounded(data, lambda x: x * 2, max_workers=2)
    print(output)
