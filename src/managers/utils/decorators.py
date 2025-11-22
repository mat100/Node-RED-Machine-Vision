"""
Utility decorators and helpers for common patterns.
"""

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer() -> Generator[dict, None, None]:
    """
    Context manager to measure execution time.

    Usage:
        with timer() as t:
            # ... code to time ...
            pass
        print(f"Took {t['ms']}ms")

    Yields:
        Dictionary with 'ms' key containing processing time in milliseconds
    """
    result = {"ms": 0}
    start_time = time.time()
    try:
        yield result
    finally:
        result["ms"] = int((time.time() - start_time) * 1000)
