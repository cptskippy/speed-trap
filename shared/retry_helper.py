"""
retry_helper.py

Provides retry-with-backoff functionality to replace hard-coded time.sleep() delays.
"""
import time
from typing import Callable, Any


def retry_with_backoff(
    func,
    *args,
    max_attempts=8,
    base_delay=2,
    max_delay=60,
    exceptions=(Exception,),
    on_empty=None,
    **kwargs,
):
    """
    Retry a function call with exponential backoff.

    Attempts the operation immediately, then retries with exponentially
    increasing delays if it fails or returns empty/unacceptable results.

    Args:
        func: Callable to retry
        *args: Positional arguments for func
        max_attempts: Maximum number of attempts before giving up
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exceptions: Tuple of exception types to catch and retry on
        on_empty: Optional predicate function. If func succeeds but
                  on_empty(func_result) returns True, the result is treated
                  as a failure and retried. (e.g., lambda r: not r for empty dict)
        **kwargs: Keyword arguments for func

    Returns:
        Result of the function call

    Raises:
        The last exception if all attempts fail, or ValueError if on_empty
        kept triggering past max_attempts.
    """
    last_exception = None

    for attempt in range(max_attempts):
        try:
            result = func(*args, **kwargs)

            if on_empty is not None and on_empty(result):
                raise ValueError(f"Result was empty/unacceptable (attempt {attempt + 1}/{max_attempts})")

            if attempt > 0:
                print(f"  Succeeded on attempt {attempt + 1}/{max_attempts}")

            return result

        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                print(f"  Attempt {attempt + 1}/{max_attempts} failed: {e}")
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  All {max_attempts} attempts exhausted.")

    if last_exception is not None:
        raise last_exception
    raise RuntimeError(f"All {max_attempts} retry attempts failed without a recorded exception")
