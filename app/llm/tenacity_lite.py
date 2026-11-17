"""Minimal exponential-backoff retry decorator (avoids an extra dependency)."""
import functools, time


def retry(attempts: int = 3, base_delay: float = 0.5, retry_if=None):
    """retry_if(exc) -> bool decides whether an exception is worth retrying (default: all)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            for i in range(attempts):
                try:
                    return fn(*a, **kw)
                except Exception as e:
                    if i == attempts - 1 or (retry_if and not retry_if(e)):
                        raise
                    time.sleep(base_delay * (2 ** i))
        return wrapper
    return deco
