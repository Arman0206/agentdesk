"""Minimal exponential-backoff retry decorator (avoids an extra dependency)."""
import functools, time


def retry(attempts: int = 3, base_delay: float = 0.5):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            for i in range(attempts):
                try:
                    return fn(*a, **kw)
                except Exception:
                    if i == attempts - 1:
                        raise
                    time.sleep(base_delay * (2 ** i))
        return wrapper
    return deco
