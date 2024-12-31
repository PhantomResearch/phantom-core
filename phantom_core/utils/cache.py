from time import time
from typing import TypeVar, Callable
from functools import wraps

T = TypeVar('T')


class TTLCache(dict[str, tuple[float, T]]):
    """
    A time-to-live (TTL) cache implementation that extends dict.
    
    This cache stores values with timestamps and automatically expires entries after
    a specified TTL period. Values are stored as tuples of (timestamp, value).
    
    The cache can optionally use a callback function to handle cache misses by
    generating missing values on demand.

    Args:
        ttl_seconds (int): Time-to-live in seconds before entries expire. Defaults to 300.
        missing_callback (Callable[[str], T] | None): Optional function to generate missing values.
            Takes the missing key as argument and returns the value. Defaults to None.
    """

    def __init__(self, ttl_seconds: int = 300, missing_callback: Callable[[str], T] | None = None):
        super().__init__()
        self.ttl = ttl_seconds
        self.missing_callback = missing_callback
    
    def __getitem__(self, key: str) -> T:

        # try to get the value from the cache
        try:
            # is the key in the cache?
            timestamp, value = super().__getitem__(key)

            # if so, if not exprired, return
            if time() - timestamp <= self.ttl:
                return value
            
            # if it is expired, raise KeyError to act like it is missing (and delete it from cache)
            else:
                del self[key]
                raise KeyError(key)
            
        # if the non-expired key is missing for either reason
        except KeyError:

            # if there is no missing callback, raise KeyError
            if self.missing_callback is None:
                raise

            # otherwise, call the missing callback and cache the result
            else:

                value = self.missing_callback(key)
                self[key] = value
                return value
    

    def get(self, key: str) -> T | None:
        """dict.get() can be non-failing"""
        try:
            return self[key]
        except KeyError:
            return None
    

    def __setitem__(self, key: str, value: T) -> None:
        super().__setitem__(key, (time(), value))
    

    def clear_expired(self) -> None:
        current_time = time()
        expired_keys = [
            key for key, (timestamp, _) in self.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            del self[key]


def ttl_cached(ttl_seconds: int = 300):
    """
    TTL Cache decorator that uses the function it decorates as a missing callback.
    """
    cache = TTLCache(ttl_seconds=ttl_seconds)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create a cache key from function args
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            result = cache.get(key)
            
            if result is None:
                result = func(*args, **kwargs)
                cache[key] = result
            
            return result
        return wrapper
    return decorator
