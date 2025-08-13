import time
from contextlib import contextmanager


@contextmanager
def timeit(name: str):
    """
    Usage:
    with timeit('name'):
        ...
    """
    start = time.time()
    yield
    end = time.time()
    print(f'{name}: {end - start} seconds')
