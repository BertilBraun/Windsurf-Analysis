# define decorator that logs exceptions and reraises them

import logging

from functools import wraps


def log_and_reraise(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f'Error in {func.__name__}: {e}')
            raise

    return wrapper
