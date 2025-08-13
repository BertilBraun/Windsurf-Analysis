import hashlib
import os
import pickle
from functools import wraps
from typing import Any


def generate_hash_code(data) -> str:
    # Serialize data with pickle
    serialized_data = pickle.dumps(data)

    # Create a hash object with MD5
    hash_object = hashlib.md5()
    hash_object.update(serialized_data)  # pickle.dumps returns bytes

    # Return the hash as a hex string
    return hash_object.hexdigest()


def cache_to_file(folder_name: str, ignore_args: list[str | int] = [], additional_args: list[Any] = []):
    # This decorator should be usable like @cache to cache the result of a function. The cache mapping should be stored in a file with the given file_name. The cache should be loaded at the beginning of the function and saved at the end of the function. The cache should be a dictionary that maps the arguments to the result of the function.
    folder_name = os.path.join('cache', folder_name)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            os.makedirs(folder_name, exist_ok=True)

            # Sort kwargs to ensure the hash is deterministic
            args_to_hash = [arg for i, arg in enumerate(args) if i not in ignore_args]
            kwargs_to_hash = {k: v for k, v in kwargs.items() if k not in ignore_args}
            hash_code = generate_hash_code(
                (func.__name__, args_to_hash, sorted(kwargs_to_hash.items()), additional_args)
            )
            cache_file_name = os.path.join(folder_name, hash_code + '.pkl')

            if os.path.exists(cache_file_name):
                with open(cache_file_name, 'rb') as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)

            with open(cache_file_name, 'wb') as f:
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator
