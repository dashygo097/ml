import cProfile
import functools
import io
import pstats
import time
from typing import Callable

from termcolor import colored


def timer(func: Callable):
    "decorator function that counts the time of a function"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tik = time.time()
        result = func(*args, **kwargs)
        tok = time.time()
        print(
            "Time elapsed of "
            + colored(f"{func.__name__}", color="yellow", attrs=["bold"])
            + ": "
            + colored(f"{tok - tik:.6f} s \n", color="red", attrs=["bold"])
        )

        return result

    return wrapper


def profile(func: Callable):
    "decorator function that profiles a function"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats()
        print(s.getvalue())
        return result

    return wrapper
