"""
 General utilities
"""

import os
import random
import numpy as np

from contextlib import ContextDecorator
from dataclasses import dataclass, field
import time
from typing import Any, Callable, ClassVar, Dict, Optional

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def seeds_generator(number, path):

    seeds = list()

    for _ in range(number + 1):
        seeds.append(random.randint(1, 2**32 - 1))

    np.save(path, seeds)

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

@dataclass
class Timer(ContextDecorator):
    """Time your code using a class, context manager, or decorator"""

    timers: ClassVar[Dict[str, float]] = dict()
    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = None
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialization: add timer to dict of timers"""
        if self.name:
            self.timers.setdefault(self.name, list())

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name].append(elapsed_time)

        return elapsed_time
    
    def get_average(self, timer) -> float:
        if timer in self.timers:
            return sum(self.timers[timer])/len(self.timers[timer])
        else:
            raise TimerError(f"Timer is not registered. Instantiate it first")
    
    def get_average_ms(self, timer) -> float:
        if timer in self.timers:
            return (sum(self.timers[timer])/len(self.timers[timer])) * 10e3
        else:
            raise TimerError(f"Timer is not registered. Instantiate it first")

    def get_time(self, timer) -> float:
        if timer in self.timers:
            return self.timers[timer]
        else:
            raise TimerError(f"Timer is not registered. Instantiate it first")

    def get_time_ms(self, timer) -> float:
        if timer in self.timers:
            return (self.timers[timer]) * 10e3
        else:
            raise TimerError(f"Timer is not registered. Instantiate it first")

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()
