"""This module implements a simple stopwatch class and context manager to profile python code.

You can use the stopwatch functionality by instantiating and using a Stopwatch object directly:
    sw = Stopwatch(name="my stopwatch name")
    sw.start()
    # do the thing that needs to be profiled
    elapsed_time = sw.stop()  # this will return the elapsed time for this first code block
    # do something else (time spent here will not affect the final elapsed time)
    sw.start()
    # do more stuff to be profiled
    elapsed_time = sw.stop()  # this will return the elapsed time for this second code block
    tot_elapsed_time = sw.total()  # this will return the TOTAL elapsed time for both blocks
    sw.log()  # this will log the total elapsed time with the default message format

You can also use the stopwatch via a context manager:
    with Stopwatch(name="my stopwatch name") as sw:
        # do the thing that needs to be profiled
    # when leaving the scope, the `log` function will be automatically called
    # you don't need the `... as sw:` part, but if you use it, you can access the time:
    elapsed_time = sw.total()
"""
import logging
import time
import typing


class Stopwatch:
    """Code profiling stopwatch class."""

    default_log_message_format: str = "{name} elapsed time: {seconds:0.4f} seconds"
    """Default message printing format used when logging; will be filled via `str.format(...)."""

    def __init__(
        self,
        name: typing.AnyStr = "stopwatch",
        log_message_format: typing.AnyStr = default_log_message_format,
        log_level: int = logging.DEBUG,
        logger: typing.Optional[logging.Logger] = None,
    ):
        """Initializes the stopwatch with the provided arguments."""
        self.name = str(name)
        self.log_message_format = log_message_format
        self.log_level = log_level
        if logger is None:
            import ssl4rs.utils.logging  # here to avoid circular imports

            default_logger = ssl4rs.utils.logging.get_logger(__name__)
            logger = default_logger
        self.logger = logger
        self._total_elapsed_time: float = 0.0  # in seconds
        self._latest_start_time: typing.Optional[float] = None

    def start(self) -> None:
        """Starts the stopwatch by logging the current time."""
        assert self._latest_start_time is None, "stopwatch is already running!"
        self._latest_start_time = time.perf_counter()

    def stop(self) -> float:
        """Stops the stopwatch, returning the time (in seconds) since `start()` was last called."""
        assert self._latest_start_time is not None, "stopwatch is not running!"
        curr_time = time.perf_counter()
        elapsed_time = curr_time - self._latest_start_time
        self._total_elapsed_time += elapsed_time
        self._latest_start_time = None
        return elapsed_time

    def total(self) -> float:
        """Returns the total time logged using this stopwatch (in seconds), across all segments."""
        return self._total_elapsed_time

    def log(self, *args, **kwargs) -> None:
        """Reports the total time logged using this stopwatch (in seconds), across all segments."""
        total_elapsed_time = self.total()
        log_message = self.log_message_format.format(
            *args,
            name=self.name,
            milliseconds=total_elapsed_time * 1000,
            seconds=total_elapsed_time,
            minutes=total_elapsed_time / 60,
            hours=total_elapsed_time / (60 * 60),
            **kwargs,
        )
        self.logger.log(self.log_level, log_message)

    def __enter__(self) -> "Stopwatch":
        """Starts the stopwatch time monitoring process, returning the current object itself."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stops the stopwatch time monitoring process, logging the total elapsed time."""
        try:
            self.stop()
        except AssertionError:
            pass
        self.log()
