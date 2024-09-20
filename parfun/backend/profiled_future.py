from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, Generic, Optional, Tuple, TypeVar

from parfun.profiler.object import TraceTime

T = TypeVar("T")


class ProfiledFuture(Future, Generic[T]):
    """Future that provides an additional duration metric used to profile the task's duration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._duration = None

    def result(self, timeout: float | None = None) -> T:
        return super().result(timeout)  # type: ignore[arg-type]

    def add_done_callback(self, function: Callable[["ProfiledFuture[T]"], Any]) -> None:
        return super().add_done_callback(function)  # type: ignore[arg-type]

    def set_result(self, result: T, duration: Optional[TraceTime] = None):
        # Sets the task duration before the result, as set_result() triggers all completion callbacks.
        self._duration = duration
        return super().set_result(result)

    def set_exception(self, exception: Optional[BaseException], duration: Optional[TraceTime] = None) -> None:
        # Sets the task duration before the exception, as set_exception() triggers all completion callbacks.
        self._duration = duration
        return super().set_exception(exception)

    def duration(self, timeout: Optional[float] = None) -> Optional[TraceTime]:
        """
        The total CPU time (i.e. user + system times) required to run the task, or `None` if the task didn't provide
        task profiling.

        This **should** include the overhead time required to schedule the task.
        """

        self.exception(timeout)  # Waits until the task finishes.

        return self._duration

    def result_and_duration(self, timeout: Optional[float] = None) -> Tuple[T, Optional[TraceTime]]:
        """
        Combines the calls to `result() and duration()`:

        .. code:: python

            result, duration = future.result(), future.duration()
            # is equivalent to
            result, duration = future.result_and_duration()

        """

        result = self.result(timeout)
        return result, self._duration
