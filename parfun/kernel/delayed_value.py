from typing import Any, Callable, Dict, Generic, Tuple

import attrs

from parfun.backend.mixins import BackendSession, ProfiledFuture
from parfun.entry_point import get_parallel_backend
from parfun.object import Args, ReturnType


@attrs.define(init=False)
class DelayedValue(Generic[ReturnType]):
    _backend_session: BackendSession = attrs.field()
    _future: ProfiledFuture[ReturnType] = attrs.field()

    def __init__(self, function: Callable[Args, ReturnType], args, kwargs):
        args, kwargs = DelayedValue._undelay_arguments(args, kwargs)

        current_backend = get_parallel_backend()
        # allows_nested_tasks = current_backend is not None and current_backend.allows_nested_tasks()

        # Note: is_nested_parallelism check should appears before any backend check, as unsupported nested function
        # calls will have an empty backend setup.
        # if is_nested_parallelism() and not allows_nested_tasks:
        #     logging.debug(
        #         f"backend does not support nested parallelism. Running {self.function.__name__} sequentially."
        #     )
        #     return self.function(*args, **kwargs)

        if current_backend is None:
            self._backend_session = None
            self._future = ProfiledFuture()
            self._future.set_result(function(*args, **kwargs))
            return

        self._backend_session = current_backend.session().__enter__()

        self._future = self._backend_session.submit(function, *args, **kwargs)

        def on_done_callback(completed_future: ProfiledFuture[ReturnType]):
            exception = completed_future.exception()
            if exception is not None:
                self._backend_session.__exit__(type(exception), exception, exception.__traceback__)
            else:
                self._backend_session.__exit__(None, None, None)

        self._future.add_done_callback(on_done_callback)

    def __getattr__(self, name: str):
        assert name not in DelayedValue.__dict__
        return getattr(self.delayed_value(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in DelayedValue.__dict__:
            return super().__setattr__(name, value)
        else:
            return setattr(self.delayed_value(), name, value)

    def delayed_value(self) -> ReturnType:
        return self._future.result()

    def __repr__(self):
        if self._future.done():
            value = self.delayed_value().__repr__()
        else:
            value = "<pending>"

        return f"DelayedValue({value})"

    @classmethod
    def _add_overloaded_method(cls, method_name: str):
        def wrapper(self, *args, **kwargs):
            args, kwargs = cls._undelay_arguments(args, kwargs)
            return getattr(self.delayed_value(), method_name)(*args, **kwargs)

        wrapper.__name__ = method_name

        return setattr(cls, method_name, wrapper)

    @classmethod
    def _undelay_arguments(
        cls, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Waits for the completion of any delayed value in positional or keyword arguments."""

        args = tuple(arg.delayed_value() if isinstance(arg, cls) else arg for arg in args)
        kwargs = {
            arg_name: arg_value.delayed_value() if isinstance(arg_value, cls) else arg_value
            for arg_name, arg_value in kwargs.items()
        }

        return args, kwargs


__OVERLOADED_METHODS = {
    # Base methods
    "__sizeof__", "__str__", "__bytes__", "__hash__", "__dir__", "__format__",

    # Container methods (sequence, mapping protocols)
    "__len__", "__getitem__", "__setitem__", "__delitem__", "__contains__", "__iter__",
    "__reversed__", "__missing__",

    # Callable objects
    "__call__",

    # Numeric and arithmetic operators
    "__abs__", "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__mod__", "__pow__", "__matmul__",
    "__neg__", "__pos__", "__divmod__", "__round__", "__trunc__",

    # In-place arithmetic operators (augmented assignment)
    "__iadd__", "__isub__", "__imul__", "__itruediv__", "__ifloordiv__", "__imod__", "__ipow__", "__imatmul__",
    "__idivmod__",

    # Reflected arithmetic operators (right-hand side)
    "__radd__", "__rsub__", "__rmul__", "__rtruediv__", "__rfloordiv__", "__rmod__", "__rpow__", "__rmatmul__",
    "__rdivmod__",

    # Bitwise operators
    "__and__", "__or__", "__xor__", "__lshift__", "__rshift__", "__invert__", "__not__",

    # In-place bitwise operators (augmented assignment)
    "__iand__", "__ior__", "__ixor__", "__ilshift__", "__irshift__",

    # Reflected bitwise operators
    "__rand__", "__ror__", "__rxor__", "__rlshift__", "__rrshift__",

    # Comparison operators
    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",

    # Reflected comparison operators
    "__req__", "__rne__", "__rlt__", "__rle__", "__rgt__", "__rge__",

    # Type conversion methods
    "__index__", "__bool__", "__int__", "__float__", "__complex__", "__round__", "__trunc__",
}


for method_name in __OVERLOADED_METHODS:
    DelayedValue._add_overloaded_method(method_name)
