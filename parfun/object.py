from typing import ParamSpec, TypeVar

Args = ParamSpec("Args")
ReturnType = TypeVar("ReturnType")

PartitionType = TypeVar("PartitionType")  # Input and output are identical for partitioning functions.
