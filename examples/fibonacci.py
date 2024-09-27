import math
import timeit

from parfun.decorators import delayed
from parfun.entry_point import set_parallel_backend_context

@delayed
def delayed_pow(a: float, b: float) -> float:
    return math.pow(a, b)

a = delayed_pow(2, 2)
b = delayed_pow(2, 2)

print(a + b)  # a and b will be computed in parallel

# This will compute all the `delayed_pow()` calls in parallel:
total_sum = sum([delayed_pow(x, 2) for x in range(0, 1000)])

# We can also use it to compute recursive functions. All items of the tree will be computed in parallel.

@delayed
def fibonacci(n: int) -> int:
    if n >= 2:
        return fibonacci(n - 1) + fibonacci(n - 2)
    else:
        return n


if __name__ == '__main__':
    with set_parallel_backend_context("local_multiprocessing"):


        print(f"@delayed fibonacci:\t{benchmark_function(lambda: parfun_count_words(text).most_common(10)):.3f} secs")

        print(f"@delayed:\t{benchmark_function(lambda: delayed_count_words(text).most_common(10)):.3f} secs")
