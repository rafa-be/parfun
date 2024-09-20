import datetime
import math
import time
import unittest
from functools import partial

from parfun.kernel.delayed_value import DelayedValue
from parfun.entry_point import BACKEND_REGISTRY, set_parallel_backend, set_parallel_backend_context
from parfun.profiler.functions import profile


class TestDelayedValue(unittest.TestCase):
    def setUp(self) -> None:
        set_parallel_backend("local_multiprocessing", max_workers=4)

    def test_value(self):
        delayed_value = DelayedValue(math.sqrt, (16,), {})

        self.assertEqual(delayed_value, math.sqrt(16))

    def test_attribute(self):
        delayed_value = DelayedValue(datetime.date, (2024, 9, 18), {})

        self.assertEqual(delayed_value.year, 2024)

    def test_methods(self):
        delayed_value = DelayedValue(datetime.date, (2024, 9, 18), {})

        self.assertEqual(delayed_value.isoformat(), "2024-09-18")

    def test_str(self):
        delayed_value = DelayedValue(datetime.datetime, (2024, 9, 18, 17, 24), {})

        self.assertEqual(str(delayed_value), str(datetime.datetime(2024, 9, 18, 17, 24)))

    def test_container(self):
        delayed_value: DelayedValue[list] = DelayedValue(list.__add__, ([1, 2, 3], [4, 5, 6]), {})

        self.assertSequenceEqual(delayed_value, [1, 2, 3, 4, 5, 6])
        self.assertEqual(list(delayed_value), [1, 2, 3, 4, 5, 6])
        self.assertEqual(sum(delayed_value), 21)
        self.assertEqual(delayed_value[2], 3)
        self.assertSequenceEqual(delayed_value[3:5], [4, 5])
        self.assertTrue(5 in delayed_value)

        delayed_value[0] = 100
        self.assertEqual(delayed_value[0], 100)

    def test_callable(self):
        delayed_value = DelayedValue(partial, (math.pow, 10), {})
        self.assertEqual(delayed_value(3), 1000)

    def test_numeric(self):
        delayed_value_1 = DelayedValue(math.sqrt, (16,), {})
        delayed_value_2 = DelayedValue(math.sqrt, (9,), {})

        self.assertEqual(delayed_value_1 + delayed_value_2, 7)
        self.assertEqual(delayed_value_1 * 2, 8)
        self.assertEqual(round(delayed_value_1), 4)
        self.assertEqual(int(delayed_value_1), 4)
        self.assertEqual(math.sqrt(delayed_value_1), 2)

    def test_comparators(self):
        delayed_value_1 = DelayedValue(math.sqrt, (16,), {})
        delayed_value_2 = DelayedValue(math.pow, (2, 2), {})

        self.assertTrue(delayed_value_1 == delayed_value_2)

        self.assertTrue(delayed_value_1 == 4)
        self.assertTrue(delayed_value_1 != 1)
        self.assertTrue(delayed_value_1 == delayed_value_2)

    def test_as_argument(self):
        delayed_value_1 = DelayedValue(math.sqrt, (16,), {})
        delayed_value_2 = DelayedValue(math.sqrt, (delayed_value_1,), {})

        self.assertEqual(delayed_value_2, 2)

    @unittest.skipUnless("scaler_local" in BACKEND_REGISTRY, "Scaler backend not installed")
    def test_nested(self):
        with set_parallel_backend_context("scaler_local", n_workers=2, per_worker_queue_size=10):
            self.assertEqual(_nested_sqrt(16), 2)

    def test_exception(self):
        delayed_value = DelayedValue(math.sqrt, (-16,), {})
        with self.assertRaises(ValueError):
            print(delayed_value)

    def test_speedup(self):
        DELAY = 500_000_000  # 500 ms
        N = 5

        with profile() as duration:
            delayed_values = [DelayedValue(_cpu_sleep, (DELAY,), {}) for _ in range(0, N)]
            sum(delayed_values)

        self.assertLess(duration.value, DELAY * N)


def _cpu_sleep(delay: int) -> int:
    begins_at = time.process_time_ns()

    while time.process_time_ns() - begins_at < delay:
        ...

    return time.process_time_ns() - begins_at


def _nested_sqrt(value):
    delayed_child_value = DelayedValue(math.sqrt, (value,), {})
    return math.sqrt(delayed_child_value)


if __name__ == "__main__":
    unittest.main()
