from collections import Counter
from typing import Callable, Iterable, List
from timeit import timeit

from parfun.decorators import delayed, parfun
from parfun.entry_point import set_parallel_backend_context
from parfun.partition.api import per_argument
from parfun.partition.collection import list_by_chunk


def read_text_file() -> List[str]:
    """
    Plain text obtained from Project Guttenberg:
    https://www.gutenberg.org/ebooks/100
    """

    with open("the_complete_works_of_william_shakespeare.txt", "rt") as f:
        return f.readlines()


def count_words(text: List[str]) -> Counter[str]:
    """
    Counts the number occurence of every words in a text.

        >>> count_words(["To be, or not to be,", "that is the question"])
        Counter({'To': 2,
                 'Be': 2,
                 'Or': 1,
                 'Not': 1,
                 'That': 1,
                 'Is': 1,
                 'The': 1,
                 'Question': 1})
    """

    counter: Counter[str] = Counter()

    for line in text:
        for word in line.split():
            word = word.strip(" ,.:;?!'").capitalize()

            if not word.isalpha():
                continue

            counter[word] += 1

    return counter


def sum_word_counts(sub_word_counts: Iterable[Counter[str]]) -> Counter[str]:
    word_count: Counter[str] = Counter()
    for sub_word_count in sub_word_counts:
        word_count.update(**sub_word_count)

    return word_count


@parfun(
    split=per_argument(text=list_by_chunk),
    combine_with=sum_word_counts,
)
def parfun_count_words(text: List[str]) -> Counter[str]:
    return count_words(text)


@delayed
def _delayed_count_words_internal(text: List[str]) -> Counter[str]:
    return count_words(text)


def delayed_count_words(text: List[str]) -> Counter[str]:
    CHUNK_SIZE = 30_000

    sub_word_counts = [_delayed_count_words_internal(text[i:i + CHUNK_SIZE]) for i in range(0, len(text), CHUNK_SIZE)]

    return sum_word_counts(sub_word_counts)  # type: ignore[arg-type]


def benchmark_function(function: Callable) -> float:
    N = 25
    function()  # warmup
    return timeit(function, number=N) / N


if __name__ == '__main__':
    text = read_text_file()

    print(f"Sequential:\t{benchmark_function(lambda: count_words(text).most_common(10)):.3f} secs")

    with set_parallel_backend_context("local_multiprocessing"):
        print(f"@parfun:\t{benchmark_function(lambda: parfun_count_words(text).most_common(10)):.3f} secs")

        print(f"@delayed:\t{benchmark_function(lambda: delayed_count_words(text).most_common(10)):.3f} secs")
