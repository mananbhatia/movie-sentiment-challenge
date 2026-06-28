import time
import statistics

import numpy as np

from src.config import MODEL_PATH
from src.data_loader import get_data
from src.model import SentimentModel

_SAMPLE_SIZE = 300
_WARMUP_SIZE = 10


def main() -> None:
    """Benchmark single-review prediction latency and print p50, p95, and mean."""
    model = SentimentModel.load(MODEL_PATH)

    X, _ = get_data()
    reviews = X.tolist()[:_SAMPLE_SIZE]

    for review in reviews[:_WARMUP_SIZE]:
        model.predict(review)

    durations_ms = []
    for review in reviews:
        start = time.perf_counter()
        model.predict(review)
        durations_ms.append((time.perf_counter() - start) * 1000)

    n = len(durations_ms)
    p50 = statistics.median(durations_ms)
    p95 = float(np.percentile(durations_ms, 95))
    mean = statistics.mean(durations_ms)

    print(f"Reviews timed : {n}")
    print(f"Mean latency  : {mean:.3f} ms")
    print(f"P50 latency   : {p50:.3f} ms")
    print(f"P95 latency   : {p95:.3f} ms")


if __name__ == "__main__":
    main()
