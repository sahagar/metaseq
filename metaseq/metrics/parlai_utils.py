from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .parlai_metrics import (
    BleuMetric,
    F1Metric,
    InterDistinctMetric,
    IntraDistinctMetric,
    Metric,
    PPLMetric,
    RougeMetric,
    SumMetric,
)


DIALOG_METRICS = {
    "f1",
    "rouge-1",
    "rouge-2",
    "rouge-L",
    "bleu-1",
    "bleu-2",
    "interdistinct-1",
    "interdistinct-2",
}

DEFAULT_METRICS = {"bleu-4", "accuracy", "f1"}
ROUGE_METRICS = {"rouge-1", "rouge-2", "rouge-L"}
BLEU_METRICS = {"bleu-1", "bleu-2", "bleu-3", "bleu-4"}
DISTINCT_METRICS = {
    "interdistinct-1",
    "interdistinct-2",
    "intradistinct-1",
    "intradistinct-2",
}
ALL_METRICS = DEFAULT_METRICS | ROUGE_METRICS | BLEU_METRICS | DISTINCT_METRICS


# core/metrics.py
class Metrics(object):
    """
    Metrics aggregator.
    """

    def __init__(self, threadsafe=False, shared=None):
        if shared and "data" in shared:
            # This is a clone
            self._data = shared["data"]
        else:
            # The original
            self._data = {}

        # recent data is to track per-example metrics, and so should never be
        # shared
        self._recent_data = {}

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return f"Metrics({repr(self._data)})"

    def add(self, key: str, value: Optional[Metric]) -> None:
        """
        Record an accumulation to a metric.
        """
        self._data[key] = self._data.get(key) + value
        # There is bug in parlai code, which is 'self._recent_data[key] = self._recent_data.get(key) + value'
        self._recent_data[key] = value

    def report(self):
        """
        Report the metrics over all data seen so far.
        """
        return self._data.copy()

    def report_values(self, ndigits=4):
        """
        Report values of metrics over all data seen so far.
        """
        values = {}
        for k, v in self._data.items():
            if hasattr(v, "value"):
                values[k] = round(v.value(), 4)
        return values

    def clear_recent(self):
        """
        Clear recent metrics (latest example).
        """
        self._recent_data.clear()

    def report_recent(self):
        """
        Report recent metrics (latest example).
        """
        return self._recent_data.copy()

    def report_recent_value(self):
        values = {}
        for k, v in self._recent_data.items():
            if hasattr(v, "value"):
                values[k] = f"{v.value():.4g}"
        return values

    def clear(self):
        """
        Clear all the metrics.
        """
        self._data.clear()
        self._recent_data.clear()

    def share(self):
        return {"data": self._data}

    def add_metrics(self, other: "Metrics") -> None:
        """
        Aggregate another Metrics objects metrics into this one.

        Note that it is assumed that the keys for metrics are disjoint between Metrics
        objects.
        """
        for k, v in other._data.items():
            self.add(k, v)


class DialogMetrics(Metrics):
    """
    Helper container which encapsulates standard metrics (F1, BLEU, ...).
    """

    def __init__(self, metrics_list: str = "default", shared: Dict[str, Any] = None) -> None:
        super().__init__(shared=shared)
        self._metrics_list = self._infer_metrics(metrics_list)
        self.eval_pr = [1, 5, 10, 100]

    @staticmethod
    def _infer_metrics(cli_arg: str) -> Set[str]:
        """
        Parse the CLI metric into a list of metrics we wish to compute.
        """
        col: Set[str] = set()
        names = cli_arg.split(",")
        for n in names:
            if n == "default":
                col |= DIALOG_METRICS
            elif n == "rouge":
                col |= ROUGE_METRICS
            elif n == "bleu":
                col |= BLEU_METRICS
            elif n == "distinct":
                col |= DISTINCT_METRICS
            elif n == "all":
                col |= ALL_METRICS
            elif n == "dialog":
                col |= DIALOG_METRICS
            else:
                col.add(n)
        return col

    def evaluate_response(self, prediction: str, labels: List[str]) -> None:
        """
        Compute all required text-based metrics based on an observation and labels.
        """
        self.add("exs", SumMetric(1))

        if prediction is not None:
            self.add("f1", F1Metric.compute(prediction, labels))

            for k in range(1, 5):  # 1..4
                if f"bleu-{k}" in self._metrics_list:
                    self.add(f"bleu-{k}", BleuMetric.compute(prediction, labels, k))
            # compute distinct-k
            for k in [1, 2]:
                if f"interdistinct-{k}" in self._metrics_list:
                    self.add(f"interdistinct-{k}", InterDistinctMetric.compute(prediction, k))
                if f"intradistinct-{k}" in self._metrics_list:
                    self.add(f"intradistinct-{k}", IntraDistinctMetric.compute(prediction, k))
            # if any of the rouges are in the list
            if self._metrics_list & ROUGE_METRICS:
                r1, r2, rL = RougeMetric.compute_many(prediction, labels)
                if "rouge-1" in self._metrics_list and r1:
                    self.add("rouge_1", r1)
                if "rouge-2" in self._metrics_list and r2:
                    self.add("rouge_2", r2)
                if "rouge-L" in self._metrics_list and rL:
                    self.add("rouge_L", rL)


if __name__ == "__main__":
    metrics = Metrics()
    metrics.add("f1", F1Metric.compute("hello world", ["hello world"]))
    ppls = PPLMetric.many([1, 2, 1], [1, 2, 2])
    print(ppls)
    print(ppls[0] + None)
    print(metrics)
    print(sum(ppls, None))
