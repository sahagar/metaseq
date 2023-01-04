from __future__ import annotations

import functools
import math
import re
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any
from typing import Counter as TCounter
from typing import List, Optional, Tuple, Union

import torch


TScalar = Union[int, float, torch.Tensor]
TVector = Union[List[TScalar], torch.Tensor]


re_art = re.compile(r"\b(a|an|the)\b")
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


@functools.total_ordering
class Metric(ABC):
    """
    Base class for storing metrics.

    Subclasses should define .value(). Examples are provided for each subclass.
    """

    @property
    def is_global(self) -> bool:
        """
        Indicates whether this metric should be reported globally or per-task.
        """
        return False

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return False

    @abstractmethod
    def value(self) -> float:
        """
        Return the value of the metric as a float.
        """

    @abstractmethod
    def __add__(self, other: Any) -> Metric:
        raise NotImplementedError

    def __iadd__(self, other):
        return self.__radd__(other)

    def __radd__(self, other: Any):
        if other is None:
            return self
        return self.__add__(other)

    def __str__(self) -> str:
        return f"{self.value():.4g}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value():.4g})"

    def __float__(self) -> float:
        return float(self.value())

    def __int__(self) -> int:
        return int(self.value())

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Metric):
            return self.value() == other.value()
        else:
            return self.value() == other

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Metric):
            return self.value() < other.value()
        else:
            return self.value() < other

    def __sub__(self, other: Any) -> float:
        """
        Used heavily for assertAlmostEqual.
        """
        if not isinstance(other, float):
            raise TypeError("Metrics.__sub__ is intentionally limited to floats.")
        return self.value() - other

    def __rsub__(self, other: Any) -> float:
        """
        Used heavily for assertAlmostEqual.

        NOTE: This is not necessary in python 3.7+.
        """
        if not isinstance(other, float):
            raise TypeError("Metrics.__rsub__ is intentionally limited to floats.")
        return other - self.value()

    @classmethod
    def as_number(cls, obj: TScalar) -> Union[int, float]:
        if isinstance(obj, torch.Tensor):
            obj_as_number: Union[int, float] = obj.item()
        else:
            obj_as_number = obj  # type: ignore
        assert isinstance(obj_as_number, int) or isinstance(obj_as_number, float), f"obj_as_number:{obj_as_number}"
        return obj_as_number

    @classmethod
    def as_float(cls, obj: TScalar) -> float:
        return float(cls.as_number(obj))

    @classmethod
    def as_int(cls, obj: TScalar) -> int:
        return int(cls.as_number(obj))

    @classmethod
    def many(cls, *objs: List[TVector]):  # -> List[Metric]:
        """
        many() 传入的是两个参数是 size相同的二维数据. 返回的是一个 List[Metric]
        PPLMetric.many( [1,2], [1,2] ) -> List[ PPLMetric(2.718), PPLMetric(2.718) ]

        Construct many of a Metric from the base parts.

        Useful if you separately compute numerators and denomenators, etc.
        """
        lengths = [len(o) for o in objs]
        objs = list(objs)  # convert from tuple for inplace modification
        for i, o in enumerate(objs):
            if isinstance(o, torch.Tensor):
                # if the tensor is on GPU, make sure we transfer the whole thing
                # at once, instead of one-element-at-a-time during our list
                # comprehension
                objs[i] = o.tolist()
        if len(set(lengths)) != 1:
            raise IndexError(f"Uneven {cls.__name__} constructions: {lengths}")
        return [cls(*items) for items in zip(*objs)]


class SumMetric(Metric):
    """
    Class that keeps a running sum of some metric.

    Examples of SumMetric include things like "exs", the number of examples seen since
    the last report, which depends exactly on a teacher.
    """

    __slots__ = ("_sum",)

    def __init__(self, sum_: TScalar = 0):
        if isinstance(sum_, torch.Tensor):
            self._sum = sum_.item()
        else:
            assert isinstance(sum_, (int, float))
            self._sum = sum_

    def __add__(self, other: Optional[SumMetric]) -> SumMetric:
        # NOTE: hinting can be cleaned up with "from __future__ import annotations" when
        # we drop Python 3.6
        if other is None:
            return self
        full_sum = self._sum + other._sum
        # always keep the same return type
        return type(self)(sum_=full_sum)

    def value(self) -> float:
        return self._sum


class AverageMetric(Metric):
    """
    Class that keeps a running average of some metric.

    Examples of AverageMetrics include hits@1, F1, accuracy, etc. These metrics all have
    per-example values that can be directly mapped back to a teacher.
    """

    __slots__ = ("_numer", "_denom")

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return True

    def __init__(self, numer: TScalar, denom: TScalar = 1):
        self._numer = self.as_number(numer)
        self._denom = self.as_number(denom)

    def __add__(self, other: Optional[AverageMetric]) -> AverageMetric:
        # NOTE: hinting can be cleaned up with "from __future__ import annotations" when
        # we drop Python 3.6
        if other is None:
            return self
        full_numer: TScalar = self._numer + other._numer
        full_denom: TScalar = self._denom + other._denom
        # always keep the same return type
        return type(self)(numer=full_numer, denom=full_denom)

    def value(self) -> float:
        if self._numer == 0 and self._denom == 0:
            # don't nan out if we haven't counted anything
            return 0.0
        if self._denom == 0:
            return float("nan")
        return self._numer / self._denom


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(" ", s)
    s = re_art.sub(" ", s)
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '
    s = " ".join(s.split())
    return s


class F1Metric(AverageMetric):
    """
    Helper class which computes token-level F1.
    """

    @staticmethod
    def _prec_recall_f1_score(pred_items, gold_items):
        """
        Compute precision, recall and f1 given a set of gold and prediction items.

        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values

        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def compute(guess: str, answers: List[str]) -> F1Metric:
        if guess is None or answers is None:
            return AverageMetric(0, 0)
        g_tokens = normalize_answer(guess).split()
        scores = [F1Metric._prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers]
        return F1Metric(max(f1 for p, r, f1 in scores), 1)


class BleuMetric(AverageMetric):
    @staticmethod
    def compute(guess: str, answers: List[str], k: int = 4) -> Optional[BleuMetric]:
        """
        Compute approximate BLEU score between guess and a set of answers.
        """
        try:
            from nltk.translate import bleu_score as nltkbleu
        except ImportError:
            # User doesn't have nltk installed, so we can't use it for bleu
            # We'll just turn off things, but we might want to warn the user
            return None

        # Warning: BLEU calculation *should* include proper tokenization and
        # punctuation etc. We're using the normalize_answer for everything though,
        # so we're over-estimating our BLEU scores.  Also note that NLTK's bleu is
        # going to be slower than fairseq's (which is written in C), but fairseq's
        # requires that everything be in arrays of ints (i.e. as tensors). NLTK's
        # works with strings, which is better suited for this module.
        weights = [1 / k for _ in range(k)]
        score = nltkbleu.sentence_bleu(
            [normalize_answer(a).split(" ") for a in answers],
            normalize_answer(guess).split(" "),
            smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
            weights=weights,
        )
        return BleuMetric(score)


class RougeMetric(AverageMetric):
    _evaluator = None

    @staticmethod
    def compute_many(
        guess: str, answers: List[str]
    ) -> Tuple[Optional[RougeMetric], Optional[RougeMetric], Optional[RougeMetric]]:
        """
        Compute ROUGE score between guess and *any* answer.

        Done with compute_many due to increased efficiency.

        :return: (rouge-1, rouge-2, rouge-L)
        """
        # possible global initialization
        try:
            import rouge
        except ImportError:
            # User doesn't have py-rouge installed, so we can't use it.
            # We'll just turn off rouge computations
            return None, None, None

        if RougeMetric._evaluator is None:
            RougeMetric._evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2)
        try:
            scores = [RougeMetric._evaluator.get_scores(normalize_answer(guess), normalize_answer(a)) for a in answers]
        except LookupError:
            warnings.warn(
                "ROUGE requires nltk punkt tokenizer. Please run " "`python -c \"import nltk; nltk.download('punkt')`"
            )
            return None, None, None

        scores_rouge1 = max(score["rouge-1"]["r"] for score in scores)
        scores_rouge2 = max(score["rouge-2"]["r"] for score in scores)
        scores_rougeL = max(score["rouge-l"]["r"] for score in scores)
        return (
            RougeMetric(scores_rouge1),
            RougeMetric(scores_rouge2),
            RougeMetric(scores_rougeL),
        )


class InterDistinctMetric(Metric):
    """
    Compute inter-distinct metric over corpus-level.
    """

    def __init__(self, counts: TCounter[Tuple]):
        """
        :param counts:
            collections.Counter of ngram -> frequency
        """
        self._counts = counts

    def __add__(self, other):
        return InterDistinctMetric(self._counts + other._counts)

    def value(self):
        return max(len(self._counts), 1e-12) / max(sum(self._counts.values()), 1e-5)

    @classmethod
    def _ngram(cls, seq, n):
        for i in range(len(seq) - n + 1):
            yield tuple(seq[i : i + n])

    @classmethod
    def compute(cls, text, ngram=1):
        tokens = normalize_answer(text).split()
        return InterDistinctMetric(Counter(cls._ngram(tokens, ngram)))


class IntraDistinctMetric(AverageMetric):
    """
    Compute intra-distinct (per-utterance).
    """

    @classmethod
    def _ngram(cls, seq, n: int):
        for i in range(len(seq) - n + 1):
            yield tuple(seq[i : i + n])

    @classmethod
    def compute(cls, text: str, ngram: int = 1):
        """
        :param text:
            The text to compute metric over
        :param ngram:
            n-gram length
        """
        tokens = normalize_answer(text).split()
        counts: Counter[Any] = Counter(cls._ngram(tokens, ngram))
        # computed per-example, macro averaged across examples
        intra = max(len(counts), 1e-12) / max(sum(counts.values()), 1e-5)
        return IntraDistinctMetric(intra, 1.0)


class PPLMetric(AverageMetric):
    def value(self):
        return math.exp(super().value())


if __name__ == "__main__":
    print(F1Metric.compute("hello", ["hello I am good"]))
    print(RougeMetric.compute_many("hello", ["hello I am good"]))
    print(BleuMetric.compute("hello I am good", ["hello I am good"]))

    sents = ["hello", "hello i am good"]
    print(InterDistinctMetric.compute(sents[0]) + InterDistinctMetric.compute(sents[1]))
