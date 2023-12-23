import numpy as np
from custom_types import CategoricalLabels

from metrics.metric import Metric
from metrics.precision import Precision
from metrics.recall import Recall


class F1(Metric):
    def __init__(self, macro: bool = True) -> None:
        super().__init__()
        self.macro = macro

    def calculate_metric(
        self, labels: CategoricalLabels, predictions: CategoricalLabels
    ) -> float:
        return (
            2
            * (prec := Precision(self.macro).calculate_metric(labels, predictions))
            * (rec := Recall(self.macro).calculate_metric(labels, predictions))
        ) / (prec + rec)
