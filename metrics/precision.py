from typing import Dict
import numpy as np

import wrapped_numpy as wnp
from metrics.metric import Metric
from metrics.utils import get_inter_results

from custom_types import CategoricalLabels, MetricInterResults


class Precision(Metric):
    def __init__(self, macro: bool = True) -> None:
        super().__init__()
        self.macro = macro

    def __macro_precision(self, inter_results: Dict[int, MetricInterResults]) -> float:
        return wnp.div(
            sum(
                wnp.div(result["correct"], result["all_clf"])
                if result["all_clf"] != 0
                else 0
                for _, result in inter_results.items()
            ),
            len(inter_results),
        )

    def __micro_precision(self, inter_results: Dict[int, MetricInterResults]) -> float:
        return wnp.div(
            sum(result["correct"] for _, result in inter_results.items()),
            sum(result["all_clf"] for _, result in inter_results.items()),
        )

    def calculate_metric(
        self, labels: CategoricalLabels, predictions: CategoricalLabels
    ) -> float:
        inter_results = {
            cat: get_inter_results(predictions, labels, cat)
            for cat in wnp.unique(labels)
        }

        return (
            self.__macro_precision(inter_results)
            if self.macro
            else self.__micro_precision(inter_results)
        )
