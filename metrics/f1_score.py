from custom_types import CategoricalLabels
import wrapped_numpy as wnp
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
        return wnp.div(
            (
                2
                * wnp.mul(
                    (
                        prec := Precision(self.macro).calculate_metric(
                            labels, predictions
                        )
                    ),
                    (rec := Recall(self.macro).calculate_metric(labels, predictions)),
                )
            ),
            wnp.add(prec, rec),
        )
