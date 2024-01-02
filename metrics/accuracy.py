import numpy as np
import wrapped_numpy as wnp
from metrics.metric import Metric

from custom_types import CategoricalLabels


class Accuracy(Metric):
    def calculate_metric(
        self, labels: CategoricalLabels, predictions: CategoricalLabels
    ) -> float:
        return wnp.div(np.sum(labels == predictions), labels.shape[0])
