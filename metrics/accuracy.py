import numpy as np
from metrics.metric import Metric

from custom_types import CategoricalLabels


class Accuracy(Metric):
    def calculate_metric(
        self, labels: CategoricalLabels, predictions: CategoricalLabels
    ) -> float:
        return np.sum(labels == predictions) / labels.shape[0]
