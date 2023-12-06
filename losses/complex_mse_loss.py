import numpy as np
import numpy.typing as npt

from losses.loss import Loss
from utils.label_utils import cat_to_unit_vector
from custom_types import CategoricalLabels, ComplexMatrix


class ComplexMSE(Loss):
    def calculate_loss(
        self, labels: CategoricalLabels, predictions: ComplexMatrix
    ) -> float:
        labels = cat_to_unit_vector(labels)
        predictions = predictions.ravel()

        return np.sum((labels - predictions) ** 2) / (2 * labels.shape[0])

    def loss_gradient(
        self, labels: CategoricalLabels, predictions: ComplexMatrix
    ) -> ComplexMatrix:
        labels = cat_to_unit_vector(labels)
        predictions = predictions.ravel()

        return ((predictions - labels) / labels.shape[0]).reshape(-1, 1)
