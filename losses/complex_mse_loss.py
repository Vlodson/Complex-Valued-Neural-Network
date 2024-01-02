import wrapped_numpy as wnp
from losses.loss import Loss
from utils.label_utils import cat_to_unit_vector
from custom_types import CategoricalLabels, ComplexMatrix


class ComplexMSE(Loss):
    def calculate_loss(
        self, labels: CategoricalLabels, predictions: ComplexMatrix
    ) -> float:
        labels = cat_to_unit_vector(labels)
        predictions = predictions.ravel()

        return wnp.div(
            wnp.axis_sum(wnp.pwr(wnp.sub(labels, predictions), 2)), 2 * labels.shape[0]
        )

    def loss_gradient(
        self, labels: CategoricalLabels, predictions: ComplexMatrix
    ) -> ComplexMatrix:
        labels = cat_to_unit_vector(labels)
        predictions = predictions.ravel()

        return wnp.div(wnp.sub(predictions, labels), labels.shape[0]).reshape(-1, 1)
