import numpy as np
import numpy.typing as npt

import wrapped_numpy as wnp
from losses.loss import Loss
from utils.label_utils import cat_to_arg_centers
from custom_types import ComplexMatrix, CategoricalLabels


class ArgumentDistance(Loss):
    def __angular_distance(
        self, theta1: npt.NDArray[np.float32], theta2: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        distance = wnp.abs_(wnp.sub(theta1, theta2))
        distance_2pi = wnp.sub(2 * np.pi, distance)

        return wnp.where(distance < distance_2pi, distance, distance_2pi)

    def __angular_distance_derivative(
        self, theta1: npt.NDArray[np.float32], theta2: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        distance = wnp.abs_(wnp.sub(theta1, theta2))
        distance_2pi = wnp.sub(2 * np.pi, distance)

        return wnp.where(
            distance < distance_2pi,
            wnp.sign(wnp.sub(theta1, theta2)),
            -wnp.sign(wnp.sub(theta1, theta2)),
        )

    def calculate_loss(
        self, labels: CategoricalLabels, predictions: ComplexMatrix
    ) -> float:
        labels = cat_to_arg_centers(labels)
        predictions = (
            predictions.ravel() if len(predictions.shape) == 2 else predictions
        )

        return wnp.div(
            wnp.sum_(self.__angular_distance(predictions, labels)), labels.shape[0]
        )

    def loss_gradient(
        self, labels: CategoricalLabels, predictions: ComplexMatrix
    ) -> npt.NDArray[np.float32]:
        labels = cat_to_arg_centers(labels)
        predictions = (
            predictions.ravel() if len(predictions.shape) == 2 else predictions
        )

        return self.__angular_distance_derivative(predictions, labels).reshape(-1, 1)
