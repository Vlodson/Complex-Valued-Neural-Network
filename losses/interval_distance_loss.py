from typing import Tuple
import numpy.typing as npt
import numpy as np

import wrapped_numpy as wnp
from losses.loss import Loss
from utils.label_utils import cat_to_arg_intervals
from custom_types import CategoricalLabels, ComplexMatrix


class IntervalDistance(Loss):
    def __labels_to_intervals(self, labels: npt.NDArray) -> npt.NDArray[np.float32]:
        return cat_to_arg_intervals(labels)

    def __angular_distance(
        self, theta1: np.float32, theta2: np.float32
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        # derivative is {1 if |t1 - t2| was used, -1 if 2pi - |t1 - t2| was used}
        # this will return false if it used |t1 - t2| and true if it used 2pi - |t1 - t2|

        difference = wnp.abs_(wnp.sub(theta1, theta2))
        difference_2pi = wnp.sub(2 * np.pi, difference)

        return np.min(difference, difference_2pi), difference < difference_2pi

    def __ravel_predictions(
        self, predictions: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        if len(predictions.shape) == 2:
            predictions = predictions.ravel()

        return predictions

    def calculate_loss(
        self, labels: CategoricalLabels, predictions: ComplexMatrix
    ) -> np.float32:
        labels = self.__labels_to_intervals(labels)
        predictions = self.__ravel_predictions(predictions)

        angular_distance_vectorized = np.vectorize(self.__angular_distance)

        correct_preds = (labels[:, 0] <= predictions) & (predictions < labels[:, 1])

        lower_distances, _ = angular_distance_vectorized(predictions, labels[:, 0])
        higher_distances, _ = angular_distance_vectorized(predictions, labels[:, 1])

        wrong_preds = ~correct_preds & (lower_distances < higher_distances)

        return wnp.div(
            np.sum(
                wnp.where(
                    correct_preds,
                    0.0,
                    wnp.where(wrong_preds, lower_distances, higher_distances),
                )
            ),
            labels.shape[0],
        )

    def loss_gradient(
        self, labels: CategoricalLabels, predictions: ComplexMatrix
    ) -> ComplexMatrix:
        labels = self.__labels_to_intervals(labels)
        predictions = self.__ravel_predictions(predictions)
        angular_distance_vectorized = np.vectorize(self.__angular_distance)

        correct_preds = (labels[:, 0] <= predictions) & (predictions < labels[:, 1])

        lower_distances, lower_distances_bools = angular_distance_vectorized(
            predictions, labels[:, 0]
        )
        higher_distances, higher_distances_bools = angular_distance_vectorized(
            predictions, labels[:, 1]
        )

        mask = wnp.where(
            lower_distances < higher_distances,
            lower_distances_bools,
            higher_distances_bools,
        )

        wrong_preds = ~correct_preds & mask

        return wnp.div(
            wnp.where(correct_preds, 0, wnp.where(wrong_preds, 1, -1)).reshape(-1, 1),
            labels.shape[0],
        )
