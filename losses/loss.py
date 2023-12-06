from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

from custom_types import ComplexMatrix, CategoricalLabels


class Loss(ABC):
    @abstractmethod
    def calculate_loss(
        self, labels: CategoricalLabels, predictions: ComplexMatrix
    ) -> float:
        """
        Calculates the loss given labels and predictions
        """

    @abstractmethod
    def loss_gradient(
        self, labels: CategoricalLabels, predictions: ComplexMatrix
    ) -> npt.NDArray:
        """
        Calculates the gradient of the loss given labels and predictions
        """
