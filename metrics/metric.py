from abc import ABC, abstractmethod

from custom_types import CategoricalLabels


class Metric(ABC):
    def __init__(self) -> None:
        super().__init__()

        self.name: str = type(self).__name__

    @abstractmethod
    def calculate_metric(
        self, labels: CategoricalLabels, predictions: CategoricalLabels
    ) -> float:
        """
        Calculate metric given labels and predictions
        """
