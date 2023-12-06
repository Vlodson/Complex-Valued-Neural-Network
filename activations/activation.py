from abc import ABC, abstractmethod
from custom_types import ComplexMatrix


class Activation(ABC):
    @abstractmethod
    def activate(self, x: ComplexMatrix) -> ComplexMatrix:
        """
        Activates the passed input
        """

    @abstractmethod
    def deactivate(self, x: ComplexMatrix) -> ComplexMatrix:
        """
        Deactivates the passed input
        """
