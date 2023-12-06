import numpy as np

from activations.activation import Activation
from custom_types import ComplexMatrix


class ComplexTanH(Activation):
    def activate(self, x: ComplexMatrix) -> ComplexMatrix:
        return np.tanh(x)

    def deactivate(self, x: ComplexMatrix) -> ComplexMatrix:
        return 1 - self.activate(x) ** 2
