import numpy as np

from activations.activation import Activation
from custom_types import ComplexMatrix


class LinearScaled(Activation):
    def activate(self, x: ComplexMatrix) -> ComplexMatrix:
        return x / np.absolute(x)

    def deactivate(self, x: ComplexMatrix) -> ComplexMatrix:
        return 1 / np.absolute(x)
