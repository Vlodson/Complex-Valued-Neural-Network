import wrapped_numpy as wnp

from activations.activation import Activation
from custom_types import ComplexMatrix


class LinearScaled(Activation):
    def activate(self, x: ComplexMatrix) -> ComplexMatrix:
        return wnp.div(x, wnp.absolute(x))

    def deactivate(self, x: ComplexMatrix) -> ComplexMatrix:
        return wnp.div(1, wnp.absolute(x))
