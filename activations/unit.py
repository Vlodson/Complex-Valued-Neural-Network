import wrapped_numpy as wnp

from activations.activation import Activation
from custom_types import ComplexMatrix


class Unit(Activation):
    # Should act like and replace relu in the complex world
    def activate(self, x: ComplexMatrix) -> ComplexMatrix:
        return wnp.where(wnp.absolute(x) > 1, 0, x)

    def deactivate(self, x: ComplexMatrix) -> ComplexMatrix:
        return wnp.where(wnp.absolute(x) > 1, 0, 1)
