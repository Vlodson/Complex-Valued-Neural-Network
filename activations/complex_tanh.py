import wrapped_numpy as wnp

from activations.activation import Activation
from custom_types import ComplexMatrix


class ComplexTanH(Activation):
    def activate(self, x: ComplexMatrix) -> ComplexMatrix:
        return wnp.tanh(x)

    def deactivate(self, x: ComplexMatrix) -> ComplexMatrix:
        return wnp.sub(1, wnp.pwr(self.activate(x), 2))
