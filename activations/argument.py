import numpy as np
import numpy.typing as npt

from activations.activation import Activation
from custom_types import ComplexMatrix


class Argument(Activation):
    def activate(self, x: ComplexMatrix) -> npt.NDArray:
        arg = np.angle(x)
        return np.where(arg >= 0, arg, arg + 2 * np.pi)

    def deactivate(self, x: ComplexMatrix) -> ComplexMatrix:
        re = np.real(x)
        im = np.imag(x)

        denom = re**2 + im**2

        # arg(z) can be seen as arctan(im / re)
        return -im / denom + (re / denom) * 1.0j
