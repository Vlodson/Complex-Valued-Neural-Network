import numpy as np
import numpy.typing as npt

import wrapped_numpy as wnp
from activations.activation import Activation
from custom_types import ComplexMatrix


class Argument(Activation):
    def activate(self, x: ComplexMatrix) -> npt.NDArray:
        arg = wnp.angle(x)
        return wnp.where(arg >= 0, arg, wnp.add(arg, 2 * np.pi))

    def deactivate(self, x: ComplexMatrix) -> ComplexMatrix:
        re = wnp.real(x)
        im = wnp.imag(x)

        denom = wnp.add(wnp.pwr(re, 2), wnp.pwr(im, 2))

        # arg(z) can be seen as arctan(im / re)
        return wnp.div(-im, wnp.add(denom, wnp.div(re, denom) * 1.0j))
