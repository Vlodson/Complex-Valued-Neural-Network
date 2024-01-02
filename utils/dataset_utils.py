from typing import Tuple

import numpy as np
import wrapped_numpy as wnp
from custom_types import ComplexMatrix, CategoricalLabels, MiniBatches


def minibatching(
    x: ComplexMatrix, y: CategoricalLabels, batch_size: int
) -> MiniBatches:
    """
    Batches the passed x and y values into smaller batches
    """
    return [
        (
            x[i * batch_size : (i + 1) * batch_size],
            y[i * batch_size : (i + 1) * batch_size],
        )
        for i in range(x.shape[0] // batch_size + 1)
    ]


def shuffle(
    x: ComplexMatrix, y: CategoricalLabels
) -> Tuple[ComplexMatrix, CategoricalLabels]:
    idxs = np.arange(x.shape[0])
    np.random.shuffle(idxs)

    return x[idxs], y[idxs]


def normalize(x: ComplexMatrix) -> ComplexMatrix:
    return wnp.add(
        wnp.div(
            wnp.sub(wnp.real(x), wnp.axis_min(wnp.real(x), axis=0)),
            wnp.sub(
                wnp.axis_max(wnp.real(x), axis=0), wnp.axis_min(wnp.real(x), axis=0)
            ),
        ),
        wnp.div(
            wnp.sub(wnp.imag(x), wnp.axis_min(wnp.imag(x), axis=0)),
            wnp.sub(
                wnp.axis_max(wnp.imag(x), axis=0), wnp.axis_min(wnp.imag(x), axis=0)
            ),
        )
        * 1j,
    )
