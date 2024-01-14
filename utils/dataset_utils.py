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
        for i in range((x.shape[0] + batch_size - 1) // batch_size)
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
            wnp.sub(wnp.real(x), np.min(wnp.real(x), axis=0)),
            wnp.sub(np.max(wnp.real(x), axis=0), np.min(wnp.real(x), axis=0)),
        ),
        wnp.div(
            wnp.sub(wnp.imag(x), np.min(wnp.imag(x), axis=0)),
            wnp.sub(np.max(wnp.imag(x), axis=0), np.min(wnp.imag(x), axis=0)),
        )
        * 1j,
    )
