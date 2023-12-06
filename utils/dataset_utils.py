from typing import Tuple

import numpy as np

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
    return (x.real - x.real.min(axis=0)) / (x.real.max(axis=0) - x.real.min(axis=0)) + (
        x.imag - x.imag.min(axis=0)
    ) / (x.imag.max(axis=0) - x.imag.min(axis=0)) * 1j
