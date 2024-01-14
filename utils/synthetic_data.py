from typing import Tuple

import numpy as np
import numpy.typing as npt

from custom_types import ComplexMatrix, CategoricalLabels
from utils.dataset_utils import shuffle, normalize

np.random.seed(42)


def __make_comlpex_feature(points: int, center: npt.ArrayLike) -> ComplexMatrix:
    c = np.random.normal(loc=center, scale=np.array([1, 1]), size=(points, 2))

    return (c[..., 0] + 1j * c[..., 1]).reshape(-1, 1)


def binary_classification(features: int) -> Tuple[ComplexMatrix, CategoricalLabels]:
    x1 = np.concatenate(
        [
            __make_comlpex_feature(points=1000, center=np.array([-5, -5]))
            for _ in range(features)
        ],
        axis=1,
    )
    x2 = np.concatenate(
        [
            __make_comlpex_feature(points=1000, center=np.array([5, 5]))
            for _ in range(features)
        ],
        axis=1,
    )
    x = np.concatenate([x1, x2], axis=0)

    y1 = np.zeros(shape=(1000,))
    y2 = np.ones(shape=(1000,))
    y = np.concatenate([y1, y2], axis=0)

    return shuffle(normalize(x), y)


def __nroots_of_i(n: int) -> npt.NDArray:
    root = np.abs(1.0j) ** (1.0 / n)
    period = np.angle(1.0j)

    return np.array(
        list(
            map(lambda k: root * np.exp((period + 2 * k * np.pi) * 1.0j / n), range(n))
        )
    )


def multiclass_classification(
    features: int, classes: int
) -> Tuple[ComplexMatrix, CategoricalLabels]:
    points_per_class = int(1000 / classes)

    # make the unit circle bigger to give more space for distances between clusters
    # drops off the more clusters you have
    centers = __nroots_of_i(classes) * 7

    x = np.concatenate(
        [
            np.concatenate(
                [
                    __make_comlpex_feature(
                        points=points_per_class, center=(center.real, center.imag)
                    )
                    for _ in range(features)
                ],
                axis=1,
            )
            for center in centers
        ],
        axis=0,
    )

    y = np.concatenate(
        [np.array([cat] * points_per_class) for cat in range(classes)], axis=0
    )

    return shuffle(normalize(x), y)
