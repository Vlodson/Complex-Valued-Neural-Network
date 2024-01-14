from typing import Tuple

import numpy as np

from custom_types import ComplexMatrix, CategoricalLabels
from utils.dataset_utils import shuffle, normalize

np.random.seed(42)


def __make_comlpex_feature(points: int, center: float) -> ComplexMatrix:
    c = np.random.normal(center, 1, size=(points, 2))

    return (c[..., 0] + 1j * c[..., 1]).reshape(-1, 1)


def binary_classification(features: int) -> Tuple[ComplexMatrix, CategoricalLabels]:
    x1 = np.concatenate(
        [__make_comlpex_feature(points=1000, center=-5) for _ in range(features)],
        axis=1,
    )
    x2 = np.concatenate(
        [__make_comlpex_feature(points=1000, center=5) for _ in range(features)], axis=1
    )
    x = np.concatenate([x1, x2], axis=0)

    y1 = np.zeros(shape=(1000,))
    y2 = np.ones(shape=(1000,))
    y = np.concatenate([y1, y2], axis=0)

    return shuffle(normalize(x), y)
