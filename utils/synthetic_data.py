from typing import Tuple

import numpy as np

import wrapped_numpy as wnp
from custom_types import ComplexMatrix, CategoricalLabels
from utils.dataset_utils import shuffle, normalize


def __generate_complex_dataset(samples: int, offset: complex):
    real = np.random.uniform(-1, 1, samples)
    imag = np.random.uniform(-1, 1, samples)

    x = wnp.add(wnp.add(real, imag * 1j), offset)

    return x


def __assign_labels(x: ComplexMatrix):
    labels = np.zeros_like(x, dtype=int)

    labels[(wnp.real(x) >= 0) & (wnp.imag(x) >= 0)] = 0
    labels[(wnp.real(x) < 0) & (wnp.imag(x) >= 0)] = 1
    labels[(wnp.real(x) < 0) & (wnp.imag(x) < 0)] = 2
    labels[(wnp.real(x) >= 0) & (wnp.imag(x) < 0)] = 3

    return labels


def make_quadrant_dataset() -> Tuple[ComplexMatrix, CategoricalLabels]:
    x = wnp.concatenate(
        [
            __generate_complex_dataset(100, 2 + 2j),
            __generate_complex_dataset(100, 2 - 2j),
            __generate_complex_dataset(100, -2 - 2j),
            __generate_complex_dataset(100, -2 + 2j),
        ],
        axis=0,
    )
    y = __assign_labels(x).reshape(-1, 1)

    x = normalize(x).reshape(-1, 1)

    return shuffle(x, y)


def __generate_complex_cloud(center, num_samples, radius):
    angles = np.random.uniform(0, 2 * np.pi, num_samples)
    complex_numbers = wnp.mul(
        wnp.add(center, radius), (wnp.add(wnp.cos_(angles), 1j * wnp.sin_(angles)))
    )
    return complex_numbers


def make_cloud_dataset() -> Tuple[ComplexMatrix, CategoricalLabels]:
    num_samples_per_cloud = 100

    centers = [2 + 2j, -2 + 2j, -2 - 2j, 2 - 2j]
    radii = [1, 1, 1, 1]

    cloud1 = __generate_complex_cloud(centers[0], num_samples_per_cloud, radii[0])
    cloud2 = __generate_complex_cloud(centers[1], num_samples_per_cloud, radii[1])
    cloud3 = __generate_complex_cloud(centers[2], num_samples_per_cloud, radii[2])
    cloud4 = __generate_complex_cloud(centers[3], num_samples_per_cloud, radii[3])

    complex_dataset = wnp.concatenate([cloud1, cloud2, cloud3, cloud4]).reshape(-1, 1)
    labels = wnp.concatenate(
        [
            np.zeros(num_samples_per_cloud),
            np.ones(num_samples_per_cloud),
            2 * np.ones(num_samples_per_cloud),
            3 * np.ones(num_samples_per_cloud),
        ]
    ).reshape(-1, 1)

    complex_dataset = normalize(complex_dataset)

    return shuffle(complex_dataset, labels)


def make_linear_binary_dataset():
    np.random.seed(42)

    x1 = wnp.add(np.random.normal(0, 1, (100, 2)), np.array([5.0, 0.0]))
    x2 = wnp.add(np.random.normal(0, 1, (100, 2)), np.array([-5.0, 0.0]))
    x = wnp.concatenate([x1, x2], axis=0)

    y = np.zeros(x.shape[0])
    y[100:] = 1.0

    x = wnp.div(
        wnp.sub(x, wnp.axis_min(x, axis=0)),
        wnp.sub(wnp.axis_max(x, axis=0), wnp.axis_min(x, axis=0)),
    )

    return shuffle(x, y)


def make_linear_multiclass_dataset():
    np.random.seed(42)

    x1 = wnp.add(np.random.normal(0, 1, (1000, 2)), np.array([-4.0, -4.0]))
    x2 = wnp.add(np.random.normal(0, 1, (1000, 2)), np.array([0.0, 0.0]))
    x3 = wnp.add(np.random.normal(0, 1, (1000, 2)), np.array([4.0, 4.0]))
    x = wnp.concatenate([x1, x2, x3], axis=0)

    y = np.zeros(x.shape[0])
    y[1000:2000] = 1.0
    y[2000:] = 2.0

    x = wnp.div(
        wnp.sub(x, wnp.axis_min(x, axis=0)),
        wnp.sub(wnp.axis_max(x, axis=0), wnp.axis_min(x, axis=0)),
    )

    return shuffle(x, y)
