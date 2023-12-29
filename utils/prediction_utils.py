import numpy as np
from numpy import typing as npt

from custom_types import ComplexMatrix, CategoricalLabels
from utils.label_utils import cat_to_arg_centers, cat_to_unit_vector


def arg_to_cat(
    arguments: npt.NDArray[np.float32], num_of_cats: int
) -> npt.NDArray[np.int_]:
    """
    Turns arguments to categorical values based on the number of classes
    """
    cat_predictions = np.empty(shape=(arguments.shape[0],), dtype=np.int_)

    interval_edges = np.linspace(0, 2 * np.pi, num_of_cats + 1)

    intervals = {
        cat: (interval_edges[i], interval_edges[i + 1])
        for i, cat in zip(range(interval_edges.shape[0] - 1), np.arange(num_of_cats))
    }

    for cat, interval in intervals.items():
        mask = (interval[0] <= arguments) & (arguments < interval[1])
        cat_predictions[mask] = cat

    return cat_predictions


def vec_to_cat(predictions: ComplexMatrix, num_of_cats: int) -> CategoricalLabels:
    labels = np.arange(num_of_cats)
    vec_labels = cat_to_unit_vector(labels)

    return np.argmin(np.absolute(predictions - vec_labels), axis=1)
