import numpy as np
import numpy.typing as npt

import wrapped_numpy as wnp
from custom_types import CategoricalLabels, ComplexMatrix


def cat_to_arg_intervals(
    labels: CategoricalLabels,
) -> npt.NDArray[np.float32]:
    """
    Transforms 1D array of categorical labels to 2D array of angle intervals
    Returns a 2D array representing the upper and lower bounds
    of the argument interval for each label
    """
    new_labels = wnp.empty(shape=(labels.shape[0], 2), dtype=np.float32)

    categories = wnp.unique(labels)

    interval_edges = wnp.linspace(0, 2 * np.pi, categories.shape[0] + 1)

    intervals = {
        cat: sorted((interval_edges[i], interval_edges[i + 1]))
        for i, cat in zip(range(interval_edges.shape[0] - 1), categories)
    }

    for cat in categories:
        mask = labels == cat
        new_labels[mask, :] = intervals[cat]

    return new_labels


def cat_to_arg_centers(labels: CategoricalLabels) -> npt.NDArray[np.float32]:
    """
    Splits the unit circle into intervals of equal size based on the number of categories
    Then gives each category a center of one of the intervals
    """
    new_labels = wnp.empty_like(labels)

    cats = wnp.unique(labels)

    intervals = wnp.linspace(0, 2 * np.pi, cats.shape[0] + 1)

    for idx, cat in enumerate(cats):
        new_labels[labels == cat] = (intervals[idx] + intervals[idx + 1]) / 2

    return new_labels


def cat_to_arg(labels: CategoricalLabels) -> npt.NDArray[np.float32]:
    """
    Samples the 2pi interval for number of unique categories uniformly
    Returns those angles as angles for vectors
    """
    new_labels = wnp.empty_like(labels).astype(np.float32)

    cats = wnp.unique(labels)

    intervals = wnp.linspace(0, 2 * np.pi, cats.shape[0] + 1)

    for cat, interval in zip(cats, intervals[:-1]):
        new_labels[labels == cat] = interval

    return new_labels


def cat_to_unit_vector(labels: CategoricalLabels) -> ComplexMatrix:
    args = cat_to_arg_centers(labels)

    return wnp.add(wnp.cos_(args), wnp.sin_(args) * 1.0j)
