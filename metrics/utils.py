import wrapped_numpy as wnp
from custom_types import CategoricalLabels, MetricInterResults


def __get_correct(
    predictions: CategoricalLabels, labels: CategoricalLabels, cat: int
) -> int:
    return wnp.sum_((predictions == cat) & (labels == cat))


def __get_instances(labels: CategoricalLabels, cat: int) -> int:
    return wnp.sum_(labels == cat)


def __get_all_clf(predictions: CategoricalLabels, cat: int) -> int:
    return wnp.sum_(predictions == cat)


def get_inter_results(
    predictions: CategoricalLabels, labels: CategoricalLabels, cat: int
) -> MetricInterResults:
    return {
        "correct": __get_correct(predictions, labels, cat),
        "instances": __get_instances(labels, cat),
        "all_clf": __get_all_clf(predictions, cat),
    }
