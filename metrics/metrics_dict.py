from custom_types import NamedObjectDict
from metrics.f1_score import F1
from metrics.metric import Metric
from metrics.accuracy import Accuracy
from metrics.precision import Precision
from metrics.recall import Recall

METRICS: NamedObjectDict[Metric] = {
    "accuracy": Accuracy(),
    "precision": Precision(macro=True),
    "recall": Recall(macro=True),
    "f1": F1(macro=True),
}
