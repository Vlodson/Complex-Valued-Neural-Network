from custom_types import NamedObjectDict
from metrics.metric import Metric
from metrics.accuracy import Accuracy

METRICS: NamedObjectDict[Metric] = {"acc": Accuracy()}
