from typing import Dict, List, Optional

from metrics.metric import Metric


class History:
    def __init__(self) -> None:
        self.history: Dict[str, List[Optional[float]]] = {}

    def add_loss(self) -> None:
        self.history["train_loss"] = [None]
        self.history["val_loss"] = [None]

    def add_metrics(self, metrics: List[Metric]) -> None:
        for metric in metrics:
            self.history["train_" + metric.name] = [None]
            self.history["val_" + metric.name] = [None]

    def update_history(self, key: str, value: float) -> None:
        self.history[key].append(value)

    def get_history_state(self) -> Dict[str, Optional[float]]:
        return {key: value[-1] for key, value in self.history.items()}

    def clean_history(self):
        # remove the Nones
        for metric, metric_values in self.history.items():
            self.history[metric] = metric_values[1:]
