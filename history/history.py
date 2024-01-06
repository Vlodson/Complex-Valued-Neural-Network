from typing import Dict, List, Optional

from metrics.metric import Metric
from utils.plotting import plot_complex_loss, plot_real_metrics


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

    def clean_history(self) -> None:
        # remove the Nones
        for metric, metric_values in self.history.items():
            self.history[metric] = metric_values[1:]

    def plot_losses(self) -> None:
        # use only after Nones are removed
        plot_complex_loss(self.history["train_loss"])
        plot_complex_loss(self.history["val_loss"])

    def plot_metrics(self) -> None:
        for metric in [metric for metric in self.history if "loss" not in metric]:
            plot_real_metrics(self.history[metric], metric)
