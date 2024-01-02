from typing import List, Dict, Tuple
import tqdm

import wrapped_numpy as wnp
from history.history import History
from layers.layer import Layer
from losses.loss import Loss
from losses.losses_dict import LOSSES
from metrics.metric import Metric
from metrics.metrics_dict import METRICS
from optimizers.optimizer import Optimizer
from optimizers.optimizers_dict import OPTIMIZERS
from custom_types import CategoricalLabels, ComplexMatrix
from utils.dataset_utils import minibatching
from utils.prediction_utils import vec_to_cat


# works only with complex mse so keep that in mind
class Model:
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers
        self.history = History()

        self.optimizer = None
        self.loss = None
        self.metrics = None

    def build(
        self, optimizer: str | Optimizer, loss: str | Loss, metrics: List[str | Metric]
    ) -> None:
        for idx, layer in enumerate(self.layers):
            _ = (
                layer.build(
                    input_shape=self.layers[idx - 1].neurons, name="layer" + str(idx)
                )
                if idx != 0
                else layer.build(name="layer" + str(idx))
            )

        self.optimizer = (
            optimizer if isinstance(optimizer, Optimizer) else OPTIMIZERS[optimizer]
        )
        self.optimizer.build(self.layers)

        self.loss = loss if isinstance(loss, Loss) else LOSSES[loss]

        self.metrics = [
            metric if isinstance(metric, Metric) else METRICS[metric]
            for metric in metrics
        ]

        self.history.add_loss()
        self.history.add_metrics(self.metrics)

    def compute_loss(
        self, labels: CategoricalLabels, predictions: ComplexMatrix
    ) -> float:
        return self.loss.calculate_loss(labels, predictions)

    def compute_metrics(
        self, labels: CategoricalLabels, predictions: CategoricalLabels
    ) -> Dict[str, float]:
        return {
            metric.name: metric.calculate_metric(labels, predictions)
            for metric in self.metrics
        }

    def __forward(self, x: ComplexMatrix) -> None:
        for idx, layer in enumerate(self.layers):
            _ = layer.forward(self.layers[idx - 1].y) if idx != 0 else layer.forward(x)

    def __back(self, y_grad: ComplexMatrix) -> None:
        reversed_layers = self.layers[::-1]
        for idx, layer in enumerate(reversed_layers):
            _ = (
                layer.backward(reversed_layers[idx - 1].x_grad)
                if idx != 0
                else layer.backward(y_grad)
            )

    def __update_metric_state(
        self, old_state: Dict[str, float], new_state: Dict[str, float]
    ) -> Dict[str, float]:
        if not old_state:
            return new_state

        for metric in new_state:
            old_state[metric] = wnp.add(old_state[metric], new_state[metric])

        return old_state

    def __update_train_metric_history(
        self, metric_state: Dict[str, float], num_of_states: int
    ) -> None:
        for metric, state in metric_state.items():
            self.history.update_history(
                "train_" + metric, wnp.div(state, num_of_states)
            )

    def __update_validation_metric_history(self, metric_state: Dict[str, float]):
        for metric, state in metric_state.items():
            self.history.update_history("val_" + metric, state)

    def test(
        self, x: ComplexMatrix, y: CategoricalLabels
    ) -> Tuple[float, Dict[str, float]]:
        # does forward on x, evaluates loss and metrics on y returns those
        raise NotImplementedError

    def train(
        self,
        x: ComplexMatrix,
        y: CategoricalLabels,
        batch_size: int,
        epochs: int,
        x_val: ComplexMatrix,
        y_val: CategoricalLabels,
    ) -> None:
        num_of_cats = wnp.unique(y).shape[0]

        batches = minibatching(x, y, batch_size)

        for _ in tqdm.tqdm(range(epochs)):
            # validation part
            self.__forward(x_val)
            self.history.update_history(
                "val_loss", self.compute_loss(y_val, self.layers[-1].y)
            )
            self.__update_validation_metric_history(
                self.compute_metrics(y_val, vec_to_cat(self.layers[-1].y, num_of_cats))
            )

            # training part
            batch_loss: List[float] = []
            batch_metrics: Dict[str, float] = {}
            for x_batch, y_batch in batches:
                self.__forward(x_batch)

                batch_loss.append(self.compute_loss(y_batch, self.layers[-1].y))
                batch_metrics = self.__update_metric_state(
                    batch_metrics,
                    self.compute_metrics(
                        y_batch, vec_to_cat(self.layers[-1].y, num_of_cats)
                    ),
                )

                y_batch_grad = self.loss.loss_gradient(y_batch, self.layers[-1].y)
                self.__back(y_batch_grad)

                self.optimizer.update_parameters()

            self.history.update_history("train_loss", sum(batch_loss))
            self.__update_train_metric_history(batch_metrics, len(batches))

        self.history.clean_history()

    def predict(self, x: ComplexMatrix) -> ComplexMatrix:
        self.__forward(x)
        return self.layers[-1].y
