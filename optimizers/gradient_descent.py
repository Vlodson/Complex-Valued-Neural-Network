from typing import List
from layers.layer import Layer
from optimizers.optimizer import Optimizer


class GradientDescent(Optimizer):
    def __init__(self, learn_rate: float = 0.1) -> None:
        super().__init__()

        self.learn_rate = learn_rate

    def build(self, layers: List[Layer]) -> None:
        self.optimizable_layers = {layer.name: layer for layer in layers}

    def update_single_layer(self, layer: Layer) -> None:
        layer.weights -= self.learn_rate * layer.weights_grad
        layer.bias -= self.learn_rate * layer.bias_grad

    def update_parameters(self) -> None:
        for _, layer in self.optimizable_layers.items():
            self.update_single_layer(layer)
