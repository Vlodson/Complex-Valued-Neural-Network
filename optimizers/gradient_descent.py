from typing import List
import wrapped_numpy as wnp
from layers.layer import Layer
from optimizers.optimizer import Optimizer


class GradientDescent(Optimizer):
    def __init__(self, learn_rate: float = 0.1) -> None:
        super().__init__()

        self.learn_rate = learn_rate

    def build(self, layers: List[Layer]) -> None:
        self.optimizable_layers = {layer.name: layer for layer in layers}

    def update_single_layer(self, layer: Layer) -> None:
        layer.weights = wnp.sub(
            layer.weights, wnp.mul(self.learn_rate, layer.weights_grad)
        )
        layer.bias = wnp.sub(layer.bias, wnp.mul(self.learn_rate, layer.bias_grad))

    def update_parameters(self) -> None:
        for _, layer in self.optimizable_layers.items():
            self.update_single_layer(layer)
