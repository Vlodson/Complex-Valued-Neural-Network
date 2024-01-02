from typing import List, Dict

import numpy.typing as npt

import wrapped_numpy as wnp
from layers.layer import Layer
from optimizers.optimizer import Optimizer


class ADAM(Optimizer):
    def __init__(
        self,
        learn_rate: float = 0.01,
        w1: float = 0.9,
        w2: float = 0.99,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()

        self.learn_rate = learn_rate
        self.w1 = w1
        self.w2 = w2
        self.eps = eps

        self.ms: Dict[str, Dict[str, npt.NDArray]] = {}
        self.vs: Dict[str, Dict[str, npt.NDArray]] = {}

    def build(self, layers: List[Layer]) -> None:
        self.optimizable_layers = {layer.name: layer for layer in layers}

        self.ms = {
            layer.name: {
                "weights": np.zeros_like(layer.weights),
                "bias": np.zeros_like(layer.bias),
            }
            for layer in layers
        }
        self.vs = {
            layer.name: {
                "weights": np.zeros_like(layer.weights),
                "bias": np.zeros_like(layer.bias),
            }
            for layer in layers
        }

    def update_single_layer_param(self, layer: Layer, param: str) -> None:
        new_m = wnp.add(
            wnp.mul(self.w1, self.ms[layer.name][param]),
            wnp.mul((wnp.sub(1, self.w1)), getattr(layer, param + "_grad")),
        )
        new_v = wnp.add(
            wnp.mul(self.w2, self.vs[layer.name][param]),
            wnp.mul((wnp.sub(1, self.w2)), wnp.pwr(getattr(layer, param + "_grad"), 2)),
        )

        m_hat = wnp.div(new_m, (wnp.sub(1, self.w1)))
        v_hat = wnp.div(wnp.abs_(new_v), (wnp.sub(1, self.w2)))

        new_param = wnp.sub(
            getattr(layer, param),
            wnp.mul(
                wnp.div(self.learn_rate, wnp.pwr(wnp.add(v_hat, self.eps), 0.5)),
                m_hat,
            ),
        )
        setattr(layer, param, new_param)

    def update_single_layer(self, layer) -> None:
        self.update_single_layer_param(layer, "weights")
        self.update_single_layer_param(layer, "bias")

    def update_parameters(self) -> None:
        for _, layer in self.optimizable_layers.items():
            self.update_single_layer(layer)
