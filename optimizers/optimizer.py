from typing import Dict, List
from abc import ABC, abstractmethod

from layers.layer import Layer


class Optimizer(ABC):
    # each optimizer should in it's own init make all the needed properties
    # and use them and only them, no other methods should take in parameters
    def __init__(self) -> None:
        super().__init__()

        self.optimizable_layers: Dict[str, Layer] = {}

    @abstractmethod
    def build(self, layers: List[Layer]) -> None:
        """
        Builds the parameters and properties that couldn't be initialized until now
        """

    @abstractmethod
    def update_parameters(self) -> None:
        """
        Updates all parameters in optimizable layers
        """
