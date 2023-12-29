from typing import Optional

import numpy as np
import numpy.typing as npt

from custom_types import ComplexMatrix
from activations.activation import Activation
from activations.activations_dict import ACTIVATIONS


class Layer:
    def __init__(
        self,
        neurons: int,
        activation: Activation | str,
        input_shape: Optional[int] = None,
        name: Optional[str] = None,
    ):
        self.neurons: Optional[int] = neurons
        self.input_shape: Optional[int] = input_shape

        self.x: ComplexMatrix = np.empty((0, 0), dtype=np.complex64)
        self.transfer: ComplexMatrix = np.empty((0, 0), dtype=np.complex64)
        self.y: ComplexMatrix = np.empty((0, 0), dtype=np.complex64)

        self.transfer_grad: ComplexMatrix = np.empty((0, 0), dtype=np.complex64)
        self.x_grad: ComplexMatrix = np.empty((0, 0), dtype=np.complex64)

        self.weights: ComplexMatrix = np.empty((0, 0), dtype=np.complex64)
        self.bias: ComplexMatrix = np.empty((0, 0), dtype=np.complex64)

        self.weights_grad: ComplexMatrix = np.empty((0, 0), dtype=np.complex64)
        self.bias_grad: ComplexMatrix = np.empty((0, 0), dtype=np.complex64)

        self.activation: Activation = (
            activation
            if isinstance(activation, Activation)
            else ACTIVATIONS[activation]
        )

        self.name: str = name

    def build(
        self, input_shape: Optional[int] = None, name: Optional[str] = None
    ) -> None:
        """
        Inits values for properties that couldn't be built on init
        """
        self.name = self.name if self.name is not None else name
        self.input_shape = (
            self.input_shape if self.input_shape is not None else input_shape
        )

        self.weights = (
            np.random.uniform(-1.0, 1.0, (self.input_shape, self.neurons))
            + np.random.uniform(-1.0, 1.0, (self.input_shape, self.neurons)) * 1.0j
        )
        self.bias = (
            np.random.uniform(-1.0, 1.0, (1, self.neurons))
            + np.random.uniform(-1.0, 1.0, (1, self.neurons)) * 1.0j
        )

    def __transfer(self, x: ComplexMatrix) -> None:
        """
        Sets the transfer field given some input x
        """
        self.transfer = x @ self.weights + self.bias

    def __activation(self) -> None:
        """
        Sets the y property by activating the transfer
        """
        self.y = self.activation.activate(self.transfer)

    def forward(self, x: ComplexMatrix) -> None:
        """
        Computes feed forward for a single layer
        """
        self.x = x
        self.__transfer(self.x)
        self.__activation()

    def __deactivate(self, y_grad: ComplexMatrix) -> None:
        """
        Deactivates the output for brackpropagation. Sets the transfer gradient
        """
        self.transfer_grad = self.activation.deactivate(self.transfer) * y_grad

    def __compute_x_grad(self) -> None:
        """
        Computes the gradient of the input based on the transfer gradient.
        Sets x_grad
        """
        self.x_grad = self.transfer_grad @ self.weights.T

    def __compute_weights_grad(self) -> None:
        """
        Computes the gradient of the weights. Sets weights_grad
        """
        self.weights_grad = self.x.T @ self.transfer_grad

    def __compute_bias_grad(self) -> None:
        """
        Computes the gradient of the biases. Sets bias_grad
        """
        self.bias_grad = np.sum(self.transfer_grad, axis=0).reshape(1, -1)

    def backward(self, y_grad: ComplexMatrix | npt.NDArray[np.float32]) -> None:
        """
        Does backprop for one layer
        """
        self.__deactivate(y_grad)
        self.__compute_x_grad()
        self.__compute_weights_grad()
        self.__compute_bias_grad()
