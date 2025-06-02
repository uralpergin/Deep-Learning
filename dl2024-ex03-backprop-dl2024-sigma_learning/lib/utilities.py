"""Helper functions for data conversion and working with models."""

import numpy as np
from typing import List

from lib.network_base import Module, Parameter
from lib.activations import ReLU


def one_hot_encoding(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one hot encoding.

    Example: y=[1, 2], num_classes=3 --> [[0, 1, 0], [0, 0, 1]]

    Args:
        y: Input labels as integers with shape (num_datapoints)
        num_classes: Number of possible classes

    Returns:
        One-hot encoded labels with shape (num_datapoints, num_classes)
    """
    encoded = np.zeros(y.shape + (num_classes,))
    encoded[np.arange(len(y)), y] = 1
    return encoded


class Dummy(Module):
    """Dummy module.

    Note:
        This module is only used to check ReLU implementation in test_gradient_relu
    """

    def __init__(self):
        super().__init__()
        self.relu1 = ReLU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass calculation for the Dummy module.

        Args:
            x: Input data with shape (batch_size, in_features).

        Returns:
            Output data with shape (batch_size, out_features).

        """
        x = -x + 0.5 * np.ones_like(x)
        x = self.relu1.forward(x)
        x = -x + 0.5 * np.ones_like(x)
        self.input_cache = x
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Calculate the backward pass of the Dummy module.

        Args:
            grad: The gradient of the following layer with shape (batch_size, out_features).
                I.e. the given partial derivative of the loss w.r.t. this module's outputs.

        Returns:
            The gradient of this module with shape (batch_size, in_features).
                I.e. The partial derivative of the loss loss w.r.t. this module's inputs.

        """
        # grad will alrady be populated from check_gradients
        grad = grad * -1
        grad = self.relu1.backward(grad)
        grad = grad * -1
        return grad

    def parameters(self) -> List[Parameter]:
        """Return module parameters.

        Returns:
            All learnable parameters of the linear module.
        """
        return []
