"""Helper functions for working with models."""

import numpy as np

from lib.network import Sequential


def extract_hidden(full_model: Sequential, x: np.ndarray) -> np.ndarray:
    """Function to extract the hidden representation from a 2-layer MLP.
    Compute `h` (hidden representation) after propagating the `x` (input)
    through the first Linear layer and the activation function ReLU
    of `full_model`.

    Args:
        full_model: The 2-layer MLP used as a classifier.
        x: Input examples with shape (nr_examples, nr_features).

    Returns:
        h: Hidden representation of inputs with shape (nr_examples, nr_features).
    """
    # START TODO #################
    # Get the first linear layer
    linear_layer = full_model.modules[0]

    # Change the weights of the first Linear layer according to the exercise sheet
    linear_layer.W.data = np.array([[1, 1],
                                   [1, 1]])
    linear_layer.b.data = np.array([0, -1])
    relu_layer = full_model.modules[1]

    # Extract the hidden features from the 2-layer MLP and compute the hidden representation after propagating
    linear_output = linear_layer.forward(x)
    h = relu_layer.forward(linear_output)

    return h
    # END TODO ##################
