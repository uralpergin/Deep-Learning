"""ReLU plot."""

import matplotlib.pyplot as plt
import numpy as np

from lib.activations import ReLU


def plot_relu() -> None:
    """Plot the ReLU function in the range (-4, 4).

    Returns:
        None
    """
    # START TODO #################
    # Create input data, run through ReLU and plot.
    x = np.linspace(-4, 4, 400)
    relu_mod = ReLU()
    y = relu_mod.forward(x)

    # Plot the ReLU function
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label="ReLU", color="blue")
    plt.title("ReLU Activation Function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.show()
    # END TODO###################
