"""Helper functions for data conversion and file handling."""

import os
import pickle
from pathlib import Path
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np


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


def save_result(filename: str, obj: object) -> None:
    """Save object to disk as pickle file.

    Args:
        filename: Name of file in ./results directory to write object to.
        obj: The object to write to file.

    """
    # make sure save directory exists
    save_path = Path("results")
    os.makedirs(save_path, exist_ok=True)

    # save the python objects as bytes
    with (save_path / f"{filename}.pkl").open("wb") as fh:
        pickle.dump(obj, fh)


def load_result(
    filename: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load object from pickled file.

    Args:
        filename: Name of file in ./results directory to load.

    """
    with (Path("results") / f"{filename}.pkl").open("rb") as fh:
        return pickle.load(fh)


def ill_conditioned_matrix(condition_number: int) -> np.ndarray:
    """Create a matrix that has the given condition number.

    Args:
        condition_number (int): Value of max(eigenvalues) / min(eigenvalues).
    Returns:
        numpy.ndarray: A 2x2 diagonal matrix that has the given condition number.
    """
    eig2 = 1
    eig1 = eig2 * condition_number
    Q = np.diag([eig1, eig2])
    return Q


def plot_contours(
    Q, x_min: float = -10, x_max: float = 10, y_min: float = -10, y_max: float = 10
) -> None:
    """
    Plot contours of the quadratic function.

    Parameters:
        Q (numpy.ndarray): The quadratic matrix.
        x_min (float): The minimum value for the x-axis. Default is -10.
        x_max (float): The maximum value for the x-axis. Default is 10.
        y_min (float): The minimum value for the y-axis. Default is -10.
        y_max (float): The maximum value for the y-axis. Default is 10.

    Returns:
        None
    """
    x = np.linspace(x_min, x_max, 120)
    y = np.linspace(y_min, y_max, 120)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (Q[0, 0] * X**2 + (Q[0, 1] + Q[1, 0]) * X * Y + Q[1, 1] * Y**2)
    plt.contour(X, Y, Z, cmap="RdBu")


def plot_colormesh(Q: np.ndarray, params: list) -> None:
    """
    Plot a colormesh plot based on the given parameters.

    Args:
        Q (numpy.ndarray): A 2x2 matrix representing the quadratic coefficients.
        params (numpy.ndarray): An array of shape (n, 2) containing the x1 and x2 coordinates.

    Returns:
        None
    """

    X = np.linspace(params[:, 0].min() - 0.1, params[:, 0].max() + 0.1, 120)
    Y = np.linspace(params[:, 1].min() - 0.1, params[:, 1].max() + 0.1, 120)
    X, Y = np.meshgrid(X, Y)
    Z = ((X * Q[0, 0] + Y * Q[1, 0]) * X + (X * Q[0, 1] + Y * Q[1, 1]) * Y) / 2
    plt.pcolormesh(
        X, Y, Z, cmap="RdBu", linewidth=0.08, antialiased=True, shading="auto"
    )
