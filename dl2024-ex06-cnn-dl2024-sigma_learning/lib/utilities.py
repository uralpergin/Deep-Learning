"""Helper functions for data conversion and file handling."""

import os
import pickle
from pathlib import Path
from typing import Tuple, Dict

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
    with (save_path / f"{filename}.pkl").open('wb') as fh:
        pickle.dump(obj, fh)


def load_result(filename: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load object from pickled file.

    Args:
        filename: Name of file in ./results directory to load.

    """
    with (Path("results") / f"{filename}.pkl").open('rb') as fh:
        return pickle.load(fh)


def ill_conditioned_matrix(condition_number):
    """Create a matrix that has given condition number.

    Args:
        condition_number: Value of max(eigenvalues) / min(eigenvalues).

    """
    eig2 = 0.1
    eig1 = eig2 * condition_number
    v = np.array([[0.05, -0.02], [0.03, 0.04]])
    lambd = np.array([[eig1, 0], [0, eig2]])
    Q = v @ lambd @ np.linalg.inv(v)
    return Q


def calculate_precondition_matrix(Q):
    """Calculate inverse hessian of given matrix.

    Args:
        Q: Matrix to calculate the precondition matrix.

    """
    a = (Q[0, 1] + Q[1, 0]) / 2
    H = [[Q[0, 0], a],
         [a, Q[1, 1]]]
    H_inv = np.linalg.inv(H)
    return H_inv
