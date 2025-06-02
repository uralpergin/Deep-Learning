import numpy as np
from typing import Tuple
import os
import torch.nn as nn
import torch
from pathlib import Path


def rescaled_sinc() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    np.random.seed(42)
    torch.manual_seed(0)
    n_train = 30
    x_train = np.random.uniform(0, 1, n_train)
    y_train = np.sinc(x_train * 10 - 5)
    x_train = torch.FloatTensor(x_train[:, None])
    y_train = torch.FloatTensor(y_train[:, ])

    x_test = np.random.uniform(0, 1, 20)
    y_test = np.sinc(x_test * 10 - 5)
    x_test = torch.from_numpy(x_test[:, None]).float()
    y_test = torch.from_numpy(y_test[:, ]).float()
    return x_train, y_train, x_test, y_test


def save_result(model: torch.nn, name: str = "") -> None:
    """Save object to disk as pickle file.

    Args:
        filename: Name of file in ./results directory to write object to.
        obj: The object to write to file.

    """
    # make sure save directory exists
    save_path = Path("results/")
    os.makedirs(save_path, exist_ok=True)

    # save the python objects as bytes
    torch.save(model.state_dict(), f"results/model_{name}")


def load_result(model: torch.nn, name: str = "") -> torch.nn:
    """Load object from pickled file.

    Args:
        filename: Name of file in ./results directory to load.

    """
    model_path = Path(f"results/model_{name}")
    model.load_state_dict(torch.load(model_path))


# The initialization of the weights turns out to be crucial for good training with the given hyperparameters,
# so we do that here.

def init_weights(module):
    # torch.manual_seed(0)
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=2)
        nn.init.constant_(module.bias, val=0.0)


def create_ensembled(num_models):

    ensembled_nets = []
    for i in np.arange(num_models):
        temp_model = nn.Sequential(*create_single_model()).apply(init_weights)
        ensembled_nets.append(temp_model)
    return ensembled_nets


def create_single_model():

    module_list = (nn.Linear(in_features=1, out_features=50, bias=True),
                   nn.Tanh(),
                   nn.Linear(in_features=50, out_features=50, bias=True),
                   nn.Tanh(),
                   nn.Linear(in_features=50, out_features=1, bias=True))

    return module_list
