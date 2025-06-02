"""Plotting functions."""

import matplotlib.pyplot as plt
import numpy as np

from lib.lr_schedulers import CosineAnnealingLR, PiecewiseConstantLR
from lib.network_base import Parameter
from lib.optimizers import SGD, Adam
from lib.utilities import (
    ill_conditioned_matrix,
    load_result,
    plot_colormesh,
    plot_contours,
)


def plot_learning_curves() -> None:
    """Plot the performance of SGD, SGD with momentum, and Adam optimizers.

    Note:
        This function requires the saved results of compare_optimizers() above, so make
        sure you run compare_optimizers() first.
    """
    optim_results = load_result("optimizers_comparison")
    # START TODO ################
    # train result are tuple(train_costs, train_accuracies, eval_costs,
    # eval_accuracies). You can access the iterable via
    # optim_results.items()
    colors = {"sgd": "blue", "sgd_momentum": "green", "adam": "red"}

    plt.figure(figsize=(8, 6))
    plt.title("Training Loss Curves")
    for opt_name, (train_costs, _, _, _) in optim_results.items():
        plt.plot(train_costs, label=f"{opt_name}", color=colors[opt_name])
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("Training Accuracy Curves")
    for opt_name, (_, train_accuracies, _, _) in optim_results.items():
        plt.plot(train_accuracies, label=f"{opt_name}", color=colors[opt_name])
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
    # END TODO ###################


def plot_lr_schedules() -> None:
    """Plot the learning rate schedules of piecewise and cosine schedulers."""
    num_epochs = 80
    base_lr = 0.1

    piecewise_scheduler = PiecewiseConstantLR(
        Adam([], lr=base_lr), [10, 20, 40, 50], [0.1, 0.05, 0.01, 0.001]
    )
    cosine_scheduler = CosineAnnealingLR(Adam([], lr=base_lr), num_epochs)

    # START TODO ################
    # plot piecewise lr and cosine lr
    piecewise_lrs = []
    cosine_lrs = []

    for epoch in range(num_epochs):
        piecewise_lrs.append(piecewise_scheduler.optimizer.lr)
        cosine_lrs.append(cosine_scheduler.optimizer.lr)

        piecewise_scheduler.step()
        cosine_scheduler.step()

    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), piecewise_lrs, label="Piecewise Constant LR")
    plt.plot(range(num_epochs), cosine_lrs, label="Cosine Annealing LR")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedules")
    plt.legend()
    plt.show()
    # END TODO ################


def plot_ill_conditioning(
    condition_number: int,
    momentum: float,
    preconditioning: bool,
    lr: float,
    init: tuple,
    steps: int = 500,
) -> None:
    """Plot optimization steps of a quadratic function with ill conditioned matrix.

    Args:
        condition_number (int): the biggest eigenvalue divided by the smallest eigenvalue.
        momentum (float): SGD momentum coefficient.
        preconditioning (bool): If true, apply preconditioning to the gradient.
        lr (float): learning rate of the optimizer.
        init (tuple): starting point.
        steps (int): number of optimization steps.
    """
    x = Parameter(np.array(init, dtype=np.float64))

    # Create an ill-conditioned matrix with the given condition number
    Q = ill_conditioned_matrix(condition_number)

    # initialize the optimizer
    optimizer = SGD([x], lr=lr, momentum=momentum)
    params = [
        x.data.copy(),
    ]

    # run the optimizer for a number of steps
    for _ in range(steps):
        optimizer.zero_grad()

        # function that we aim to minimize
        f = x.data.T @ Q.data @ x.data / 2.0
        # assume we reached the goal
        if f < 1e-5:
            break
        nabla_f = Q @ x.data
        # populate gradient of x Parameter for optimizer
        x.grad.data = nabla_f

        if preconditioning:
            x.grad.data = np.linalg.inv(Q) @ x.grad

        optimizer.step()
        params.append(x.data.copy())
    params = np.array(params)

    if np.isnan(params).any():
        print("-" * 80)
        print("\n>>>> Optimization diverged. Try reducing the learning rate.\n")
        print("-" * 80)
        return

    print(f"final function value: {f}")
    # plot contours of the quadratic function
    plot_contours(
        Q,
        x_min=params[:, 0].min() - 0.1,
        x_max=params[:, 0].max() + 0.1,
        y_min=params[:, 1].min() - 0.1,
        y_max=params[:, 1].max() + 0.1,
    )

    # plot colormesh
    plot_colormesh(Q, params)
    plt.plot(params[:, 0], params[:, 1], "k-", marker="x", markersize=5, linewidth=1)
    # Plot the initial point x
    plt.plot(init[0], init[1], "ro", label="Initial point")
    # Plot the final point x
    plt.plot(params[-1, 0], params[-1, 1], "bo", label="Final point")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title(
        r"$\kappa({})$, momentum: {}, preconditioned: {}".format(
            condition_number, momentum, preconditioning
        )
    )
    plt.colorbar(label="Function Value")
    plt.show()
