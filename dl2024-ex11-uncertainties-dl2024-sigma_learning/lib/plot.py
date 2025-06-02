"""Plotting functions."""
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats
from typing import Union
from lib.model import DNGO, EnsembleFeedForward


def plot_predictions(
    model: DNGO,
    grid: np.array,
    fvals: np.array,
    grid_preds: np.array,
    y_train_pred: np.array,
    X_train: np.array,
    y_train: np.array,
) -> None:
    """ Plot predictions of DNGO model

        Args:
            model: DNGO model
            grid: Grid of values that will be plotted
            fvals: Actual function values
            grid_preds: Model predictions for points in the grid
            y_train_pred: Predictions for the training data
            X_train: Numpy array of shape (n, D) n - number of samples and
                      D -  dimension of sample
            y_train: Numpy array of shape (n, D) n - number of samples and
                      D -  dimension of sample

        Returns:
            None

        Note:
            This plot shows a comparison between ordinary Deep Learning and DNGO.
            Under the label Predicted function values and Predicted train values lie the predictions of a conventional
            Neural Network.
            Under the Mean posterior prediction label lie the predictions of DNGO using the mean of the posterior of
            the weights.

    """

    xlim = (-0, 1)
    ylim = (-2, 2)
    plt.rc('font', size=15.0, family='serif')
    plt.rcParams['figure.figsize'] = (12.0, 8.0)
    # Get mean prediction from DNGO
    pred_y_mean, pred_y_std = model.predict_mean_and_std(torch.from_numpy(
        grid[:, None]).float())
    # Sample posterior predictions from DNGO
    num_samples = 20
    distr_post = scipy.stats.multivariate_normal(model.mu_post.reshape(-1), model.Sigma_post)
    sampled_weights = distr_post.rvs(size=num_samples)
    grid_last_hidden_layer = model.last_hidden_layer(torch.from_numpy(grid[:, None]).float()).detach().numpy()
    bias = np.ones((grid_last_hidden_layer.shape[0], 1))
    grid_last_hidden_layer = np.hstack([grid_last_hidden_layer, bias])
    grid_last_hidden_layer = grid_last_hidden_layer.T
    for i in range(num_samples):
        grid_pred_mean = np.dot(grid_last_hidden_layer.T, sampled_weights[i])
        plt.plot(grid, grid_pred_mean, c='r', alpha=0.05)
    plt.plot([], [], c='r', label="Predicted function values for samples from posterior (DNGO)", alpha=0.2)
    pred_y_mean = np.squeeze(pred_y_mean[:, 0])
    plt.plot(grid, pred_y_mean, c='r', label="Mean posterior prediction (DNGO)", alpha=0.5)

    plt.rc('font', size=15.0, family='serif')
    plt.rcParams['figure.figsize'] = (12.0, 8.0)

    plt.plot(grid, fvals, "k--", label='True function values')

    plt.plot(grid, grid_preds[:, 0], "b", alpha=0.2,
             label='Predicted function values (ordinary DL, not Bayesian)')

    plt.plot(X_train.numpy(), y_train.numpy(), "ko", label='True train values')
    plt.plot(X_train.numpy(), y_train_pred, "bo", label='Predicted train values (ordinary DL, not Bayesian)')
    plt.grid()
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend(loc='lower right')
    plt.xlabel('x value')
    plt.ylabel('y value')

    plt.show()


def plot_uncertainty(model: Union[DNGO, EnsembleFeedForward], X_train: np.array,
                     y_train: np.array, x_test: np.array, y_test: np.array) -> None:
    """ Plot uncertainty of Ensemble or DNGO model

        Args:
            model: DNGO model
            X_train: Numpy array of shape (n1, D) n1 - number of samples and
                      D -  dimension of sample
            y_train: Numpy array of shape (n, D) n1 - number of samples and
                      D -  dimension of sample
            x_test: Numpy array of shape (n2, D) n2 - number of samples and
                      D -  dimension of sample
            y_test: Numpy array of shape (n2, D) n2 - number of samples and
                      D -  dimension of sample

        Returns:
            None

        Note:
            In addition to the grid values, the [train & test] true and predicted values are plotted as well
    """

    xlim = (-0, 1)
    ylim = (-2, 2)

    grid = np.linspace(*xlim, 200, dtype=np.float32)
    fvals = np.sinc(grid * 10 - 5)

    plt.rc('font', size=15.0, family='serif')
    plt.rcParams['figure.figsize'] = (12.0, 8.0)

    grid_mean, grid_std = model.predict_mean_and_std(
        torch.from_numpy(grid[:, None]))
    grid_mean = np.squeeze(grid_mean[:, 0])
    grid_std = np.squeeze(grid_std[:, 0])
    plt.plot(grid, fvals, "k--", label='True values')
    plt.plot(grid, grid_mean, "r--", label='Predicted values')
    plt.fill_between(
        grid,
        grid_mean +
        grid_std,
        grid_mean -
        grid_std,
        color="orange",
        alpha=0.8,
        label="Confidence band 1-std.dev.")
    plt.fill_between(
        grid,
        grid_mean +
        2 *
        grid_std,
        grid_mean -
        2 *
        grid_std,
        color="orange",
        alpha=0.6,
        label="Confidence band 2-std.dev.")
    plt.fill_between(
        grid,
        grid_mean +
        3 *
        grid_std,
        grid_mean -
        3 *
        grid_std,
        color="orange",
        alpha=0.4,
        label="Confidence band 3-std.dev.")

    pred_y, pred_y_std = model.predict_mean_and_std(X_train)
    pred_y = np.squeeze(pred_y[:, 0])

    pred_y_test, pred_y_test_std = model.predict_mean_and_std(x_test)
    pred_y_test = np.squeeze(pred_y_test[:, 0])

    plt.plot(X_train.numpy(), y_train.numpy(), "ko", label='True train values')
    plt.plot(X_train.numpy(), pred_y, "ro", label='Predicted train values')

    plt.plot(x_test.numpy(), y_test.numpy(), "kx", label='True test values')
    plt.plot(x_test.numpy(), pred_y_test, "rx", label='Predicted test values')

    plt.grid()
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    plt.legend(loc='lower right', fontsize='x-small')
    plt.xlabel('x value')
    plt.ylabel('y value')

    plt.show()


def plot_multiple_predictions(model: EnsembleFeedForward, X_train: np.array, y_train: np.array) -> None:
    """ Plot multiple predictions of Ensemble

        Args:

            model: EnsembleFeedForward model
            X_train: Numpy array of shape (n1, D) n1 - number of samples and
                      D -  dimension of sample
            y_train: Numpy array of shape (n, D) n1 - number of samples and
                      D -  dimension of sample

        Returns:
            None
    """

    xlim = (-0, 1)
    ylim = (-2, 2)
    grid = np.linspace(*xlim, 200, dtype=np.float32)
    fvals = np.sinc(grid * 10 - 5)

    plt.rc('font', size=15.0)
    plt.rcParams['figure.figsize'] = (12.0, 8.0)
    plt.plot(grid, fvals, "k--", label='True values')

    # START TODO ########################
    # Hint:
    # Use the individual_predictions() function of the ensemble to get individual predictions
    # (over a grid as in plot_uncertainty) for each of the members of the ensemble.
    # Plot all of these one by one using plot() with some transparency using its alpha parameter
    # so that you get a good visualization of the individual predictions
    # Also plot the mean predictions of the ensemble and remember to label the plotted data.
    preds = model.individual_predictions(torch.from_numpy(grid[:, None]))

    for i in range(preds.shape[2]):
        plt.plot(grid, preds[:, 0, i], alpha=0.2, label="Individual model" if i == 0 else None)

    pred_mean, pred_std = model.predict_mean_and_std(torch.from_numpy(grid[:, None]))
    plt.plot(grid, pred_mean, "r-", label="Mean prediction")
    # END TODO ########################
    pred_y, pred_y_std = model.predict_mean_and_std(X_train)
    pred_y = np.squeeze(pred_y[:, 0])

    plt.plot(X_train.numpy(), y_train.numpy(), "ko", label='True train values')
    plt.plot(X_train.numpy(), pred_y, "ro", label='Predicted train values')
    plt.grid()
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    plt.legend(loc='best')
    plt.xlabel('x value')
    plt.ylabel('y value')

    plt.show()
