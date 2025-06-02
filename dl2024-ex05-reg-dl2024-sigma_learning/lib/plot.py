"""Weight distribution plot."""

import matplotlib.pyplot as plt
import numpy as np

from lib.utilities import load_result


def plot_distribution() -> None:
    """Plot the histograms of the weight values of the five models (before_training, no regularization,
    L1 regularization, L2 regularization and Dropout) from -1 to 1 with 100 bins.

    Returns:
        None
    """

    models = load_result('trained_models')

    fig, axs = plt.subplots(figsize=(12, 6), ncols=5, sharey=True)
    axs[0].set(yscale='log', ylabel='total frequency')
    for model, ax in zip(models.keys(), axs):
        ax.set(title=model, xlabel='value')
        # START TODO ################
        # Retrieve all the weights (exclude the biases!) of the parameters for each of the five models
        # (before training, no regularization, L1 regularization, L2 regularization and Dropout)
        # and then plot the histogram as specified
        weights = np.concatenate(
            [param.data.flatten() for param in models[model].parameters() if "W" in param.name]
        )

        ax.hist(weights, bins=100, range=(-1, 1), color='blue', alpha=0.7)

        # END TODO ################
    fig.tight_layout()
    plt.show()
