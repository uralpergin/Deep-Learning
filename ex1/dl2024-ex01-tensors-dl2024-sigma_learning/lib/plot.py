"""Plotting functions."""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from typing import Union
from lib.blr import BLR


def plot_bayes_linear_regression(lbr_1d: BLR, X_train: np.array,
                                 y_train: np.array, num_samples: int = 100) -> None:
    """ Plot sample function from prior and posterior distribution.
        Additionally, plot the mean for prior and posterior distribution
        from which the samples were drawn from.

        Args:
            lbr_1d: Bayesian Linear Regression object
            X_train : Numpy array of shape (n, D) n - number of samples and
                      D -  dimension of sample
            Y_train : Numpy array of shape (n) n - number of samples and
            num_samples: number of samples to draw from prior and posterior
            noise: noise

        Returns:
            None

        Note:
            Prior mu and prior sigma are already defined in run_lin_regression.py
            Please use different colors for prior and posterior samples

    """

    # get quantities
    mu_pre = lbr_1d.Mu_pre
    Sigma_pre = lbr_1d.Sigma_pre

    # call linreg_bayes to get posterior mean and variance
    mu_post, Sigma_post = lbr_1d.linreg_bayes(X_train, y_train)

    # create multivariate guassian distribution once for prior and posterior
    distr_prior = scipy.stats.multivariate_normal(mu_pre, Sigma_pre)
    distr_post = scipy.stats.multivariate_normal(mu_post, Sigma_post)

    # can be useful for plotting the different sampled functions (alpha
    # parameter)
    pdf_max_val = distr_post.pdf(mu_post)

    # sample num_samples times from the prior and posterior distribution
    samples_prior = distr_prior.rvs(size=num_samples)
    samples_pdf_prior = distr_prior.pdf(samples_prior)

    samples_post = distr_post.rvs(size=num_samples)
    samples_pdf_post = distr_post.pdf(samples_post)

    print('Avg weight value (sampled from prior)', samples_prior.mean())
    x = np.linspace(-1.0, 1.0, 100)

    plt.rcParams['figure.figsize'] = (12.0, 8.0)
    plt.scatter(X_train[:, 0], y_train, c='r', s=100, label='Data points', zorder=5000)
    
    # START TODO ########################
    # Plot sampled functions from prior distribution
    for samp in samples_prior:
        y_prior = x * samp
        plt.plot(x, y_prior, color='blue', alpha=0.2, label='Prior samples', zorder=100)

    # END TODO ########################
    print('Avg weight value (sampled from posterior)', samples_post.mean())
    # START TODO ########################
    # Plot sampled functions from posterior distribution
    for samp in samples_post:
        y_post = x * samp
        plt.plot(x, y_post, color='green', alpha=0.5, label='Posterior samples', zorder=100)

    # END TODO ########################
    

    plt.legend()

    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.show()


def plot_contour(lbr_2d: BLR, X_train: np.array, y_train: np.array, num_samples: int = 1000) -> None:
    """ Plot contour plot of the multivariate gaussian distribution for the prior and the posterior
        distribution determined by mean and variance

        Args:
            lbr_2d: Bayesian Linear Regression object for 2-d data
            num_samples: number of samples to draw from prior and posterior
            X_train : Numpy array of shape (n, D) n - number of samples and D -  dimension of sample
            y_train : Numpy array of shape (n) n - number of samples

        Returns:
            None

        Note:
            mu_pre and sigma_pre are already included in lbr_2d
    """
    distr_prior = None
    distr_post = None

    # START TODO #######################
    # Call linreg_bayes function to compute posterior mean and variance based on X_train and y_train
    mu_p, Sigma_p= lbr_2d.linreg_bayes(X_train, y_train)

    # Define prior and posterior distributions
    distr_prior = scipy.stats.multivariate_normal(lbr_2d.Mu_pre, lbr_2d.Sigma_pre)
    distr_post = scipy.stats.multivariate_normal(mu_p, Sigma_p)

    # END TODO ########################

    samples = distr_post.rvs(size=num_samples)
    print('Avg weight values (sampled from posterior)',
          samples[:, 0].mean(), samples[:, 1].mean())

    delta = 0.01
    x, y = np.mgrid[-2.1:2.1:delta, -2.0:2.0:delta]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    plt.figure(figsize=(12, 15))

    plt.subplot(211)
    plt.contourf(x, y, distr_prior.pdf(pos), cmap='Purples')
    plt.title("Prior Contour Lines")
    plt.xlabel('Weight dim 0 values')
    plt.ylabel('Weight dim 1 values')
    plt.colorbar()

    plt.subplot(212)
    plt.contourf(x, y, distr_post.pdf(pos), cmap='Purples')
    plt.title("Posterior Contour Lines")
    plt.xlabel('Weight dim 0 values')
    plt.ylabel('Weight dim 1 values')
    plt.colorbar()

    plt.subplots_adjust(hspace=0.325)
    plt.show()
