import numpy as np

from lib.distributions import plot_clt


def test_plot_clt():
    samples, means = plot_clt(n_repetitions=2, sample_sizes=(3, 4), plot=False)

    correct_samples_1 = [
        np.array(
            [[0.30384489, 0.17278791, 0.32590539], [0.61492192, 0.38713495, 0.73062648]]
        ),
        np.array(
            [[0.44274929, 1.33721701, 2.54883597], [-0.37073656, 2.4252914, 0.72053609]]
        ),
        np.array(
            [
                [1.58045489, 0.44826025, 0.53803229, 0.87769071],
                [1.42783234, 0.20801901, 0.33991215, 1.10932605],
            ]
        ),
        np.array(
            [
                [-0.32939545, 0.99854526, -0.31465268, 0.62038826],
                [2.26521065, 1.12066774, 1.14794178, -1.75372579],
            ]
        ),
    ]
    correct_means_1 = [
        np.array([0.26751273, 0.57756111]),
        np.array([1.44293409, 0.92503031]),
        np.array([0.86110954, 0.77127239]),
        np.array([0.24372135, 0.69502359]),
    ]

    correct_means_2 = [
        np.array([0.33896174, 0.5061121]),
        np.array([1.80562555, 0.56233885]),
        np.array([0.97155792, 0.66082401]),
        np.array([0.69227607, 0.24646887]),
    ]
    correct_samples_2 = [
        np.array(
            [
                [0.30384489, 0.17278791],
                [0.32590539, 0.61492192],
                [0.38713495, 0.73062648],
            ]
        ),
        np.array(
            [
                [0.44274929, 1.33721701],
                [2.54883597, -0.37073656],
                [2.4252914, 0.72053609],
            ]
        ),
        np.array(
            [
                [1.58045489, 0.44826025],
                [0.53803229, 0.87769071],
                [1.42783234, 0.20801901],
                [0.33991215, 1.10932605],
            ]
        ),
        np.array(
            [
                [-0.32939545, 0.99854526],
                [-0.31465268, 0.62038826],
                [2.26521065, 1.12066774],
                [1.14794178, -1.75372579],
            ]
        ),
    ]
    for sample, correct_sample_1, mean, correct_mean_1, correct_sample_2, correct_mean_2 in zip(
        samples, correct_samples_1, means, correct_means_1, correct_samples_2, correct_means_2
    ):
        assert (
            sample.shape == correct_sample_1.shape or sample.shape == correct_sample_2.shape
        ), "The sample arrays do not have the correct shape"
        assert (
            mean.shape == correct_mean_1.shape or mean.shape == correct_mean_2.shape
        ), "The mean arrays do not have the correct shape"

        try:
            np.testing.assert_allclose(
                sample,
                correct_sample_1,
                err_msg="Calculation of samples not implemented correctly",
            )
            np.testing.assert_allclose(
                mean, correct_mean_1, err_msg="Calculation of mean not implemented correctly"
            )
        except AssertionError:
            np.testing.assert_allclose(
                sample,
                correct_sample_2,
                err_msg="Calculation of samples not implemented correctly",
            )
            np.testing.assert_allclose(
                mean, correct_mean_2, err_msg="Calculation of mean not implemented correctly"
            )


if __name__ == "__main__":
    test_plot_clt()
    print("Test complete.")
