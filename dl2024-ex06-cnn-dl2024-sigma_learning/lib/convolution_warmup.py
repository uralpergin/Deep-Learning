import numpy as np
from typing import Tuple


def pad_array(X: np.ndarray, pad: Tuple[int, int]) -> np.ndarray:
    """Pad the given numpy array with the given padding values.

    Args:
        X: Array of shape (N_batch, n_channels, height, width).
        pad: Tuple of shape (n_height_pad, n_width_pad).

    Returns:
        Array X padded according to parameter pad.
    """

    # START TODO ################
    # Use np.pad to pad the array according to the given padding
    height_pad, width_pad = pad
    result = np.pad(
        X,
        pad_width=((0, 0), (0, 0), (height_pad, height_pad), (width_pad, width_pad)),
        mode='constant',  # Pads with zeros by default
    )
    # END TODO ################
    return result


def dot_product_single_filter(X: np.ndarray, F: np.ndarray, index: Tuple[int, int]) -> float:
    """Calculate the dot product of a small chunk of X with a single filter.

    Args:
        X: Array of shape (n_channels, image_height, image_width).
        F: The filter to apply. Array of shape (n_channels, filter_height, filter_width).
        index: Index of X where the top left corner of the filter should be placed.

    Returns:
        Scalar value which is the sum of product of the filter and section of X.

        For example, consider that your input X is a 5x5 image with only one channel as follows:

        X =
        [[[ 0,  1,  2,  3,  4],
          [ 5,  6,  7,  8,  9],
          [10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19],
          [20, 21, 22, 23, 24]]]

        X.shape = (1, 5, 5)

        and we're using a 3x3 kernel as follows:

        F =
        [[[0, 1, 2],
          [3, 4, 5],
          [6, 7, 8]]]

        F.shape = (1, 3, 3)

        If the given index is (2, 1), then you should compute the sum of the elements of :

        [[[11, 12, 13],              [[[0, 1, 2],
          [16, 17, 18],       *        [3, 4, 5],
          [21, 22, 23]]]               [6, 7, 8]]]

        where * is an elementwise multiplication operation.

        Assume that the kernel will not go outside the boundaries of X. For example, in this case,
        the index (2, 3) is invalid.

        The scalar value that is returned corresponds to a single element in a single activation
        map (see slide 30).
    """
    result = 0
    # START TODO ################
    ind_1 = index[0]
    ind_2 = index[1]
    for k in range(F.shape[0]):
        for i in range(F.shape[1]):
            for j in range(F.shape[2]):

                result += X[k][ind_1 + i][ind_2 + j] * F[k][i][j]
    # END TODO ################

    return result


def dot_product_single_filter_with_batches(X: np.ndarray, F: np.ndarray, index: Tuple[int, int]) -> np.ndarray:
    """Calculate the dot product of a chunk of X with the a single filter, where X has batch_size number of entries.

    Args:
        X: Array of shape (batch_size, n_channels, image_height, image_width).
        F: The filter to apply. Array of shape (n_channels, filter_height, filter_width).
        index: Index of X where the top left corner of the filter should be placed.

    Returns:
        np.ndarray of shape (batch_size, ) which has the sum of product of the filter and section of X
        for each batch.

        This function is practically the same as dot_product_single_filter above but with a new axis in X for the batch.
        Accordingly, it returns a np array of shape (batch_size, ).
    """

    result = np.zeros((len(X)))

    # START TODO ################
    ind_1 = index[0]
    ind_2 = index[1]
    for c in range(len(result)):
        for k in range(F.shape[0]):
            for i in range(F.shape[1]):
                for j in range(F.shape[2]):
                    result[c] += X[c][k][ind_1 + i][ind_2 + j] * F[k][i][j]
    # END TODO ################

    return result


def dot_product_multiple_filters_with_batches(X: np.ndarray, filters: np.ndarray, index: Tuple[int, int]) -> np.ndarray:
    """Calculate the dot product of a chunk of X with a number of filters, where X has batch_size number of entries.

    Args:
        X: Array of shape (batch_size, n_channels, image_height, image_width)
        filters: The filters to apply. Array of shape (n_filters, n_channels, filter_height, filter_width)

    Returns:
        np.ndarray of shape (batch_size, n_filters)

        This function is the same as dot_product_single_filter_with_batches above, except now, we consider more than
        one filters.
    """

    # START TODO ################
    # Hint: ensure that the shape of the filters array matches the description in the function docstring
    # The filters and section of X no longer have shapes which can be broadcasted together.
    # Use np.expand_dims to fix that.
    result = np.zeros((len(X), len(filters)))

    ind_1 = index[0]
    ind_2 = index[1]
    for c in range(len(result)):
        for z in range(filters.shape[0]):
            for k in range(filters.shape[1]):
                for i in range(filters.shape[2]):
                    for j in range(filters.shape[3]):
                        result[c][z] += X[c][k][ind_1 + i][ind_2 + j] * filters[z][k][i][j]

    # END TODO ################

    return result
