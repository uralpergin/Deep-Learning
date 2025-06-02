import numpy as np
from lib.experiments import shift_data


def test_black_and_white_images_shift():
    err_msg = 'shift_data() not implemented correctly'
    X = np.arange(18).reshape(2, 1, 3, 3)
    shift_output = shift_data(X, 1)
    X_expected = np.array(
        [[[[0, 0, 0],
           [0, 0, 1],
           [0, 3, 4]]],

         [[[0, 0, 0],
           [0, 9, 10],
           [0, 12, 13]]]]
    )
    print(shift_output)
    np.testing.assert_equal(shift_output, X_expected, err_msg=err_msg)


def test_color_images_shift():
    err_msg = 'shift_data() not implemented correctly'
    X = np.arange(96).reshape(2, 3, 4, 4)
    shift_output = shift_data(X, 2)

    X_expected = np.array(
        [[[[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 4, 5]],
          [[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 16, 17],
           [0, 0, 20, 21]],
          [[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 32, 33],
           [0, 0, 36, 37]]],
         [[[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 48, 49],
           [0, 0, 52, 53]],
          [[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 64, 65],
           [0, 0, 68, 69]],
          [[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 80, 81],
           [0, 0, 84, 85]]]]
    )
    np.testing.assert_equal(shift_output, X_expected, err_msg=err_msg)


if __name__ == '__main__':
    test_black_and_white_images_shift()
    test_color_images_shift()
    print('Tests complete.')
