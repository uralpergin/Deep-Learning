"""Script to test LSTMCell implementation"""

import numpy as np
import torch

from lib.models import LSTMCell


def test_lstm_cell_internals():
    torch.manual_seed(0)
    cell = LSTMCell(10, 5)

    # randomly sample from [-1, 1]
    candidate_cell_state = torch.rand(2, 5) * 2 - 1
    forget_gate = torch.rand(2, 5)
    input_gate = torch.rand(2, 5) * 2 - 1
    output_gate = torch.rand(2, 5) * 2 - 1
    previous_cell_state = torch.rand(2, 5) * 2 - 1

    current_cell_state = cell.update_internal_state(forget_gate, previous_cell_state, input_gate, candidate_cell_state)
    new_hidden_state = cell.update_hidden_state(current_cell_state, output_gate)

    expected_current_cell_state = torch.Tensor([[-0.315862, -0.459214, -1.042611, 0.597564, 0.503567],
                                                [1.135485, 0.892584, -0.568514, -0.338591, -0.372323]])
    expected_hidden_state = torch.Tensor([[-0.126354, -0.294586, -0.354513, 0.319110, 0.253050],
                                          [0.491627, 0.296364, -0.318606, -0.108354, -0.198028]])

    err_msg = 'LSTMCell update methods not implemented correctly'
    np.testing.assert_almost_equal(
        current_cell_state.detach().numpy(),
        expected_current_cell_state,
        decimal=5,
        err_msg=err_msg)
    np.testing.assert_almost_equal(new_hidden_state.detach().numpy(), expected_hidden_state, decimal=5, err_msg=err_msg)


def test_lstm_cell_foward():
    torch.manual_seed(0)
    cell = LSTMCell(4, 2)

    hx = (torch.rand(2, 2), torch.rand(2, 2))
    x = torch.rand(2, 4)

    new_hidden_state, current_cell_state = cell.forward(x, hx)
    expected_new_hidden_state = torch.Tensor(
        [[0.0666, 0.1642],
         [0.0239, 0.1679]],
    )

    expected_current_cell_state = torch.Tensor(
        [[0.1096, 0.4519],
         [0.0390, 0.5000]],
    )
    err_msg = 'LSTMCell forward pass not implemented correctly'
    np.testing.assert_almost_equal(
        new_hidden_state.detach().numpy(),
        expected_new_hidden_state,
        decimal=4,
        err_msg=err_msg
    )

    np.testing.assert_almost_equal(
        current_cell_state.detach().numpy(),
        expected_current_cell_state,
        decimal=4,
        err_msg=err_msg
    )


if __name__ == '__main__':
    test_lstm_cell_internals()
    test_lstm_cell_foward()
    print('Test complete.')
