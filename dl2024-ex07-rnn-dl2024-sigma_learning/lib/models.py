"""LSTM and noise removal models"""

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """The LSTM layer."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # You should define a single linear layer which takes the current input
        # and the hidden state as input, and outputs the linear transformation of all gates
        # You will chunk the output of the linear layer to four predictions
        # (candidate_cell_state, forget_gate, input_gate, output_gate) (we recommend that order)
        # during the forward propagation
        # we use hidden_size * 4 units as we can chunk/split the output later
        self.linear = nn.Linear(input_size + hidden_size, hidden_size * 4, bias=True)

    def forward(self, x: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x: The input tensor with shape (batch_size, feature_dim)
            hx: The initial hidden state, optional. Is a two-tuple consisting of
                the current hidden state with shape (batch_size, hidden_size)
                the previous cell state (C_{t-1}) with shape (batch_size, hidden_size)

        Returns:
            Tuple of:
                A torch.Tensor of the new hidden state with shape (batch_size, hidden_size)
                A torch.Tensor of the current cell state (C_{t}) with shape (batch_size, hidden_size)
        """

        if hx is None:
            hx = self._init_hidden_state(x)
        hidden_state, previous_cell_state = hx

        # START TODO #############
        # compute the gating and cell update vectors. use torch cat to merge input and hidden state
        # split the output into the four predictions (candidate_cell_state, forget_gate, input_gate, output_gate).
        # use torch chunk to split tensor
        # update the internal and hidden state
        # we intend you to not use any activation functions in this part of the code, but rather in the update function
        combined = torch.cat((x, hidden_state), dim=1)
        gates = self.linear(combined)
        candidate, forget, input_g, output = torch.chunk(gates, chunks=4, dim=1)

        current_cell_state = self.update_internal_state(forget, previous_cell_state, input_g, candidate)
        new_hidden_state = self.update_hidden_state(current_cell_state, output)
        # END TODO #############
        return new_hidden_state, current_cell_state

    def update_internal_state(self, forget_gate: torch.Tensor, previous_cell_state: torch.Tensor,
                              input_gate: torch.Tensor, candidate_cell_state: torch.Tensor) -> torch.Tensor:
        """
        Update the internal state based on the equation given in the slides.

        Notes:
            Here we follow the updating rules in the lecture slides, which is different from DL book.

        Args:
            forget_gate: A torch.Tensor which becomes forget gate (f_t) after passing through activation
                function with shape (batch_size, hidden_size)
            previous_cell_state: A torch.Tensor of the previous cell state (C_{t-1}) with
                shape (batch_size, hidden_size)
            input_gate: A torch.Tensor which becomes external input gate tensor (i_t)
                after passing through activation function with shape (batch_size, hidden_size)
            candidate_cell_state: A torch.Tensor which becomes candidate cell state (\tilde{C}_t) after
                passing through activation function with shape (batch_size, hidden_size)

        Returns:
            A torch.Tensor of the current cell state with shape (batch_size, hidden_size)
        """
        # START TODO #############
        # calculate the new internal state, applying the activation functions to the respective tensors
        forget_gate = torch.sigmoid(forget_gate)
        input_gate = torch.sigmoid(input_gate)
        candidate_cell_state = torch.tanh(candidate_cell_state)
        current_cell_state = forget_gate * previous_cell_state + input_gate * candidate_cell_state
        # END TODO #############
        return current_cell_state

    def update_hidden_state(self, current_cell_state: torch.Tensor,
                            output_gate: torch.Tensor) -> torch.Tensor:
        """
        Update the hidden state based on the equation given in the slides.

        Args:
            current_cell_state: A torch.Tensor of the current cell state (C_{t})
                with shape (batch_size, hidden_size)
            output_gate: A torch.Tensor which becomes output gate tensor (o_t)
                after passing through activation function with shape (batch_size, hidden_size)

        Returns:
            A torch.Tensor of the new hidden state (h_t) with shape (batch_size, hidden_size)
        """
        # START TODO #############
        output_gate = torch.sigmoid(output_gate)

        new_hidden_state = output_gate * torch.tanh(current_cell_state)
        # END TODO #############
        return new_hidden_state

    def _init_hidden_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the hidden state with zeros.

        Returns:
            A two-tuple (initial_hidden_state with shape (batch_size, hidden_size),
                initial_cell_state with shape (batch_size, hidden_size)).
        """

        # START TODO #############
        batch_size = x.shape[0]
        initial_hidden_state = torch.zeros(batch_size, self.hidden_size, device=x.device)
        initial_cell_state = torch.zeros(batch_size, self.hidden_size, device=x.device)
        # END TODO #############
        return initial_hidden_state, initial_cell_state


class LSTM(nn.Module):
    """
    Convenience class that automatically iterates over the sequence.

    Args:
        input_size: Input dimension.
        hidden_size: Hidden dimension.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm_cell = LSTMCell(input_size, hidden_size)

    def forward(self, x: Union[np.ndarray, torch.Tensor], hx=None) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over the sequence and apply the LSTM cell.

        Args:
            x: The input tensor with shape (batch, seq_len, input_size)
            hx: The initial hidden state, optional. Is a two-tuple consisting of
                the current hidden state and the internal cell state. Both have
                shape (batch_size, hidden_size). If None, set to zero.

        Returns:
            Tuple of:
                output_stacked_hidden, the stacked output of all LSTMCells with shape (batch, seq_len, hidden_size)
                    (excluding the cell state!)
                Tuple of:
                    last_hidden_state with shape (batch_size, hidden_size)
                    last_new_internal_state with shape (batch_size, hidden_size)
        """

        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x, requires_grad=True)

        # START TODO #############
        batch_size, seq_len, _ = x.shape
        if hx is None:

            hx = (
                torch.zeros(batch_size, self.lstm_cell.hidden_size, device=x.device),
                torch.zeros(batch_size, self.lstm_cell.hidden_size, device=x.device),
            )

        hidden_state, cell_state = hx

        out_h_states = []

        for i in range(seq_len):
            x_i = x[:, i, :]

            hidden_state, cell_state = self.lstm_cell(x_i, (hidden_state, cell_state))

            out_h_states.append(hidden_state)

        stacked_hid = torch.stack(out_h_states, dim=1)

        finals = (hidden_state, cell_state)

        return stacked_hid, finals
        # END TODO #############


class NoiseRemovalModel(nn.Module):
    """
    Model which uses LSTMs to remove noise from a noisy signal.

    Args:
        hidden_size: The number of units of the LSTM hidden state size.
        shift: The number of steps the RNN is run before its output is considered ("many-to-many shifted to the right").
    """

    def __init__(self, hidden_size: int, shift: int = 10):
        super().__init__()
        self.shift = shift
        # START TODO #############
        # Create the 2 LSTM and 1 Linear module as described in the assignment.
        self.lstm1 = LSTM(input_size=1, hidden_size=hidden_size)
        self.lstm2 = LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        # END TODO #############

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of noise removal.

        This function
        1) pads the input sequence with self.shift zeros at the end,
        2) perform forward passing of the first LSTM
        3) cuts the first self.shift outputs
        4) perform forward passing of the second LSTM
        5) applies Linear layer.

        Args:
            x: The input tensor with shape (batch_size, sequence length, 1)

        Returns:
            A torch.Tensor of shape (batch_size, sequence length, 1)
        """

        # Pad input sequence x at the end (shifted many-to-many model).
        # This allows the model to see a few numbers before it has to guess
        # the noiseless output.

        # START TODO #############
        batch_size, seq_length, _ = x.shape

        pad = torch.zeros((batch_size, self.shift, 1), device=x.device)
        pad_x = torch.cat((x, pad), dim=1)

        lstm1_o, _ = self.lstm1(pad_x)

        lstm1_shifted = lstm1_o[:, self.shift:, :]

        lstm2_o, _ = self.lstm2(lstm1_shifted)

        return self.linear(lstm2_o)
        # END TODO #############
