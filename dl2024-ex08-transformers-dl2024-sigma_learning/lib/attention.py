from typing import Tuple

import numpy as np
import torch


def attention_function(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dk: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate attention function given in Vaswani et al. 2017 but with sigmoid instead of the softmax.

    Args:
        q: The query tensor with shape (batch_size, vocab_size, dk)
        k: The key tensor with shape (batch_size, sequence_length, dk)
        v: The value tensor with shape (batch_size, sequence_length, dk)
        dk: The embedding size

    Returns:
        Tuple of:
            A torch.Tensor of the attention with shape (batch_size, vocab_size, output)
            A torch.Tensor of the attention weights with shape (batch_size, vocab_size, sequence_length)
    """
    # Calculate the scaled dot product attention and attention weights
    # Use torch.bmm for batch matrix multiplication.
    # START TODO #############
    # Note: For this specific task we deviate a little from the original attention function
    # given in Vaswani et al. 2017 and we apply sigmoid instead of the softmax operation
    k_t = k.transpose(1, 2)
    sqrt_dk = torch.sqrt(torch.tensor(dk, dtype=torch.float32))
    scores = torch.bmm(q, k_t) / sqrt_dk
    attention_weights = torch.sigmoid(scores)
    attention = torch.bmm(attention_weights, v)
    # END TODO #############
    return attention, attention_weights
