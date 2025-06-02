from typing import Tuple, List

import numpy as np

from lib.network_base import Module, Parameter


def calculate_conv_output(input_shape: List[int], kernel_size: List[int],
                          stride: List[int], padding: List[int]) -> List[int]:
    """Calculate the output shape of a convolution given the input shape and the convolution parameters.
    The input and output lists will all be of the same length N for an N-dimensional convolution.

    Args:
        input_shape: Input dimension sizes
        kernel_size: Kernel size per dimension
        stride: Stride per dimension
        padding: Padding per dimension

    Returns:
        Convolution output shape
    """
    assert len(input_shape) == len(kernel_size) == len(stride) == len(padding),\
        "All inputs need to be of the same length."

    # START TODO ################
    output = []
    for i in range(len(input_shape)):
        output_size = (input_shape[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1
        output.append(output_size)
    # END TODO ################

    return output


class Conv2d(Module):
    """2D convolution module (cross correlation to be precise).

    Args:
            in_channels: Number of incoming channels.
            out_channels: Number of output channels after the convolution.
            kernel_size: Size of the kernel.
            stride: Stride while applying convolution.
            padding: Padding to apply for convolution.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int, int], stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0)):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding
        w = np.random.randn(out_channels, in_channels, *kernel_size) * 0.1
        self.W = Parameter(w, name="conv_filter")
        self.b = Parameter(np.zeros((out_channels, 1, 1)), name="conv_bias")

    def __repr__(self) -> str:
        """Return string representation of the module.

        Returns:
            String representation
        """
        return ("Conv2d(in={},out={},"
                "kernel={},stride={},pad={})"
                .format(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.pad))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the module.

        Args:
            x: input shaped (batch_size, channels, height, width)

        Returns:
            output (batch_size, output_channels, output_height, output_width)
        """
        assert len(x.shape) == 4
        feature_map = np.empty(self._compute_output_shape(x.shape))

        # START TODO ################
        pad_height, pad_width = self.pad
        x_padded = np.pad(x,
                          ((0, 0),
                           (0, 0),
                           (pad_height, pad_height),
                           (pad_width, pad_width)),
                          mode='constant', constant_values=0)

        self.input_cache = x_padded

        output_shape = self._compute_output_shape(x.shape)

        batch_size, out_channels, output_height, output_width = output_shape

        for i in range(output_height):
            i_s = i * self.stride[0]
            i_e = i_s + self.kernel_size[0]
            for j in range(output_width):
                j_s = j * self.stride[1]
                j_e = j_s + self.kernel_size[1]

                sliced_x = x_padded[:, :, i_s: i_e, j_s: j_e]

                feature_map[:, :, i, j] = (np.tensordot(sliced_x, self.W.data, axes=([1, 2, 3], [1, 2, 3]))
                                           + self.b.data[:, 0, 0])
        return feature_map
        # We recommend you go about completing this part of the assignment iteratively.
        # First, consider the case without padding and a stride of 1.
        # Once you've cracked that, move on to the case with different strides and padding.
        # Remember to consider the bias also (the exercises in convolution_warmup.py do not consider the bias)
        # Lastly, don't forget to store the padded input in self.input_cache so that it may be used in the
        # backward pass

        # END TODO ################

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the module.

        Args:
            grad: Gradient of the module after this module.

        Returns:
            This module's gradient.
        """
        x = self.input_cache

        grad_wrt_x = np.zeros_like(x)
        k0, k1 = self.kernel_size
        W = np.expand_dims(self.W.data, 0)
        for i in range(grad.shape[-2]):
            i_s = i * self.stride[0]  # strided i
            for j in range(grad.shape[-1]):
                grad_slice = grad[..., i, j, None, None, None]
                j_s = j * self.stride[1]  # strided j
                h_slice = slice(i_s, i_s + k0)
                w_slice = slice(j_s, j_s + k1)
                grad_wrt_x[..., h_slice, w_slice] += np.sum(grad_slice * W, 1)
                grad_wrt_w = grad_slice * x[..., None, :, h_slice, w_slice]
                self.W.grad += np.sum(grad_wrt_w, axis=0)  # sum over batch axis
        self.b.grad += np.sum(grad, axis=(0, 2, 3))[..., None, None]
        # undo padding
        # `-ph or None` is only None when p_h is 0, then we don't need slicing
        p_h, p_w = self.pad
        return grad_wrt_x[..., p_h:-p_h or None, p_w:-p_w or None]

    def parameters(self) -> List[Parameter]:
        """Return the module parameters.

        Returns:
            List of module Parameters
        """
        return self.W, self.b

    def _compute_output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Compute this module's output shape.

        Args:
            input_shape: Input dimension sizes

        Returns:
            Convolution output shape
        """
        [out_h, out_w] = calculate_conv_output(input_shape[2:], self.kernel_size, self.stride, self.pad)
        # We never silently pad! Note that usually libraries silently pad for uneven output sizes.
        assert out_h % 1 == 0., "invalid combination of conv parameters"
        assert out_w % 1 == 0., "invalid combination of conv parameters"
        batch_size = input_shape[0]
        return batch_size, self.out_channels, int(out_h), int(out_w)


class Flatten(Module):
    """ Flatten feature dimension to (batch_size, feature_dim).

        Note: This layer allows us to create the model using `Sequential` only.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        old_shape = self.input_cache
        return grad.reshape(old_shape)
