# Authors: Doruk Efe Gökmen
# Date: 03/02/2025

import torch
import torch.nn as nn
import numpy as np


class MultiDense(nn.Module):
    """
    Fully connected (or dense) layer that accepts an input
    tensor of general shape.

    Args:
        hidden_dim (int): dimensionality of output tensor

    Attributes:
        kernel (torch.nn.Parameter): weights of the layer
        bias (torch.nn.Parameter): bias of the layer

    """

    def __init__(self, hidden_dim: int):
        super(MultiDense, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel = None
        self.bias = None

    def forward(self, x):
        if self.kernel is None:
            input_shape = x.shape
            rank = len(input_shape)
            kernel_shape = list(input_shape[1:]) + [self.hidden_dim]
            self.kernel = nn.Parameter(torch.randn(*kernel_shape))
            self.bias = nn.Parameter(torch.zeros(self.hidden_dim))

        dims_x = list(range(1, len(x.shape)))  # all input dims except for the batch dim
        dims_kernel = list(
            range(len(self.kernel.shape) - 1)
        )  # all kernel dims except for the hidden dim
        out = torch.tensordot(x, self.kernel, dims=(dims_x, dims_kernel))
        return out + self.bias


def mlp(
    hidden_dim: int,
    output_dim: int,
    layers: int,
    activation,
    use_dropout: bool = False,
    dropout_rate: float = 0.2,
):
    """Constructs a multi-layer perceptron (MLP) with given number of hidden layers.

    Args:
        hidden_dim (int): dimension of hidden dense layers
        output_dim (int): dimension of the output tensor
        layers (int): number of hidden dense layers
        activation (torch.nn.Module): activation function of the neurons
        use_dropout (bool, optional): dropout after hidden layers Defaults to False.
        dropout_rate (float, optional): Defaults to 0.2.

    Returns:
       The MLP network (torch.nn.Sequential)
    """
    modules = []
    for _ in range(layers):
        modules.append(nn.Linear(hidden_dim if modules else hidden_dim, hidden_dim))
        modules.append(activation)
        if use_dropout:
            modules.append(nn.Dropout(dropout_rate))
    modules.append(nn.Linear(hidden_dim if layers > 0 else hidden_dim, output_dim))
    return nn.Sequential(*modules)


def multi_mlp(
    hidden_dim: int,
    output_dim: int,
    layers: int,
    activation=torch.nn.ReLU,
    input_shape=None,
    use_dropout: bool = False,
    dropout_rate: float = 0.2,
):
    """Constructs an extended multi-layer perceptron (MLP) critic
    with given number of hidden layers with tensor inputs.

    Args:
        hidden_dim (int): dimension of hidden dense layers
        output_dim (int): dimension of the output tensor
        layers (int): number of hidden dense layers
        activation (torch.nn.Module): activation function of the neurons
        input_shape (tuple, optional): shape of the input tensor. Defaults to None.
        use_dropout (bool, optional): add dropout after hidden layers. Defaults to False.
        dropout_rate (float, optional): Defaults to 0.2.

    Returns:
        The multi-MLP network (torch.nn.Sequential)
    """

    model_seq = []

    if input_shape is not None:
        model_seq.append(MultiDense(hidden_dim))
        if activation is not None:
            model_seq.append(activation())
        layers -= 1

    if use_dropout:
        for _ in range(layers):
            model_seq.append(nn.Linear(hidden_dim, hidden_dim))
            if activation is not None:
                model_seq.append(activation())
            model_seq.append(nn.Dropout(dropout_rate))
    else:
        for _ in range(layers):
            model_seq.append(nn.Linear(hidden_dim, hidden_dim))
            if activation is not None:
                model_seq.append(activation())

    model_seq.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*model_seq)


def logmeanexp_offdiag(x, axis=None):
    """
    Contracts the tensor x on its off-diagonal elements and takes the logarithm.

    Args:
        x (torch.Tensor): input tensor
        axis (int, optional): axis to contract the tensor (default None)
            Note: if None, the tensor is contracted on all axes.

    Based on code by Ben Poole. Copyright 2019 Google LLC.
    """

    num_samples = x.size(0)
    if axis is not None:
        log_num_elem = torch.log(
            torch.tensor(num_samples - 1, dtype=x.dtype, device=x.device)
        )
    else:
        log_num_elem = torch.log(
            torch.tensor(
                num_samples * (num_samples - 1), dtype=x.dtype, device=x.device
            )
        )

    inf_diag = torch.full((num_samples,), float("inf"), dtype=x.dtype, device=x.device)
    x_no_diag = x - torch.diag(inf_diag)

    if axis is None:
        return torch.logsumexp(x_no_diag, dim=(0, 1)) - log_num_elem
    else:
        return torch.logsumexp(x_no_diag, dim=axis) - log_num_elem


def logmeanexp_masked(x, mask, axis=None):
    """_summary_

    Args:
        x (_type_): _description_
        mask (_type_): _description_
        axis (_type_, optional): _description_. Defaults to None.
    """

    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=x.device)

    masked_input = torch.where(
        mask, x, torch.tensor(-np.inf, dtype=x.dtype, device=x.device)
    )

    log_n = torch.log(torch.sum(mask_tensor.to(masked_input.dtype), dim=axis))
    
    return torch.logsumexp(masked_input, dim=axis) - log_n


def array2tensor(z, dtype=torch.float32):
    """Converts numpy arrays into torch tensors.

    Args:
        z (numpy array): input numpy array
        dtype (torch.dtype): data type of tensor entries (default float32)

    Returns:
        torch.Tensor: converted tensor
    """
    if z.ndim == 1:  # special case where input is a vector
        z = z.reshape(z.shape[0], 1)
    return torch.tensor(z, dtype=dtype)


def const_fn(x, const=1.0):
    """Function mapping any argument to a constant float value.

    Keyword arguments:
    x -- dummy argument
    const (float) -- constant value of the image
    """
    return const
