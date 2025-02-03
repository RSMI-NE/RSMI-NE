# Authors: Doruk Efe Gökmen
# Date: 03/02/2025

# NOTE: filename used to be `critics.py` but was changed to `critics_baselines.py`
# TODO: implement baseline class for TUBA bound

import torch
import torch.nn as nn
from rsmine.mi_estimator.utils import multi_mlp

class SeparableCritic(nn.Module):
    """Separable ansatz (critic) for computing a lower-bound of mutual information I(X:Y).

    Attributes: 
    _g (_h) -- MLP ansatz for X (Y) variable

    Methods:
    forward(x, y) -- calls the ansatz as a function for samples x, y
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        layers: int,
        activation,
        input_shapes: list = [None, None],
        use_dropout: bool = False,
        dropout_rate: float = 0.2,
        **extra_kwargs
    ):
        super(SeparableCritic, self).__init__()
        self._g = multi_mlp(
            hidden_dim,
            embed_dim,
            layers,
            activation,
            input_shape=input_shapes[1],
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )
        self._h = multi_mlp(
            hidden_dim,
            embed_dim,
            layers,
            activation,
            input_shape=input_shapes[0],
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )

    def forward(self, x, y):
        """Constructs unnormalized likelihood matrix (aka the scores matrix)
        from the two separate MLPs for x and y data.
        
        Outer product is taken along the batch dimension,
        and inner product is taken along the output dimension of the MLPs.
        """
        
        return torch.einsum("ij,kj->ik", [self._h(y), self._g(x)])


class ConcatenatedCritic(nn.Module):
    """Concatenated ansatz (critic) for computing a lower-bound of mutual information I(X:Y).

    Attributes: 
    _f -- MLP ansatz for concatenated X and Y variables

    Methods:
    forward(x, y) -- calls the ansatz as a function for samples x, y
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        layers: int,
        activation,
        input_shapes: list = [None, None],
        use_dropout: bool = False,
        dropout_rate: float = 0.2,
        **extra_kwargs
    ):
        super(ConcatenatedCritic, self).__init__()
        self._f = multi_mlp(
            hidden_dim,
            embed_dim,
            layers,
            activation,
            input_shape=input_shapes[0] + input_shapes[1],
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )

    
    def forward(self, x, y):
        """Constructs unnormalized likelihood matrix (aka the scores matrix)
        from the concatenated MLP for x and y data.
        """
        
        batch_dim = x.shape[0]
        x_expanded = x.unsqueeze(1).expand(batch_dim, batch_dim, -1)
        y_expanded = y.unsqueeze(0).expand(batch_dim, batch_dim, -1)
        
        x_y = torch.cat([x_expanded, y_expanded], dim=-1)
        x_y.reshape(batch_dim**2, -1)
        
        flat_scores = self._f(x_y)
        scores = flat_scores.reshape(batch_dim, batch_dim)
        return scores
