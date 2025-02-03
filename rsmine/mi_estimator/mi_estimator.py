# NOTE: filename used to be `MI_estimators.py` but was changed to `mi_estimator.py`

import torch
from rsmine.mi_estimator.training import train_estimator


class VBMI:
    """Interface for mutual information estimation by maximising variational lower bounds.

    Author: Doruk Efe Gökmen
    Date: 03/02/2025
    """

    def __init__(
        self,
        batch_size,
        input_shapes: list = [None, None],
        layers: int = 2,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        activation: torch.nn.Module = torch.nn.ReLU,
        iterations: int = 600,
        shuffle: int = 1,
        learning_rate: float = 5e-3,
        bound: str = "infonce",
        use_dropout: bool = False,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        """
        Mutual information estimation by maximising variational lower bounds
        represented by neural network ansätze.

        Attributes:
            batch_size (int) -- total number of samples in a batch
            layers (int) -- number of hidden layers of the MLPs (default 2)
            embed_dim (int) -- embedding dimension of the separable ansatz (default 16)
            hidden_dim (int) -- hidden dimension of the dense layers in the MLP (default 64)
            activation (torch.nn.Module) -- activation function of the neurons (default ReLU)
            iterations (int) -- number of epochs for training (default 600)
            shuffle (int) -- size of shuffled blocks of sample data (default 1)
            learning_rate (float) -- learning rate for the optimiser (default 5e-3)
            bound (str) -- type of mutual information lower-bound for estimation (default 'infonce')
            use_dropout (bool) -- whether to use dropout in the network (default False)
            dropout_rate (float) -- dropout rate if dropout is used (default 0.2)
        """
        
        self.batch_size = batch_size
        self.iterations = iterations
        self.bound = bound

        self.critic_params = {
            "input_shapes": input_shapes,
            "layers": layers,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "activation": activation,
            "use_dropout": use_dropout,
            "dropout_rate": dropout_rate,
        }

        self.opt_params = {
            "batch_size": batch_size,
            "iterations": iterations,
            "shuffle": shuffle,
            "learning_rate": learning_rate,
        }

    def MI(self, x, y):
        """Returns the estimate for I(X:Y).

        Keyword arguments:
        x, y -- sample datasets (torch.Tensor or np.ndarray)
        """
        return train_estimator(x, y, self.critic_params, self.opt_params, self.bound)
