"""Constructs separable critic for VBMI calculation
[Based on arXiv:1905.06922v1 Poole et al. (2019)]

Author: Ben Poole, Doruk Efe Gökmen
Date: 10/01/2021
"""

import tensorflow as tf
from utils import mlp

class SeparableCritic(tf.keras.Model):
  """Separable ansatz (critic) for MI bound.
  
  Attributes: 
  _g (_h) -- MLP ansatz for X (_Y) variable

  Methods:
  call(x, y) -- calls the ansatz as a function for samples x, y
  """

  def __init__(self, hidden_dim: int, embed_dim: int, layers: int, 
              activation,use_dropout: bool=False, dropout_rate: float=0.2, 
              **extra_kwargs):
    """Constructs two separate MLP critics of same structure for x and y data.

    Keyword arguments:
    hidden_dim (int) -- dimensionality of hidden dense layers
    embed_dim (int) -- dimension of contracted output of the dense nets _g and _h
    layers (int) -- number of hidden layers in dense nets _g and _h
    activation -- activation function of the neurons
    use_dropout (bool) -- add dropout after hidden layers
    dropout_rate (float)

    :Authors: 
      Ben Poole, Doruk Efe Gökmen, Copyright 2019 Google LLC.
    """

    super(SeparableCritic, self).__init__()
    self._g = mlp(hidden_dim, embed_dim, layers, activation, 
                  use_dropout=use_dropout, dropout_rate=dropout_rate) 
    self._h = mlp(hidden_dim, embed_dim, layers, activation,
                  use_dropout=use_dropout, dropout_rate=dropout_rate)

  def call(self, x, y):
    """Constructs unnormalised likelihood matrix (or scores) 
    from the two separate MLP's for x and y data.

    Keyword arguments:
    x -- a sample for the random variable X
    y -- a sample for the random variable Y
    """

    return tf.einsum('ij,kj->ik', self._h(y), self._g(x))
