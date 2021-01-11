"""
Definition of the coarse-graining
and embedding network layers for the RSMI-NE.

Author: Doruk Efe Gökmen
Date: 21/07/2020
"""

# pylint: disable=import-error

import numpy as np  # used for exponential moving average
import tensorflow as tf
from tensorflow.keras import datasets, models, regularizers, backend  
from tensorflow.python.framework import ops

import tensorflow_probability as tfp
tfd = tfp.distributions
tfkl = tf.keras.layers
tfp = tfp.layers

def array2tensor(z, dtype=tf.float32):
    """Helper function to convert numpy arrays into tensorflow tensors."""
    if len(np.shape(z)) == 1:  # special case where input is a vector
        return tf.cast(np.reshape(z, (np.shape(z)[0], 1)), dtype)
    else:
        return tf.cast(z, dtype)


class Conv2DSingle(tfkl.Layer):
  """
  Custom convolution layer to produce 
  the (stochastic) coarse-graining map
  from 2(or 1)-d visible degrees of freedom.
  """
  def __init__(self, hidden_dim, input_shape=(2, 2), init_rule=None):
    super(Conv2DSingle, self).__init__()

    if isinstance(init_rule, np.ndarray):
      init = init_rule
    else:
      w_init = tf.random_normal_initializer()
      init = w_init(shape=input_shape+(hidden_dim,), dtype='float32')
      
    self.ws = tf.Variable(initial_value=init, trainable=True)

  def call(self, inputs):
    return tf.einsum('tijk,ijs->tsk', inputs, self.ws)


class Conv3DSingle(tfkl.Layer):
  """
  Custom convolution layer to produce 
  the (stochastic) coarse-graining map
  from 3-d visible degrees of freedom.
  """
  def __init__(self, hidden_dim, input_shape=(2, 2, 2)):
    super(Conv3DSingle, self).__init__()

    w_init = tf.random_normal_initializer()
    self.ws = tf.Variable(initial_value=w_init(shape=input_shape+(hidden_dim,),
                                               dtype='float32'), trainable=True)

  def call(self, inputs):
    return tf.einsum('tijkl,ijks->tsl', inputs, self.ws)


class CoarseGrainer(tf.keras.Model):
  def __init__(self, ll, num_hiddens, conv_activation='tanh', 
              Nq=None, STE=False, h_embed=False, init_rule=None, 
              relaxation_rate=0.01, min_temperature=0.05, init_temperature=2, 
              use_logits=True, use_probs=False, **extra_kwargs):
    """
    Network representing the variational ansatz
    producing the coarse-grained degrees of freedom.
    """
    super(CoarseGrainer, self).__init__()

    self.Nq = Nq

    self._global_step = 0  # intialise the global iteration step in training

    if len(ll) == 2: # i.e. if dimensionality (d) is 2
      self.coarse_grainer = Conv2DSingle(num_hiddens, ll, init_rule)
    elif len(ll) == 3: # if d=3
      self.coarse_grainer = Conv3DSingle(num_hiddens, ll)
      
    if h_embed:  
      # sample approx. discrete coarse-grained variable using Gumbel-softmax trick
      self.method = 'pseudo-categorical sampling'
      self.r = relaxation_rate
      self.min_tau = min_temperature
      self.init_tau = init_temperature

      if self.Nq == None: # if alphabet size for dof. is unspecified, assume binary variable
        if use_probs:
          self.embedder = tfkl.Lambda( #better call this embedder
              lambda x: tfd.RelaxedBernoulli(self.tau, probs=x).sample())
          # stack the convolution, activation and embedding layers
          self._Λ = tf.keras.Sequential([self.coarse_grainer, 
                                         tfkl.Activation(tf.nn.softmax), 
                                         tfkl.Flatten(), 
                                         self.embedder])
        elif use_logits:
          self.embedder = tfkl.Lambda(
              lambda x: tfd.RelaxedBernoulli(self.tau, logits=x).sample())
          # stack the convolution and embedding layers
          self._Λ = tf.keras.Sequential([self.coarse_grainer, 
                                         tfkl.Flatten(), 
                                         self.embedder])
        
      else: # sample Nq-valued discrete (categorical) variables using CNN kernel
        if use_probs:
          self.embedder = tfkl.Lambda(lambda x: tfd.RelaxedOneHotCategorical(
                                      self.tau, probs=x).sample())
          # stack the convolution, activation and embedding layers
          self._Λ = tf.keras.Sequential([self.coarse_grainer, 
                                         tfkl.Activation(tf.nn.softmax), 
                                         tfkl.Flatten(), 
                                         self.embedder])
        elif use_logits:
          self.embedder = tfkl.Lambda(lambda x: tfd.RelaxedOneHotCategorical(
                                      self.tau, logits=x).sample()) 
          # stack the convolution and embedding layers
          self._Λ = tf.keras.Sequential([self.coarse_grainer, 
                                         tfkl.Flatten(), 
                                         self.embedder])

    else:
      # directly use the CNN output as the coarse-grained variable
      self.method = 'convolved variables'
      self._Λ = tf.keras.Sequential(
          [self.coarse_grainer, tfkl.Activation(conv_activation), tfkl.Flatten()])

  def call(self, V):
    """
    The coarse-grainer network is called by providing the 
    block degrees of freedom (V) as the input.
    """
    return self._Λ(V)
      
  @property
  def global_step(self):  
    # get step of iteration for annealing the GS temperature parameter (tau)
    return self._global_step

  @global_step.setter  
  def global_step(self, step):
    # update step of iteration (with value "step") for annealing tau
    self._global_step = np.float32(step)

  @property
  def tau(self):  
    """
    Anealing schedule for Gumbel-softmax temperature parameter (tau).
    Returns the updated value of tau according to current stage of training.
    """
    return np.float32(max(self.min_tau,
                          self.init_tau*np.exp(-self.r*self._global_step)))
