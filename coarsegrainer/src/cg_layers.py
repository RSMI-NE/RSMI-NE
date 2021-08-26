"""Definition of the coarse-graining
and embedding network layers for the RSMI-NE.

Classes: 
Conv2DSingle -- Convolves V with Λ for 1- and 2-d systems
Conv3DSingle -- Convolves V with Λ for 3-d systems
ConvGraphSingle -- Convolves V with Λ for systems defined on a (networkx) graph
CoarseGrainer -- Stacks convolution and embedding layers and maps V to H

Author: Doruk Efe Gökmen, Maciej Koch-Janusz
Date: 26/08/2021
"""


import numpy as np  # used for exponential moving average
import tensorflow as tf
#from tensorflow.keras import datasets, models, regularizers, backend  
#from tensorflow.python.framework import ops

import tensorflow_probability as tfp
tfd = tfp.distributions
tfkl = tf.keras.layers
tfp = tfp.layers


class Conv2DSingle(tfkl.Layer):
  """Custom convolution layer to produce the (stochastic) coarse-graining map
  from 2(or 1)-d visible degrees of freedom.
  """

  def __init__(self, hidden_dim: int, visible_dim: int = 1, input_shape=(2, 2), init_rule=None):
    """Constructs the convolutional net.
    
    Attributes:
    ws -- weights of the kernel

    Methods: 
    call() -- call the network as a function
    """

    super(Conv2DSingle, self).__init__()

    if isinstance(init_rule, np.ndarray):
      init = init_rule
    else:
      w_init = tf.random_normal_initializer()
      init = w_init(shape=input_shape+(visible_dim,) +
                    (hidden_dim,), dtype='float32')

    self.ws = tf.Variable(initial_value=init, trainable=True)

  def call(self, inputs):
    """Computes the dot product between the input and kernel weights.
    Currently it does not mix the entries in the one-hot encoding dimension.

    The indices represent the following
    :t: sample number
    :ij: 2D spatial location in the configuration
    :a: component of the original degrees of freedom (visible_dim)
    :b: component of the coarse-grained degrees of freedom (hidden_dim)
    :d: one-hot encoding direction

    TODO: Debug the one-hot encoding.
    TODO: Debug handling of multi-component degrees of freedom.
    Might need to make changes in build_dataset.py and cg_optimisers.py!
    
    Keyword arguments:
    inputs -- tensor encoding the visible block (V) to be coarse-grained
    """

    return tf.einsum('tijad,ijab->tbd', inputs, self.ws)


class Conv3DSingle(tfkl.Layer):
  """Custom convolution layer to produce the (stochastic) 
  coarse-graining map for 3-d systems.

  TODO: Handle multicomponent original degrees of freedom.
  """

  def __init__(self, hidden_dim: int, input_shape: tuple=(2, 2, 2), init_rule=None):
    """Constructs the convolutional net.
    
    Attributes:
    ws -- weights of the kernel

    Methods: 
    call() -- call the network as a function
    """

    super(Conv3DSingle, self).__init__()
    
    if isinstance(init_rule, np.ndarray):
      init = init_rule
    
    else:
        w_init = tf.random_normal_initializer()
        init = w_init(shape=input_shape+(hidden_dim,), dtype='float32')
        
    self.ws = tf.Variable(initial_value=init, trainable=True)

  def call(self, inputs):
    """Computes the dot product between the input and kernel weights.

    Keyword arguments:
    inputs -- tensor encoding the visible block to be coarse-grained
    """

    return tf.einsum('tijkl,ijks->tsl', inputs, self.ws)

    
class ConvGraphSingle(tfkl.Layer):
  """Custom convolution layer to produce the (stochastic) coarse-graining map
  from visible degrees of freedom on a (netoworkx) graph. The difference to
  Conv2DSingle is that ll in *not* a tuple, but an int defining the radius of V, 
  all the configurations are otherwise one-dimensional. 
  The input_shape is (#edges in V,)
  """

  def __init__(self, hidden_dim: int, input_shape: tuple=(2,), init_rule=None):
    """Constructs the convolutional net.
    
    Attributes:
    ws -- weights of the kernel

    Methods: 
    call() -- call the network as a function
    """

    super(ConvGraphSingle, self).__init__()

    if isinstance(init_rule, np.ndarray):
      init = init_rule
    else:
      w_init = tf.random_normal_initializer()
      init = w_init(shape=input_shape+(hidden_dim,), dtype='float32')
      
    self.ws = tf.Variable(initial_value=init, trainable=True)

  def call(self, inputs):
    """Computes the dot product between the input and kernel weights.

    Keyword arguments:
    inputs -- tensor encoding the visible block to be coarse-grained
    """

    return tf.einsum('tik,is->tsk', inputs, self.ws)


class CoarseGrainer(tf.keras.Model):
  def __init__(self, ll: tuple=None, size_V: int=None, num_hiddens: int=1, 
              conv_activation='tanh', Nq=None, h_embed: bool=False, 
              init_rule=None, relaxation_rate: float=0.01, 
              min_temperature: float=0.05, init_temperature: float=2, 
              use_logits: bool=True, use_probs: bool=False, **extra_kwargs):
    """Stacked network representing the variational ansatz that
    generates the coarse-grained degrees of freedom H from V.

    Note that for the cases where Nq is None, the output variable is flattened.

    Attributes:
    ll (tuple of ints) -- shape of the visible block V, for regular lattices !!!
        !!! for graphs ll (int) is the topological radius around center of V and
    size_V (int) -- is the number of edges in V defined by the radius ll
    num_hiddens (int) -- number of components of the coarse-grained variable H
    conv_activation (str) -- (nonlinear) activation function to map H (default tanh)
    Nq (int) -- number of states for a Potts degree of freedom (default None)
    h_embed (bool) -- embed H into a (pseudo) discrete valued variable (default False)
    init_rule -- initial conditions for the weights of the convolution net
    relaxation_rate (float) -- Gumbel-softmax rate for exponential annealing schedule
    min_temperature (float) -- minimum value for the Gumbel-softmax relaxation parameter
    init_temperature (float) -- initial value for the Gumbel-softmax relaxation parameter
    use_logits (bool) -- switch for treating the convolved values as logits
    use_probs (bool) -- switch for treating the convolved values as probabilities

    Functions and methods:
    call() -- call function: V -> H
    global_step() -- updates the iteration index locally
    tau() -- anneals the Gumbel-softmax temperature parameter using the global iteration step
    """

    super(CoarseGrainer, self).__init__()

    self.Nq = Nq

    self._global_step = 0  # intialise the global iteration step in training
    
    if isinstance(size_V, int):
      self.coarse_grainer = ConvGraphSingle(num_hiddens, (size_V,), init_rule)
    elif len(ll) == 2: # i.e. if dimensionality (d) is 2
      self.coarse_grainer = Conv2DSingle(num_hiddens, ll, init_rule)
    elif len(ll) == 3: # if d=3
      self.coarse_grainer = Conv3DSingle(num_hiddens, ll, init_rule)
    
      
    if h_embed:  
      # sample pseudo-discrete coarse-grained variable using Gumbel-softmax trick
      self.method = 'pseudo-categorical sampling'
      self.r = relaxation_rate
      self.min_tau = min_temperature
      self.init_tau = init_temperature

      if self.Nq == None: # if alphabet size for dof. is unspecified, assume binary variable
        if use_probs:
          # This old version leads to arithmetic underflow in the log.
          # self.embedder = tfkl.Lambda( 
          #    lambda x: tfd.RelaxedBernoulli(self.tau, probs=x).sample())

          # stack the convolution, activation and embedding layers:
          #self._Λ = tf.keras.Sequential([self.coarse_grainer,
          #                               tfkl.Activation(tf.nn.sigmoid),
          #                               tfkl.Flatten(),
          #                               self.embedder])

          self.embedder = tfkl.Lambda(
              lambda x: tfd.RelaxedBernoulli(self.tau, 
                                    logits=tf.nn.log_softmax(x)).sample())

          # stack the convolution, activation and embedding layers
          self._Λ = tf.keras.Sequential([self.coarse_grainer, 
                                         tfkl.Flatten(), 
                                         self.embedder])

        elif use_logits:
          self.embedder = tfkl.Lambda(
              lambda x: tfd.RelaxedBernoulli(self.tau, logits=x).sample())
          # stack the convolution and embedding layers
          self._Λ = tf.keras.Sequential([self.coarse_grainer, 
                                         tfkl.Flatten(), 
                                         self.embedder])
        
      else: # Sample Nq-valued discrete (categorical) variables using CNN kernel.
        # In fact, we use Nq - 1 as the number of possible states of the discrete variable
        # since the Nq'th state is redundant.

        # TODO: flattening the convolutional output messes up the one-hot encoding direction.
        # We actually should not flatten the output, but instead preserve the one-hot encoding!
        # But the current implementation takes flat vectors for MI estimation.
        # TODO: We should generalise it to address this.

        # We specify the one-hot encoding axis inside the softmax 
        # (i.e. axis=1) for multi-component coarse-grained variables.
        # TODO: Debug this.

        if use_probs:
          self.embedder = tfkl.Lambda(lambda x: tfd.RelaxedOneHotCategorical(
                                self.tau, logits=tf.nn.log_softmax(x, axis=1)).sample())
          # stack the convolution, activation and embedding layers
          self._Λ = tf.keras.Sequential([self.coarse_grainer, 
                                         self.embedder]) # TODO: squeeze the output?
        elif use_logits:
          self.embedder = tfkl.Lambda(lambda x: tfd.RelaxedOneHotCategorical(
                                      self.tau, logits=x).sample()) 
          # stack the convolution and embedding layers
          self._Λ = tf.keras.Sequential([self.coarse_grainer, 
                                         self.embedder])  # TODO: squeeze the output?

    else:
      # directly use the CNN output as the coarse-grained variable
      self.method = 'convolved variables'
      self._Λ = tf.keras.Sequential(
          [self.coarse_grainer, tfkl.Activation(conv_activation), tfkl.Flatten()])

  def call(self, V):
    """
    The coarse-grainer network is called by providing the 
    block degrees of freedom (V) as the input.

    Keyword arguments:
    V -- sample dataset for the visible block
    """

    return self._Λ(V)
      
  @property
  def global_step(self):  
    """Gets global step of iteration for annealing the GS temperature parameter tau"""

    return self._global_step

  @global_step.setter  
  def global_step(self, step):
    """Update step of iteration (with value "step") for annealing tau

    Arguments:
    step -- current (global) iteration step for training
    """

    self._global_step = np.float32(step)

  @property
  def tau(self):  
    """
    Anealing schedule for Gumbel-softmax temperature parameter (tau).
    Returns the updated value of tau according to current stage of training.
    """

    return np.float32(max(self.min_tau,
                          self.init_tau*np.exp(-self.r*self._global_step)))
