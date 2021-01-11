import numpy as np
import tensorflow as tf
tfkl = tf.keras.layers

def array2tensor(z, dtype=tf.float32):
  """Converts numpy arrays into tensorflow tensors.
  
  Keyword arguments:
  z -- numpy array
  dtype -- data type of tensor entries (default float32)
  """
  if len(np.shape(z)) == 1:  # special case where input is a vector
    return tf.cast(np.reshape(z, (np.shape(z)[0], 1)), dtype)
  else:
    return tf.cast(z, dtype)


def reduce_logmeanexp_offdiag(x, axis=None):
  """Contracts the tensor x on its off-diagonal elements and takes the logarithm.

  Keyword arguments:
  x -- tensorflow tensor
  axis (int) -- contraction axis (default None)
  if axis=None, does full contraction 

  :Authors:
    Ben Poole
    Copyright 2019 Google LLC.
  """

  num_samples = x.shape[0].value
  if axis:
    log_num_elem = tf.math.log(num_samples - 1)
  else:
    log_num_elem = tf.math.log(num_samples * (num_samples - 1))  
  return tf.reduce_logsumexp(x - tf.linalg.tensor_diag(np.inf  * tf.ones(num_samples)), axis=axis)\
         - log_num_elem

def const_fn(x, const=1.0):
  """Function mapping any argument to a constant float value.
  
  Keyword arguments:
  x -- dummy argument
  const (float) -- constant value of the image
  """
  return const
  

def mlp(hidden_dim, output_dim, layers, activation):
  """Constructs multi-layer perceptron (MLP) critic with given number of hidden layers.

  Keyword arguments:
  hidden_dim (int) -- dimensionality of hidden dense layers
  output_dim (int) -- dimensionality of the output tensor
  layers (int) -- number of hidden dense layers
  activation -- activation function of the neurons
  """

  return tf.keras.Sequential(
      [tfkl.Dense(hidden_dim, activation) for _ in range(layers)] +
      [tfkl.Dense(output_dim)])
