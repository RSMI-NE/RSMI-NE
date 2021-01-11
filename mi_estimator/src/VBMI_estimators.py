"""Mutual information estimation by maximising 
variational lower bounds via stochastic gradient descent
training of neural-network parameters.
Implemented in Tensorflow.

[Based on arXiv:1905.06922v1 Poole et al. (2019)]

Authors: Doruk Efe Gökmen
Date: 04/08/2020
"""

# pylint: disable-msg=E0611

#from tqdm import tqdm # use this for command-line execution
from tqdm.notebook import tqdm  # use this for notebook execution
import numpy as np
import pandas as pd
import tensorflow as tf

from VBMI_bounds import lowerbounds
from critics import SeparableCritic
from utils import mlp


def train_estimator(X, Y, critic_params, opt_params, bound='infonce'):
  """Main training loop to estimate MI.

  Keyword arguments: 
  X -- full dataset for the random variable X
  Y -- full dataset for the random variable Y
  critic_params (dict) -- set of parameters for the ansatz/critic function
  opt_param (dict) -- set of parameters for the optimiser
  bound (str) -- mutual information lower-bound (default InfoNCE)
  """

  f_ansatz = SeparableCritic(**critic_params)

  opt = tf.keras.optimizers.Adam(
      opt_params['learning_rate'])  # set optimiser as Adam

  # prepare the dataset using tf.data api
  num_samples = X.shape[0]

  dat = tf.data.Dataset.from_tensor_slices((X, Y))
  dat = dat.shuffle(opt_params['shuffle']).batch(
    opt_params['batch_size']).repeat(opt_params['iterations'])

  estimates = []

  pbar = tqdm(total=opt_params['iterations'] *
              num_samples//opt_params['batch_size'], desc = '')

  i = 0
  for y, x in dat:
    with tf.GradientTape() as tape:
      """
      Takes a sample of random variable pair (x,y)
      and computes the gradient of the mutual information
      lower bound with respect to variational parameters
      of the estimator (f) ansatz.
      """
      
      mi = lowerbounds[bound](x, y, f_ansatz)
      cost = -mi

      trainable_vars = []
      trainable_vars += f_ansatz.trainable_variables
      grads = tape.gradient(cost, trainable_vars)
      opt.apply_gradients(zip(grads, trainable_vars))
    estimates.append(mi.numpy())

    pbar.set_description(f'I={mi:.2f}')
    pbar.update(1)  # update progress bar for each iteration step
    i += 1

  return np.array(estimates)
