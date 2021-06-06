"""Real-space mutual information (RSMI) maximisation with respect to
coarse-graining filters by maximising variational lower-bounds
of RSMI expressed by neural network ansätze.
Implemented in Tensorflow.

Functions:
RSMI_estimate() -- Evaluates the exponential moving average of the RSMI series.
train_RSMI_optimiser() -- Performs the training loop for maximising RSMI.

Author: Doruk Efe Gökmen
Date: 08/04/2021
"""

import os
import sys
import numpy as np 
import pandas as pd
from tqdm.autonotebook import tqdm  
import tensorflow as tf
tfkl = tf.keras.layers

import build_dataset as ds
from cg_layers import CoarseGrainer
abspath = os.path.abspath(__file__)
os.chdir(os.path.dirname(abspath))
sys.path.append(os.path.abspath(os.path.join(os.pardir,os.pardir,"mi_estimator","src")))
import VBMI_estimators 
from VBMI_bounds import lowerbounds
SeparableCritic = VBMI_estimators.SeparableCritic


def RSMI_estimate(mis, ema_span=5000):
  """Exponential moving average  with span ema_span for the series of mi estimates.

  Keyword arguments:
  mis -- time series of mutual information estimates
  ema_span (int) -- span for evaluating average with exponentally larger weigths at later times
  """

  return pd.Series(mis).ewm(span=ema_span).mean().tolist()[-1]


def train_RSMI_optimiser(CG_params, critic_params, opt_params, 
                         data_params, bound='infonce', coarse_grain=True, 
                         init_rule=None, optimizer=None, use_GPU=False, 
                         index=None, buffer_size=None, env_size=None,
                         load_data_from_generators=False,
                         load_data_from_disk=False, 
                         E=None, V=None, verbose=True, **kwargs):
  """Main training loop for maximisation of RSMI [I(H:E)] for coarse-graining optimisation.

  Keyword arguments:
  E (tensorflow array) -- sample dataset for the environment random variable E
    (needed if load_data_from_generators=load_data_from_disk=False)
  V (tensorflow array) -- sample dataset for the visible block V
    (needed if load_data_from_generators=load_data_from_disk=False)
  index (tuple) -- upper-left index of the visible block V 
    (needed if load_data_from_generators=True)
  buffer_size (int) -- width of the buffer
    (needed if load_data_from_generators=True)
  env_size (int) -- width of the environment region
    (needed if load_data_from_generators=True)
  critic params (dict) -- parameters for the ansatz function of the MI lower-bound
  opt_params (dict) -- parameters of the optimiser
  data_params (dict) -- parameters for the sample dataset and the physical system
  bound (str) -- MI lower-bound (default InfoNCE)
  coarse_grain (bool) -- switch for coarse-graining (default True)
  init_rule -- initialisation for the coarse-graining rule (or initial conditions of Λ) 
    (default None)
  optimizer -- choice for stochastic gradient descent optimiser (default None: Adam)
  use_GPU (bool) -- switch for using a GPU device (default False)
  verbose (bool) -- switch verbose output (default True)
  """

  # prepare the dataset using tf.data api
  if load_data_from_disk:
    dat = ds.link_RSMIdat(data_params)

  elif load_data_from_generators:
    ll = CG_params['ll']
    generator=ds.dataset(**data_params)

    dat = tf.data.Dataset.from_generator(lambda: 
                generator.gen_rsmi_data(index, ll, buffer_size=buffer_size, 
                    cap=ll[0]+2*buffer_size+env_size), 
                    output_types=(tf.float32, tf.float32), 
                    output_shapes=(list(ll+(1,)), None))

  else:
    dat = tf.data.Dataset.from_tensor_slices((V, E))

  # adjust the shuffling and batching structure  
  dat = dat.shuffle(opt_params['shuffle']).batch(
      opt_params['batch_size']).repeat(opt_params['iterations'])

  # import coarse-graining model
  CG = CoarseGrainer(init_rule=init_rule, **CG_params)  
  f_ansatz = SeparableCritic(**critic_params)   

  if optimizer == None:
    # set optimiser as adam with given learning rate
    opt = tf.keras.optimizers.Adam(
           opt_params['learning_rate'])  
  else:
    opt = optimizer

  @tf.function
  def train_step(x, y):
    """Single training step: performs gradient descent 
    on the coarse-graining network and vbmi net simultaneously.
    Returns the most recent value of the RSMI estimate and the
    corresponding set of coarse-grained random variables H.

    Keyword arguments:
    x, y -- samples for random variables E and V, respectively. 
    """

    with tf.GradientTape() as tape:  
      if coarse_grain:
        h = CG(y) 
      else:
        h = y

      if use_GPU:
        with tf.device('/GPU:' + str(0)):
          mi = lowerbounds[bound](x, h, f_ansatz)
      else:
        mi = lowerbounds[bound](x, h, f_ansatz)
      loss = -mi 

      trainable_vars = []
      # train VBMI critic and coarse-graining filters simulatenously
      if isinstance(CG, tf.keras.Model):
        trainable_vars += CG.trainable_variables
      if isinstance(f_ansatz, tf.keras.Model):
        trainable_vars += f_ansatz.trainable_variables
      grads = tape.gradient(loss, trainable_vars)
      opt.apply_gradients(zip(grads, trainable_vars))
    return mi, h

  estimates = []
  coarse_vars = []
  filters = []

  pbar = tqdm(total=opt_params['iterations']*data_params['N_samples']\
                    //opt_params['batch_size'], desc='')
  i = 0
  for V, E in dat:
    CG.global_step = i

    # train coarse-graining filters and vbmi critic parameters simultaneously
    mi, h = train_step(E, V)
    coarse_vars.append(h.numpy())
    estimates.append(mi.numpy())
    filters.append(CG.coarse_grainer.get_weights()[0])

    if CG.method == 'pseudo-categorical sampling':
        pbar.set_description(
            f'Gumbel-softmax temperature {CG.tau:.2f}, I={mi:.2f}')
    elif CG.method == 'STE quantisation':
        pbar.set_description(f'STE quantisation, I={mi:.2f}')
    else:
        pbar.set_description(f'Convolution, I={mi:.2f}')

    pbar.update(1) # update progress bar for each iteration step
    i += 1

  print('Training complete.')
  return np.array(estimates), np.array(coarse_vars), np.array(filters), CG._Λ

