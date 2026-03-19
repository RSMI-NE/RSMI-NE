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
import warnings
import math
import numpy as np 
import pandas as pd
import tensorflow as tf
tfkl = tf.keras.layers

import wandb
try:
    from wandb.integration.keras import WandbCallback
except ImportError:
    from wandb.keras import WandbCallback

import rsmine.coarsegrainer.build_dataset as ds
from rsmine.coarsegrainer.cg_layers import CoarseGrainer

#SeparableCritic = VBMI_estimators.SeparableCritic
import rsmine.mi_estimator.VBMI_estimators as VBMI_estimators
from rsmine.mi_estimator.VBMI_bounds import lowerbounds
SeparableCritic = VBMI_estimators.SeparableCritic


def RSMI_estimate(mis: np.ndarray, ema_span: int=5000) -> float:
  """Exponential moving average  with span ema_span for the series of mi estimates.

  Keyword arguments:
  mis -- time series of mutual information estimates
  ema_span (int) -- span for evaluating average with exponentally larger weigths at later times
  """

  return pd.Series(mis).ewm(span=ema_span).mean().tolist()[-1]


def train_RSMI_optimiser(CG_params: dict, critic_params: dict, opt_params: dict,
                         data_params: dict, bound: str='infonce',
                         coarse_grain: bool=True, init_rule=None, optimizer=None,
                         index=None, buffer_size=None, env_size=None,
                         load_data_from_generators: bool=False, use_GPU: bool=False,
                         load_data_from_disk: bool=False, use_wandb: bool=False,
                         E=None, V=None, verbose=True, init_steps=100,
                         use_notebook=None,
                         discrete_center_steps: bool=False,
                         center_step_size: float=1.0,
                         center_update_every: int=10,
                         center_vote_threshold: float=0.0,
                         center_lr_multiplier: float=1.0,
                         **kwargs):
  """Main training loop for maximisation of RSMI [I(H:E)] 
  for coarse-graining optimisation.

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
  CG_params (dict) -- parameters of the coarse-grainer, includes distinction between
    regular lattices and arbitrary graph cases
  opt_params (dict) -- parameters of the optimiser
  data_params (dict) -- parameters for the sample dataset and the physical system
  bound (str) -- MI lower-bound (default InfoNCE)
  coarse_grain (bool) -- switch for coarse-graining (default True)
  init_rule -- initialisation for the coarse-graining rule (or initial conditions of Λ) 
    (default None)
  optimizer -- choice for stochastic gradient descent optimiser (default None: Adam)
  use_GPU (bool) -- switch for using a GPU device (default False)
  verbose (bool) -- switch verbose output (default True)
  use_notebook (bool) -- switch to Jupyter notebook version of tqdm (default None)
  """

  if use_notebook:
    from tqdm.notebook import tqdm
  else:
    from tqdm import tqdm

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

  # --- Identify center variables for discrete stepping ---
  # When discrete_center_steps=True, we exclude center params from the Adam
  # optimizer and instead update them via accumulated gradient-sign voting.
  center_vars = []
  if discrete_center_steps and hasattr(CG.coarse_grainer, 'effective_centers'):
    cg_layer = CG.coarse_grainer
    if hasattr(cg_layer, 'raw_centers'):
      center_vars = [cg_layer.raw_centers]
    elif hasattr(cg_layer, 'centers'):
      center_vars = [cg_layer.centers]
    center_var_ids = {id(v) for v in center_vars}
  else:
    center_var_ids = set()

  # Accumulator for gradient sign votes (one per center variable)
  center_vote_accum = [np.zeros(v.shape, dtype=np.float32) for v in center_vars]

  if discrete_center_steps and center_vars:
    print(f"Discrete center stepping: {len(center_vars)} center variable(s), "
          f"update every {center_update_every} steps, "
          f"vote threshold {center_vote_threshold}, step size {center_step_size}")

  @tf.function
  @tf.autograph.experimental.do_not_convert
  def train_step(x, y):
    """Single training step: performs gradient descent
    on the coarse-graining network and vbmi net simultaneously.
    Returns the most recent value of the RSMI estimate, the
    corresponding set of coarse-grained random variables H, and
    (optionally) the gradients w.r.t. center variables.

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
      loss = -mi + CG.regularisation_loss()

      # Collect all trainable variables
      all_trainable = []
      if isinstance(CG, tf.keras.Model):
        all_trainable += CG.trainable_variables
      if isinstance(f_ansatz, tf.keras.Model):
        all_trainable += f_ansatz.trainable_variables

      grads = tape.gradient(loss, all_trainable)

      # Separate center grads from the rest
      opt_pairs = []
      center_grads_out = []
      for g, v in zip(grads, all_trainable):
        if id(v) in center_var_ids:
          center_grads_out.append(g)
        else:
          if g is not None:
            opt_pairs.append((g, v))

      # Apply Adam only to non-center variables
      if opt_pairs:
        opt.apply_gradients(opt_pairs)

    return mi, h, center_grads_out


  def apply_discrete_center_update(step_idx):
    """Apply accumulated gradient-sign votes to shift centers by discrete steps."""
    for cv, accum in zip(center_vars, center_vote_accum):
      # Check if accumulated vote exceeds threshold
      # accum stores sign(-grad_loss) = sign(grad_MI) accumulated over the window
      # Positive accum => MI increases when center increases => move center right (+)
      # Negative accum => MI increases when center decreases => move center left (-)
      move = np.zeros_like(accum)
      move[accum > center_vote_threshold * center_update_every] = center_step_size
      move[accum < -center_vote_threshold * center_update_every] = -center_step_size

      if hasattr(CG.coarse_grainer, 'raw_centers') and cv is CG.coarse_grainer.raw_centers:
        # For sigmoid-parameterised centers, convert step in position-space
        # to a step in raw-space: delta_raw ≈ delta_pos / (sigmoid' * L)
        # Use a learning-rate multiplied step in raw space instead
        current_centers = CG.coarse_grainer.effective_centers.numpy()
        target_centers = current_centers + move * center_lr_multiplier
        # Clip to valid range and convert back to raw space
        L = CG.coarse_grainer.L
        target_norm = np.clip(target_centers / L, 1e-4, 1 - 1e-4)
        new_raw = np.log(target_norm / (1.0 - target_norm)).astype(np.float32)
        cv.assign(new_raw)
      else:
        # Direct center parameterisation: just shift
        cv.assign_add(move.astype(np.float32) * center_lr_multiplier)

    # Reset accumulators
    for j in range(len(center_vote_accum)):
      center_vote_accum[j][:] = 0.0


  estimates = []
  coarse_vars = []
  filters = []

  pbar = tqdm(total=opt_params['iterations']
  *int(np.ceil(data_params['N_samples']/opt_params['batch_size'])), desc='')

  epoch_id = 0
  print("Len dat: ",len(dat))
  for i, (V, E) in enumerate(dat):

    CG.global_step = i

    # train coarse-graining filters and vbmi critic parameters simultaneously
    mi, h, center_grads = train_step(E, V)

    # --- Discrete center stepping logic ---
    if discrete_center_steps and center_vars:
      # Accumulate gradient signs for center variables
      for j, cg in enumerate(center_grads):
        if cg is not None:
          # Sign of negative loss gradient = direction that increases MI
          center_vote_accum[j] += np.sign(-cg.numpy())

      # Every center_update_every steps, apply the discrete move
      if (i + 1) % center_update_every == 0:
        old_centers = CG.coarse_grainer.effective_centers.numpy()
        apply_discrete_center_update(i)
        new_centers = CG.coarse_grainer.effective_centers.numpy()
        if not np.allclose(old_centers, new_centers):
          print(f"  step {i+1}: centers {old_centers} -> {new_centers}")

    if i > init_steps and math.isnan(mi):
      if verbose:
        print('RSMI is found to be NaN.')
        warnings.warn('A numerical instability encountered during training.')
        print('Please try using a larger sampling or disable discretisation.')
      return np.array(estimates), np.array(coarse_vars), np.array(filters), CG
      raise SystemExit(0)
    else:
      if i % int(np.ceil(data_params['N_samples']/opt_params['batch_size'])) == 0:
        coarse_vars.append(h.numpy())
        estimates.append(mi.numpy())
        if CG_params['nonlinearCG'] is None or CG_params['nonlinearCG']==[0]:
            filters.append(CG.coarse_grainer.get_weights()[0])
        else:
            filters.append(CG.coarse_grainer.get_weights())    # this is currently a PLACEHOLDER

        if use_wandb:
          # log metrics using Weights and Biases API
          wandb.log({'EMA_30 MI': pd.Series(estimates).ewm(span=30).mean().to_numpy()[-1]})

        epoch_id += 1


      if CG.method == 'pseudo-categorical sampling':
          pbar.set_description(
              f'Gumbel-softmax temperature {CG.tau:.2f}, I={mi:.2f}')
      elif CG.method == 'STE quantisation':
          pbar.set_description(f'STE quantisation, I={mi:.2f}')
      else:
          pbar.set_description(f'Convolution, I={mi:.2f}')

      pbar.update(1) # update progress bar for each iteration step

  if verbose:
    print('Training complete.')
  return np.array(estimates), np.array(coarse_vars), np.array(filters), CG

