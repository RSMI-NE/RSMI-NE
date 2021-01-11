"""Implementation of variational lower bounds 
of mutual information using neural-networks.
Implemented in Tensorflow.
[Based on arXiv:1905.06922v1 Poole et al. (2019)]

Functions:
infonce_lower_bound() -- evaluates InfoNCE lower-bound
dv_upper_lower_bound() -- evaluates DV estimator

Author: Doruk Efe Gökmen
Date: 10/01/2021
"""

# pylint: disable-msg=E0611
# pylint: disable=invalid-unary-operand-type

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from utils import reduce_logmeanexp_offdiag, const_fn


def infonce_lower_bound(x, y, f_ansatz):
  """InfoNCE replica lower-bound (van den Oord et al. 2018)
  for estimating I(X:Y).

  Keyword arguments:
  x -- full sample dataset for random variable Y
  y -- full sample dataset for random variable Y
  f_ansatz (str) -- type of ansatz function
  """

  scores = f_ansatz(x, y)
  num_samples = scores.shape[0]
  positive_mask = tf.eye(num_samples, dtype=bool)

  return tfp.vi.mutual_information.lower_bound_info_nce(\
                logu=scores, joint_sample_mask=positive_mask)


def dv_upper_lower_bound(x, y, f_ansatz):
  """Donsker-Varadhan estimator for I(X:Y)

  Keyword arguments:
  x -- full sample dataset for random variable Y
  y -- full sample dataset for random variable Y
  f_ansatz (str) -- type of ansatz function
  """

  scores = f_ansatz(x, y)
  return tf.linalg.tensor_diag_part(scores)\
         - reduce_logmeanexp_offdiag(scores)


lowerbounds = {
    'infonce': infonce_lower_bound,
    'dv': dv_upper_lower_bound,
}

