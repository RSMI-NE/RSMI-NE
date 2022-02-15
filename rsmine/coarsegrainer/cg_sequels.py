import numpy as np
import tensorflow as tf


def cg_configs(Xs, rule, embedding=None, stride=None):
	"""Generate the coarse-grained configurations.

	Keyword arguments:
	Xs [np(tf).array] -- array of original degrees of freedom
	rule [np(tf).array with shape (block shape)+(#components or output channels)] 
		-- coarse-graining filter
	embedding (tf function) -- non-linear embedding 
		determining the type of coarse-grained variables 
		(or coarse-graining in state space)
	stride (int or tuple of ints) -- custom stride for the convolution
		(by default takes the shape of the coarse-graining filter)

	Returns:
	coarse_Xs (tf.array) -- coarse-grained (and embedded) configurations
	"""

	N_samples = Xs.shape[0]
	L = Xs.shape[1]
	# reformat the shape: (#samples,)+(system shape)+(#components of DOFs)
	fine_Xs = Xs.reshape((N_samples, L, L, 1))

	if stride is None:
		stride = rule.shape[:1] # stride over non-overlapping blocks
	
	convolved_Xs = tf.nn.convolution(Xs, rule, strides=stride)
	coarse_Xs = embedding(convolved_Xs)
	return coarse_Xs


def correlator(Xs, rule, embedding=None, stride=None):
	"""Computes the correlator for the given coarse-graining rule.

	Keyword arguments:
	Xs [np(tf).array] -- array of original degrees of freedom
	rule [np(tf).array with shape (block shape)+(#components or output channels)] 
		-- coarse-graining filter
	embedding (tf function) -- non-linear embedding 
		determining the type of coarse-grained variables 
		(or coarse-graining in state space)
	stride (int or tuple of ints) -- custom stride for the convolution
		(by default takes the shape of the coarse-graining filter)

	Returns:
	correlation
	"""

	coarse_Xs = cg_configs(Xs, rule, embedding=None, stride=None)

	correlation = tf.einsum('ijk,ijk->jk', coarse_Xs, coarse_Xs)
	return correlation
