"""
Author: Doruk Efe Gökmen
Date: 10/01/2021
"""

import os
import json
import scipy.io
import numpy as np
import tensorflow as tf

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


def loadNSplit_DimerandVBS(bwImMatFN, Li, Lo, corr_diag_spins=False):
    """Transforms the (Li,Li) images containing 0..3 on the vertices, to (2Li,2Li) image 
    containing spins (-1,1) on vertices, bonds, and faces. 
    The bonds represent dimers, the vertices and faces are spins, 
    which are kept in order to make the lattice square along the same orientation. 
    Instead of making the spin degrees of freedom fixed, or random, 
    it is interesting to take them to be a VBS on the link going in the (-x,-y) direction.   

    :Authors: 
        Maciej Koch-Janusz, Zohar Ringel (2018)
    """
    mat = scipy.io.loadmat(bwImMatFN)
    raw = mat['Data_set_Z']

    raw_l = len(raw[0, 0, :])
    raw_n = len(raw[:, 0, 0])
    rawfat = np.zeros((raw_n, raw_l*2, raw_l*2))

    # Resolving dimers and adding the extra spins
    for n in range(raw_n):
      for i in range(raw_l):
        for j in range(raw_l):
            # [i=x,j=y] and these represent the vertices of the 2x2 unit cell
            rawfat[n, 2*i, 2*j] = np.floor(np.random.rand()*2)
            if corr_diag_spins:  # if neighbouring diagonal spin pairs are correlated
                rawfat[n, 2*i-1, 2*j-1] = rawfat[n, 2*i, 2*j]
            else:  # all spin values are totally uncorrelated
                rawfat[n, 2*i-1, 2*j-1] = np.floor(np.random.rand()*2)

            # i.e. dimer is pointing "up" (0,1) from (i,j)
            if raw[n, i, j] == 2:
                rawfat[n, 2*i-1, 2*j] = 0
                rawfat[n, 2*i, 2*j-1] = 0
            # i.e. dimer is pointing "down" (0,-1) from (i,j)
            if raw[n, i, j] == 0:
                rawfat[n, 2*i-1, 2*j] = 0
                rawfat[n, 2*i, 2*j-1] = 1
            # i.e. dimer is pointing "right" (1,0) from (i,j)
            if raw[n, i, j] == 1:
                rawfat[n, 2*i-1, 2*j] = 0
                rawfat[n, 2*i, 2*j-1] = 0
            # i.e. dimer is pointing "left" (-1,0) from (i,j)
            if raw[n, i, j] == 3:
                rawfat[n, 2*i-1, 2*j] = 1
                rawfat[n, 2*i, 2*j-1] = 0

    # Adjusting Li and Lo to account for the extra spins
    Li = Li*2
    Lo = Lo*2

    IOmargins = (Lo-Li)//2

    n_tiles = raw_l*2 // Lo

    bwImSetI = np.zeros(
        (len(raw[:, 0, 0])*n_tiles**2, Li, Li), dtype=np.float32)
    bwImSetO = np.zeros(
        (len(raw[:, 0, 0])*n_tiles**2, Lo, Lo), dtype=np.float32)

    c = 0
    for n in range(len(rawfat[:, 0, 0])):
        for i in range(n_tiles):
            for j in range(n_tiles):
                bwImSetI[c, :, :] = rawfat[n, i*Lo + IOmargins:(i+1)*Lo-IOmargins,
                                           j*Lo+IOmargins:(j+1)*Lo-IOmargins]
                bwImSetO[c, :, :] = rawfat[n, i*Lo:(i+1)*Lo, j*Lo:(j+1)*Lo]
                c += 1

    # Get flattened images
    bwImSetI = np.reshape(bwImSetI, (raw_n*n_tiles**2, Li**2), order='C')

    bwImSetO = np.reshape(bwImSetO, (raw_n*n_tiles**2, Lo**2), order='C')
    return (bwImSetI, bwImSetO)
