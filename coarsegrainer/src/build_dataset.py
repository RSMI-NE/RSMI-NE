"""Builds the dataset for training convolutional neural networks 
to extract optimal coarse-graining rules.
Also prepares the (h, e) data for for a given filter Lambda 
to calculate the real-space mutual information I_Lambda(h:e).

Author: Doruk Efe Gökmen
Date: 10/01/2021
"""

# pylint: disable-msg=E0611

import os
import sys
from tqdm.notebook import tqdm
import itertools
#from tqdm import tqdm
import numpy as np
#import pandas as pd
import tensorflow as tf
from cg_utils import array2tensor #, loadNSplit_DimerandVBS

def filename(model, lattice, L, J=None, T=None, srn_correlation=None, fileformat='txt', basedir='data', prefix='configs'):
    """Generates filename (str) according to naming scheme from specified model parameters.

    Keyword arguments:
    lattice (str) -- type of the underlying lattice, e.g. square, triangular
    L (int) -- linear size of the lattice
    J (float) -- Ising coupling constant (default None)
    T (float) -- temperature of the system (default None)
    srn_correlation (bool) -- full correlation for corrupted dimer variables on lattice faces
    """

    if basedir == 'data':
        basedir = os.path.join(os.pardir, basedir)

    if model[0:5] == 'ising':
        return os.path.join(basedir, prefix+"_%s_%s_L%i_K%.2f.%s" \
                            % (model, lattice, L, J, fileformat))
    elif model[0:8] == 'intdimer':
        return os.path.join(basedir, prefix+"_%s_%s_L%i_T%.3f.%s"
                            % (model, lattice, L, T, fileformat))
    else:
        return os.path.join(basedir, prefix+"_%s_%s.%s" \
                            % (model, lattice, fileformat))


def partition_x(x, index, L_B, ll, cap=None):
    """Partitions a sample configuration into a visible block
    and an annular environment separated by a buffer. 

    Keyword arguments:
    x -- a sample configuration
    index (tuple of int) -- index of upper-left corner site of the visible block V
    L_B (int) -- width of the buffer
    ll (tuple of int) -- shape of the visible block V
    cap (int) -- linear size of the finite subsystem capped from x
    """

    dim = len(index)
    L = x.shape[0]

    def cap_minus(d, cap=L):
        return (cap - ll[d])//2

    def cap_plus(d, cap=L):
        return (cap + ll[d])//2

    if cap is None:
        cap = L

    x_ext = np.pad(x, [(cap_minus(d, cap=cap), cap_plus(d, cap=cap))
                       for d in range(dim)], 'wrap')
    index = tuple(
        np.add(index, tuple([cap_minus(d, cap=cap) for d in range(dim)])))

    # get environment
    t = np.zeros(x_ext.shape, dtype=bool)
    cap_slice = tuple([slice(index[d]-cap_minus(d, cap=cap), index[d]+cap_plus(d, cap=cap))
                       for d in range(dim)])
    t[cap_slice] = np.ones(dim*(cap, ), dtype=bool)

    buffer_slice = tuple([slice(index[d]-L_B, index[d]+L_B+ll[d])
                          for d in range(dim)])
    buffer_mask_size = tuple([2*L_B+ll[d] for d in range(dim)])
    t[buffer_slice] = np.zeros(buffer_mask_size, dtype=bool)
    e = x_ext[t]

    # get visible block
    visible_slice = tuple([slice(index[d], index[d]+ll[d])
                           for d in range(dim)])
    v = x_ext[visible_slice]
    v = v.flatten()

    return v, e


class dataset():
    """
    Class generating the dataset from full raw sample dataset of degrees of freedom for
    generating the visible block and environment dataset for rsmi optimisation.
    """

    def __init__(self, model, L, lattice_type, dimension=2, configurations = None, N_samples=None, J=None, Nq=None,
                    T=None, srn_correlation=None, basedir='data', verbose=True):
        """ Constructs all necessary attributes of the physical system.        
        
        Attributes:
        model (str) -- type of the physical model
        L (int) -- linear system size
        lattice_type (str) -- e.g. square or triangular
        dimension (int) -- dimensionality of the sytem
        configurations (np.array) -- input configurations pre-loaded into memory (default None)
        N_samples (int) -- total number of sample configurations (default None)
        J (float) -- Ising coupling constant (default None)
        Nq (int) -- number of states for a Potts degree of freedom (default None)
        T (float) -- temperature of the system
        srn_correlation -- full correlation for corrupted dimer variables on lattice faces
        basedir (str) -- directory name of the input data
        verbose (bool)

        Methods:
        rsmi_data() -- returns samples of V and E
        """

        self.model = model
        self.J = J
        self.Nq = Nq #number of states for a Potts variable
        self.T = T
        self.srn_correlation = srn_correlation
        self.L = L 
        self.dimension = dimension 
        self.N_samples = N_samples
        self.N_configs = self.N_samples
        self.lattice_type = lattice_type
        self.basedir = basedir
        self.verbose = verbose

        self.system_params = {
            'model': self.model,
            'lattice': self.lattice_type,
            'J': self.J,
            'L': self.L,
            'srn_correlation': self.srn_correlation
        }
        
        self.dtype = int

        self.fileformat = 'txt'
        if self.model == 'dimer2d':
            self.fileformat = 'mat'
        elif self.model == 'intdimer2d':
            self.fileformat = 'npy'

        if isinstance(configurations, np.ndarray):
            self.configurations = configurations

        else:
            if os.path.isfile(filename(**self.system_params, T=self.T, fileformat=self.fileformat, basedir=basedir)):
                if self.verbose:
                        print("Existing data found.\n Loading the data...")

                if self.model == 'intdimer2d':
                    x = np.load(filename(**self.system_params, T=self.T,
                                        fileformat='npy', basedir=basedir))
                    self.N_configs = len(x)
                    self.configurations = np.reshape(x, (self.N_configs, self.L*self.L))
                    
                else:
                    # self.configurations = pd.read_csv(
                    #     filename(**self.system_params, fileformat=self.fileformat, basedir=basedir), delimiter=' ',
                    #             header=None).to_numpy(dtype=int)[:, 0:-1]
                    self.configurations = np.loadtxt(filename(
                        **self.system_params, fileformat=self.fileformat, basedir=basedir), dtype=int)

                if self.verbose:
                    print("Loading complete.")


    def rsmi_data(self, index, ll, buffer_size=2, cap=None, shape=None):
        """Returns data for the visible block V and its environment E.

        Keyword arguments:
        index (tuple of int) -- index of the upper-left corner site of V
        ll (tuple of int) -- shape of V
        buffer_size (int) -- buffer width (default 2)
        cap (int) -- subsystem size<L to cap the environment 
            (default None: environment is the rest of the system)
        shape (tuple of int) -- shape of the configurations
            (default None: assumes square system, i.e. shape=(L,L))
        """

        if self.verbose:
            print('Preparing the RSMI dataset...')

        if shape is None:
            shape = self.dimension * (self.L, )

        Vs = []
        Es = []
        for t in range(self.N_configs):
            config = self.configurations[t].reshape(shape)

            v, e = partition_x(config, index, buffer_size, ll, cap=cap)
            Vs.append(v)
            Es.append(e)
        
        # additional dimension for one-hot encoding
        Vs = np.reshape(Vs, (np.shape(Vs)[0],) + ll + (1,)) 

        if self.verbose:
            print('RSMI dataset prepared.')

        return array2tensor(Vs), array2tensor(Es)

    def chop_data(self, stride, ll, buffer_size, cap=None, shape=None):
        """Chops real-space configurations according to some stride 
        to generate many from a given dataset (V,E) samples.
        Note: Using this might be dangerous in the absence of 
        translation invariance.

        Keyword arguments:
        stride = int
        ll (tuple of int) -- shape of V
        buffer_size (int) -- buffer width (default 2)
        cap (int) -- subsystem size<L to cap the environment 
            (default None: environment is the rest of the system)
        shape (tuple of int) -- shape of the configurations
            (default None: assumes square system, i.e. shape=(L,L))
        """

        dim = len(ll)
        env_shell = (cap - ll[0] - 2*buffer_size)//2

        index = np.array([env_shell+1 for _ in range(dim)])
        Vs, Es = self.rsmi_data(index, ll,
                        buffer_size=buffer_size, cap=cap, shape=shape)

        lin_size = self.L-2*env_shell
        for index in itertools.product(*[range(0,lin_size,stride) for _ in range(dim)]):
            Vs_, Es_ = self.rsmi_data(tuple(index), ll, 
                    buffer_size=buffer_size, cap=cap, shape=shape)

            Vs = tf.concat([Vs, Vs_], 0)
            Es = tf.concat([Es, Es_], 0)

        return Vs, Es
