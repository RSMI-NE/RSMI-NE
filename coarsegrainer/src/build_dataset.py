"""Builds the dataset for training convolutional neural networks 
to extract optimal coarse-graining rules.
Also prepares the (h, e) data for for a given filter Lambda 
to calculate the real-space mutual information I_Lambda(h:e).

Author: Doruk Efe Gökmen
Date: 08/04/2021
"""

import os
import warnings
from tqdm.autonotebook import tqdm
import itertools
import numpy as np
import tensorflow as tf
from cg_utils import array2tensor

def filename(model, lattice, L, J=None, T=None, 
            fileformat='txt', basedir='data', prefix='configs'):
    """Returns filename (str) according to naming scheme from specified model parameters.

    Keyword arguments:
    lattice (str) -- type of the underlying lattice, e.g. square, triangular
    L (int) -- linear size of the lattice
    J (float) -- Ising coupling constant (default None)
    T (float) -- temperature of the system (default None)
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


def RSMIdat_filename(model, lattice_type, L, T, buffer_size, 
                    region='V', dir='RSMIdat', **kwargs):
    """Returns filename (str) for the RSMI dataset containing the V, E samples.
    
    Keyword arguments:
    data_params (dict) -- specifications for the physical system and sampling
    region (str, either 'V' or 'E') -- specify whether to read V or E samples
    """

    if region != 'V' and region != 'E':
        warnings.warn("Warning: choose either 'V' or 'E' for the region.")

    name = region +'dat_'+model+'_'+lattice_type\
            +'_L%i_T%.3f_buffer%i.tfrecord'%(L, T, buffer_size)

    return os.path.join(os.pardir, "data", dir, name)


def link_RSMIdat(data_params, type=tf.float32):
    """Links the RSMI dataset in TFRecords format saved in the disk 
    with location and filename given by `RSMIdat_filename(data_params)`
    to the memory.

    Keyword arguments:
    data_params (dict) -- specifications for the physical system and sampling
    type (tensorflow.python.framework.dtypes.DType) -- type of the degrees of freedom
        in the physical model
    """

    def read_map_fn(x, type=type):
        return tf.io.parse_tensor(x, type)
    
    features = ['V', 'E']
    parts = []
    for i, feat in enumerate(features):
        parts.append(tf.data.TFRecordDataset(
            RSMIdat_filename(**data_params, region=feat)).map(read_map_fn))

    return tf.data.Dataset.zip(tuple(parts))


def save_RSMIdat(data_params, V, E):
    """Writes the RSMI dataset (V, E) pairs into `TFRecords` format
    Also returns the RSMI dataset in TF dataset format (Python iterator).

    Keyword arguments:
    data_params (dict) -- specifications for the physical system and sampling
    V (tensorflow Tensor) -- samples for the visible region
    E (tensorflow Tensor) -- samples for the environment region
    """

    features = ['V', 'E']

    RSMIdat = tf.data.Dataset.from_tensor_slices((V, E))

    for i, _ in enumerate(RSMIdat.element_spec):
        ds_i = RSMIdat.map(lambda *args: args[i]).map(tf.io.serialize_tensor)
        writer = tf.data.experimental.TFRecordWriter(
            RSMIdat_filename(**data_params, region=features[i]))
        writer.write(ds_i)
    return RSMIdat


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


def get_V(x, index, ll):
    """Get the region to be coarse-grained.

    Keyword arguments:
    x -- a sample configuration
    index (tuple of int) -- index of upper-left corner site of the visible block V
    ll (tuple of int) -- shape of the visible block V
    """

    dim = len(index)
    L = x.shape[0]

    x_ext = np.pad(x, [(0, ll[d]) for d in range(dim)], 'wrap')

    visible_slice = tuple([slice(index[d], index[d]+ll[d])
                           for d in range(dim)])
    v = x_ext[visible_slice]

    return v.flatten()


def get_E(x, index, L_B, ll, cap=None):
    """Get the environment E of the coarse-grained region.

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

    return e



class dataset():
    """
    Class generating the dataset from full raw sample dataset of degrees of freedom for
    generating the visible block and environment dataset for rsmi optimisation.

    Methods:
    get_Vs -- samples for the region to be coarse-grained (visible region)
    get_Es -- samples for the environment region
    rsmi_data() -- returns samples of V and E
    chop_data()
    """

    def __init__(self, model, L, lattice_type, dimension=2, configurations = None, N_samples=None, 
                J=None, Nq=None, T=None, basedir='data', verbose=True, **kwargs):
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
        basedir (str) -- directory name of the input data
        verbose (bool)
        """

        self.model = model
        self.J = J
        self.Nq = Nq #number of states for a Potts variable
        self.T = T
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
        }
        
        self.dtype = int

        self.fileformat = 'txt'
        if self.model == 'dimer2d':
            self.fileformat = 'mat'
        elif self.model == 'intdimer2d':
            self.fileformat = 'npy'

        if isinstance(configurations, np.ndarray):
            self.configurations = configurations
            if len(self.configurations) > self.N_configs:
                self.N_configs = len(self.configurations)

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
            else:
                warnings.warn("Warning: the dataset with desired system parameters could not be found.")


    def gen_Vs(self, indices, ll, shape=None):
        """Generator for for the visible block to be coarse-grained.

        Keyword arguments:
        indices (list of tuples of int) -- index of the upper-left corner site of V
        ll (tuple of int) -- shape of V
        shape (tuple of int) -- shape of the configurations
            (default None: assumes square system, i.e. shape=(L,L))
        """

        def get_index(indices, t):
            if type(indices) is list:
                return indices[t]
            else:
                return indices


        if shape is None:
            shape = self.dimension * (self.L, )

        if self.verbose:
            print('Preparing the visible block dataset...')

        for t in range(self.N_configs):
            index = get_index(indices, t)
            config = self.configurations[t].reshape(shape)

            # additional dimension for one-hot encoding
            yield array2tensor(get_V(config, index, ll).reshape(ll + (1,)))

        if self.verbose:
            print('Visible block dataset prepared.')


    def gen_Es(self, indices, ll, buffer_size=2, cap=None, shape=None):
        """Generator for for the environment E of the visible block.

        Keyword arguments:
        indices (list of tuples of int) -- index of the upper-left corner site of V
        ll (tuple of int) -- shape of V
        buffer_size (int) -- buffer width (default 2)
        cap (int) -- subsystem size<L to cap the environment 
            (default None: environment is the rest of the system)
        shape (tuple of int) -- shape of the configurations
            (default None: assumes square system, i.e. shape=(L,L))
        """

        def get_index(indices, t):
            if type(indices) is list:
                return indices[t]
            else:
                return indices


        if self.verbose:
            print('Preparing the environment dataset...')

        if shape is None:
            shape = self.dimension * (self.L, )

        for t in range(self.N_configs):
            index = get_index(indices, t)
            config = self.configurations[t].reshape(shape)

            yield tf.cast(get_E(config, index, buffer_size, ll, cap=cap), tf.float32)

        if self.verbose:
            print('Environment dataset prepared.')


    def gen_rsmi_data(self, index, ll, buffer_size=2, cap=None, shape=None):
        """Generator for for the visible block V and its environment E.

        Keyword arguments:
        index (int) -- index of the upper-left corner site of V
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

        configs = self.configurations.reshape((len(self.configurations), ) + shape)

        for t in range(self.N_configs):
            v, e = partition_x(configs[t], index, buffer_size, ll, cap=cap)

            V = array2tensor(v.reshape(ll + (1,))) # additional dimension for one-hot encoding
            E = tf.cast(e, tf.float32)

            yield V, E

        if self.verbose:
            print('RSMI dataset prepared.')


    def rsmi_data(self, indices, ll, buffer_size=2, cap=None, shape=None):
        """Returns data for the visible block V and its environment E.

        Keyword arguments:
        indices (list of tuples of int) -- index of the upper-left corner site of V
        ll (tuple of int) -- shape of V
        buffer_size (int) -- buffer width (default 2)
        cap (int) -- subsystem size<L to cap the environment 
            (default None: environment is the rest of the system)
        shape (tuple of int) -- shape of the configurations
            (default None: assumes square system, i.e. shape=(L,L))
        """

        def get_index(indices, t):
            if type(indices) is list:
                return indices[t]
            else:
                return indices


        if self.verbose:
            print('Preparing the RSMI dataset...')

        if shape is None:
            shape = self.dimension * (self.L, )

        configs = self.configurations.reshape((len(self.configurations), ) + shape)

        Vs = []
        Es = []

        for t in range(self.N_configs):
            index = get_index(indices, t)

            v, e = partition_x(configs[t], index, buffer_size, ll, cap=cap)
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

        if cap is None:
        	cap = self.L

        dim = len(ll)
        offset = cap - ll[0] # boundary offset for the visible block
        lin_size = self.L - offset

        for i, index in tqdm(enumerate(itertools.product(
                            *[range(0, lin_size, stride) for _ in range(dim)]))):
            if i == 0:
                index_0 = np.array([offset for _ in range(dim)]) # index of V

                Vs, Es = self.rsmi_data(tuple(index_0), ll,
                        buffer_size=buffer_size, cap=cap, shape=shape)
            else:
                index += index_0 # index of V

                Vs_, Es_ = self.rsmi_data(tuple(index), ll, 
                        buffer_size=buffer_size, cap=cap, shape=shape)

                Vs = tf.concat([Vs, Vs_], 0)
                Es = tf.concat([Es, Es_], 0)

        return Vs, Es
