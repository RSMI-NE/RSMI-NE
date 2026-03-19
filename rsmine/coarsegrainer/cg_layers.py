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


import numpy as np 
import tensorflow as tf
#from tensorflow.keras import datasets, models, regularizers, backend  
#from tensorflow.python.framework import ops

import tensorflow_probability as tfp
tfd = tfp.distributions
tfkl = tf.keras.layers
#tfpl = tfp.layers


def filter_regularisation_loss(ws, l1_reg=0.0, orthogonality_reg=0.0):
  """Computes the regularisation loss for the coarse-graining filter weights.

  The loss consists of:
  1. L1 penalty on the filter weights (encourages sparsity).
  2. Soft orthogonality penalty ||W^T W - I||_F^2 (encourages filters
     to capture independent features).

  Keyword arguments:
  ws -- filter weight tensor; the last axis indexes the hidden (filter) dimension
  l1_reg (float) -- strength of L1 regularisation
  orthogonality_reg (float) -- strength of orthogonality regularisation
  """

  reg_loss = tf.constant(0.0)

  if l1_reg > 0:
    reg_loss += l1_reg * tf.reduce_sum(tf.abs(ws))

  if orthogonality_reg > 0:
    # Flatten each filter to a vector: reshape to (num_spatial, num_filters)
    w_flat = tf.reshape(ws, (-1, tf.shape(ws)[-1]))
    # Normalise each filter column to unit norm for a scale-invariant penalty
    w_norm = tf.math.l2_normalize(w_flat, axis=0)
    # Gram matrix of normalised filters
    gram = tf.matmul(tf.transpose(w_norm), w_norm)
    identity = tf.eye(tf.shape(gram)[0])
    reg_loss += orthogonality_reg * tf.reduce_sum(tf.square(gram - identity))

  return reg_loss


def _ste_round(x):
  """Straight-through estimator for rounding.
  Forward: round to nearest integer. Backward: identity (pass gradients through).
  """
  return x + tf.stop_gradient(tf.round(x) - x)


def _ste_sign(x):
  """Straight-through estimator for sign/binarisation.
  Forward: sign(x). Backward: identity (pass gradients through).
  """
  return x + tf.stop_gradient(tf.sign(x) - x)


def _make_window(positions, centers, widths, window_type='rectangular', tau=1.0):
  """Constructs a differentiable window mask over spatial positions.

  Arguments:
  positions -- 1D tensor of spatial position indices, shape (L,)
  centers -- learnable center positions, shape (num_filters,)
  widths -- half-widths for each filter, shape (num_filters,)
  window_type -- 'rectangular' (sigmoid edges) or 'gaussian'
  tau -- temperature/sharpness for the sigmoid edges (rectangular only)

  Returns:
  window -- tensor of shape (L, num_filters) with values in [0, 1]
  """

  # positions: (L,) -> (L, 1), centers: (F,) -> (1, F)
  pos = tf.cast(tf.reshape(positions, (-1, 1)), tf.float32)
  c = tf.reshape(centers, (1, -1))
  w = tf.reshape(widths, (1, -1))

  if window_type == 'gaussian':
    # Gaussian envelope: exp(-(x - c)^2 / (2 * w^2))
    return tf.exp(-0.5 * tf.square((pos - c) / (w + 1e-6)))

  else:  # 'rectangular' with differentiable sigmoid edges
    # Eq. (7) from paper: W_νi = σ_τ(i - c + w/2) · σ_τ(c + w/2 - i)
    # where w is the FULL width of the window
    half_w = w / 2.0
    left = tf.sigmoid((pos - c + half_w) / tau)
    right = tf.sigmoid((c + half_w - pos) / tau)
    return left * right


class Conv2DWindowed(tfkl.Layer):
  """Windowed convolution layer for coarse-graining.

  Each filter has learnable weights multiplied by a spatial window
  function parameterised by a learnable center (and optionally width).
  This encourages spatially localised filters, suitable for e.g.
  regulatory sequence analysis where binding sites are localised.
  """

  def __init__(self, hidden_dim: int, visible_dim: int=1,
               input_shape=(1, 160), init_rule=None,
               init_centers=None, max_width: float=25.0,
               adaptive_width: bool=False, window_tau: float=1.0,
               window_type: str='rectangular',
               use_sigmoid_centers: bool=False):
    """
    Keyword arguments:
    hidden_dim -- number of filters (coarse-grained components)
    visible_dim -- number of components of original degrees of freedom
    input_shape -- (spatial_dim_1, spatial_dim_2) shape of visible block
    init_rule -- initial weight values (np.ndarray or None)
    init_centers -- initial center positions for each filter, shape (hidden_dim,)
    max_width -- maximum half-width of the window (or fixed width if not adaptive)
    adaptive_width -- if True, each filter learns its own width
    window_tau -- temperature for sigmoid edges (rectangular window)
    window_type -- 'rectangular' or 'gaussian'
    use_sigmoid_centers -- if True, parameterise centers via sigmoid to [0, L]
    """

    super(Conv2DWindowed, self).__init__()

    self.hidden_dim = hidden_dim
    self.input_shape_ = input_shape
    self.visible_dim = visible_dim
    self.max_width = max_width
    self.adaptive_width = adaptive_width
    self.window_tau = window_tau
    self.window_type = window_type
    self.use_sigmoid_centers = use_sigmoid_centers

    # Spatial extent along the "long" axis (for 1D sequences: input_shape = (1, L))
    self.L = input_shape[-1]

    # Filter weights (same shape as Conv2DSingle)
    if isinstance(init_rule, np.ndarray):
      initializer = tf.constant_initializer(init_rule)
    else:
      initializer = tf.random_normal_initializer()

    self._raw_ws = self.add_weight(
        name="ws",
        shape=input_shape + (visible_dim,) + (hidden_dim,),
        initializer=initializer,
        trainable=True)

    # Center positions
    if init_centers is not None:
      center_init = np.array(init_centers, dtype=np.float32)
    else:
      # Default: evenly spaced centers
      center_init = np.linspace(0, self.L - 1, hidden_dim).astype(np.float32)

    if use_sigmoid_centers:
      # Store raw (pre-sigmoid) parameters; inverse sigmoid of normalised centers
      center_init_normalised = np.clip(center_init / self.L, 1e-4, 1 - 1e-4)
      raw_init = np.log(center_init_normalised / (1.0 - center_init_normalised))
      self.raw_centers = self.add_weight(
          name="raw_centers",
          shape=(hidden_dim,),
          initializer=tf.constant_initializer(raw_init),
          trainable=True)
    else:
      self.centers = self.add_weight(
          name="centers",
          shape=(hidden_dim,),
          initializer=tf.constant_initializer(center_init),
          trainable=True)

    # Widths
    if adaptive_width:
      width_init = np.full(hidden_dim, max_width / 2.0, dtype=np.float32)
      # Store raw (pre-sigmoid) width params
      raw_width_init = np.log(width_init / (max_width - width_init + 1e-6))
      self.raw_widths = self.add_weight(
          name="raw_widths",
          shape=(hidden_dim,),
          initializer=tf.constant_initializer(raw_width_init),
          trainable=True)

  @property
  def effective_centers(self):
    """Returns the effective center positions."""
    if self.use_sigmoid_centers:
      return tf.sigmoid(self.raw_centers) * self.L
    else:
      return self.centers

  @property
  def effective_widths(self):
    """Returns the effective half-widths."""
    if self.adaptive_width:
      return tf.sigmoid(self.raw_widths) * self.max_width
    else:
      return tf.constant(self.max_width, shape=(self.hidden_dim,))

  def get_window(self):
    """Compute and return the current window functions for all filters.

    Returns:
      window -- tensor of shape (L, hidden_dim) with values in [0, 1]
    """
    positions = tf.range(self.L, dtype=tf.float32)
    return _make_window(positions, self.effective_centers,
                        self.effective_widths,
                        window_type=self.window_type,
                        tau=self.window_tau)

  @property
  def ws(self):
    """Returns the effective (windowed) filter weights Λ_νi = W_νi · λ_νi.

    The window W is kept differentiable w.r.t. center/width parameters
    so that ∂I/∂c_ν can be computed for the discrete center voting mechanism.
    Center variables must be EXCLUDED from the Adam optimizer (handled by
    train_RSMI_optimiser with discrete_center_steps=True) to prevent
    continuous sliding of windows.
    """
    window = self.get_window()
    window_broadcast = tf.reshape(window, (1, self.L, 1, self.hidden_dim))
    return self._raw_ws * window_broadcast

  # Keep get_effective_weights as an alias for clarity
  def get_effective_weights(self):
    """Alias for the ws property."""
    return self.ws

  def get_weights(self):
    """Override get_weights to return the effective (windowed) weights
    as the first element, since those are the physically meaningful filters.
    """
    effective = self.ws.numpy()
    rest = [v.numpy() for v in self.trainable_variables if v is not self._raw_ws]
    return [effective] + rest

  def call(self, inputs):
    """Applies windowed filter to the input.

    Keyword arguments:
    inputs -- tensor encoding the visible block (V), shape (batch, i, j, a, d)
    """

    return tf.einsum('tijad,ijab->tbd', inputs, self.ws)


class Conv2DGaussian(tfkl.Layer):
  """Pure Gaussian envelope filter for coarse-graining 1D sequences.

  Each filter is parameterised by a scalar amplitude, a learnable center,
  and a learnable width (biased towards ~20 bp).  There are no free
  per-position weights — the filter IS the Gaussian (variational ansatz 1
  in the paper).  This drastically reduces the number of free parameters
  from N per filter to just 3 (amplitude, center, width), which is
  essential when the data is noisy and the unconstrained filter is
  overparameterised relative to the information signal.

  Optionally allows bipartite sites (two Gaussian components per filter)
  to capture split binding motifs like the RNAP -10/-35 boxes.
  """

  def __init__(self, hidden_dim: int, visible_dim: int=1,
               input_shape=(1, 160), init_rule=None,
               init_centers=None,
               target_width: float=20.0,
               min_width: float=10.0,
               max_width: float=30.0,
               use_sigmoid_centers: bool=True,
               allow_bipartite: bool=False):
    super(Conv2DGaussian, self).__init__()

    self.hidden_dim = hidden_dim
    self.visible_dim = visible_dim
    self.L = input_shape[-1]
    self.target_width = target_width
    self.min_width = min_width
    self.max_width = max_width
    self.use_sigmoid_centers = use_sigmoid_centers
    self.allow_bipartite = allow_bipartite
    self.n_components = 2 if allow_bipartite else 1

    # --- Centers ---
    if init_centers is not None:
      center_init = np.array(init_centers, dtype=np.float32)
      if allow_bipartite and center_init.ndim == 1:
        # duplicate for two components with small offset
        center_init = np.stack([center_init, center_init + 20.0], axis=-1)
    else:
      center_init = np.linspace(0.2 * self.L, 0.8 * self.L,
                                hidden_dim).astype(np.float32)
      if allow_bipartite:
        center_init = np.stack([center_init, center_init + 20.0], axis=-1)

    if not allow_bipartite and center_init.ndim == 1:
      center_init = center_init[:, None]  # (hidden_dim, 1)

    if use_sigmoid_centers:
      c_norm = np.clip(center_init / self.L, 1e-4, 1 - 1e-4)
      raw_init = np.log(c_norm / (1.0 - c_norm))
      self.raw_centers = self.add_weight(
          name="raw_centers", shape=(hidden_dim, self.n_components),
          initializer=tf.constant_initializer(raw_init), trainable=True)
    else:
      self.centers = self.add_weight(
          name="centers", shape=(hidden_dim, self.n_components),
          initializer=tf.constant_initializer(center_init), trainable=True)

    # --- Widths ---  (parameterised so that tanh(0) -> target_width)
    width_init = np.zeros((hidden_dim, self.n_components), dtype=np.float32)
    self.raw_widths = self.add_weight(
        name="raw_widths", shape=(hidden_dim, self.n_components),
        initializer=tf.constant_initializer(width_init), trainable=True)

    # --- Amplitudes ---
    amp_init = np.random.randn(hidden_dim, self.n_components).astype(np.float32) * 0.1
    self.amplitudes = self.add_weight(
        name="amplitudes", shape=(hidden_dim, self.n_components),
        initializer=tf.constant_initializer(amp_init), trainable=True)

    # --- Mixing logits (bipartite only) ---
    if allow_bipartite:
      self.mixing_logits = self.add_weight(
          name="mixing_logits", shape=(hidden_dim, self.n_components),
          initializer=tf.ones_initializer(), trainable=True)

  @property
  def effective_centers(self):
    if self.use_sigmoid_centers:
      return tf.sigmoid(self.raw_centers) * self.L
    else:
      return self.centers

  @property
  def effective_widths(self):
    width_range = (self.max_width - self.min_width) / 2.0
    width_center = (self.max_width + self.min_width) / 2.0
    return width_center + width_range * tf.nn.tanh(self.raw_widths)

  @property
  def ws(self):
    """Compute the explicit filter weight matrix from Gaussian parameters.
    Returns shape (L, hidden_dim) for compatibility with get_weights().
    """
    return self._compute_filters()

  def _compute_filters(self):
    """Construct filters as W_vi = sum_k alpha_k * G(i; c_k, w_k)."""
    positions = tf.cast(tf.range(self.L), tf.float32)
    pos = tf.reshape(positions, (self.L, 1, 1))        # (L, 1, 1)
    c = tf.reshape(self.effective_centers, (1, self.hidden_dim, self.n_components))
    w = tf.reshape(self.effective_widths, (1, self.hidden_dim, self.n_components))
    a = tf.reshape(self.amplitudes, (1, self.hidden_dim, self.n_components))

    components = a * tf.exp(-0.5 * tf.square((pos - c) / (w + 1e-6)))

    if self.allow_bipartite:
      mixing = tf.nn.softmax(self.mixing_logits, axis=1)
      mixing = tf.reshape(mixing, (1, self.hidden_dim, self.n_components))
      filters = tf.reduce_sum(mixing * components, axis=2)  # (L, hidden_dim)
    else:
      filters = tf.squeeze(components, axis=2)

    return filters

  def call(self, inputs):
    """inputs: (batch, 1, L, visible_dim, encoding) or similar."""
    filters = self._compute_filters()  # (L, hidden_dim)
    # Reshape to (1, L, 1, hidden_dim) to match Conv2DSingle ws shape convention
    ws_4d = tf.reshape(filters, (1, self.L, 1, self.hidden_dim))
    return tf.einsum('tijad,ijab->tbd', inputs, ws_4d)

  def get_binding_sites(self, threshold=0.1):
    """Extract predicted TF binding site locations."""
    centers_np = self.effective_centers.numpy()
    widths_np = self.effective_widths.numpy()
    amplitudes_np = self.amplitudes.numpy()

    sites = []
    for i in range(self.hidden_dim):
      for j in range(self.n_components):
        if abs(amplitudes_np[i, j]) > threshold:
          sites.append({
              'filter_id': i, 'component': j,
              'center': centers_np[i, j],
              'width': widths_np[i, j],
              'amplitude': amplitudes_np[i, j]})
    return sites


class Conv2DSingle(tfkl.Layer):
  """Custom convolution layer to produce the (stochastic) coarse-graining map
  from 2(or 1)-d visible degrees of freedom.
  """

  def __init__(self, hidden_dim: int, visible_dim: int=1, input_shape=(2, 2), init_rule=None):
    """Constructs the convolutional net.

    Attributes:
    ws -- weights of the kernel

    Methods:
    call() -- call the network as a function
    """

    super(Conv2DSingle, self).__init__()

    if isinstance(init_rule, np.ndarray):
      initializer = tf.constant_initializer(init_rule)
    else:
      initializer = tf.random_normal_initializer()

    self.ws = self.add_weight(
        name="ws",
        shape=input_shape + (visible_dim,) + (hidden_dim,),
        initializer=initializer,
        trainable=True)

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

  def __init__(self, hidden_dim: int, visible_dim: int=1,
              input_shape: tuple=(2, 2, 2), init_rule=None):
    """Constructs the convolutional net.
    
    Attributes:
    ws -- weights of the kernel

    Methods: 
    call() -- call the network as a function
    """

    super(Conv3DSingle, self).__init__()

    if isinstance(init_rule, np.ndarray):
      initializer = tf.constant_initializer(init_rule)
    else:
      initializer = tf.random_normal_initializer()

    self.ws = self.add_weight(
        name="ws",
        shape=input_shape + (hidden_dim,),
        initializer=initializer,
        trainable=True)

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

  def __init__(self, hidden_dim: int,  visible_dim: int=1,
                input_shape: tuple=(2,), init_rule=None):
    """Constructs the convolutional net.
    
    Attributes:
    ws -- weights of the kernel

    Methods: 
    call() -- call the network as a function
    """

    super(ConvGraphSingle, self).__init__()

    if isinstance(init_rule, np.ndarray):
      initializer = tf.constant_initializer(init_rule)
    else:
      initializer = tf.random_normal_initializer()

    self.ws = self.add_weight(
        name="ws",
        shape=input_shape + (hidden_dim,),
        initializer=initializer,
        trainable=True)

  def call(self, inputs):
    """Computes the dot product between the input and kernel weights.

    Keyword arguments:
    inputs -- tensor encoding the visible block to be coarse-grained
    """

    return tf.einsum('tik,is->tsk', inputs, self.ws)
    
class ConvGraphMultiple(tfkl.Layer):#(tf.keras.Model):#(tfkl.Layer):
  """Custom convolution layer to produce the (stochastic) coarse-graining map
  from visible degrees of freedom on a (netoworkx) graph. The difference to
  Conv2DSingle is that ll in *not* a tuple, but an int defining the radius of V, 
  all the configurations are otherwise one-dimensional. 
  The input_shape is (#edges in V,)
  """

  def __init__(self, hidden_dim, input_shape=(2,), layer_sizes = None, init_rule=None):
    """Constructs the convolutional net.
    
    Attributes:
    ws -- weights of the kernel

    Methods: 
    call() -- call the network as a function
    """

    super(ConvGraphMultiple, self).__init__()
    
    #self.layer_list = [tfkl.Dense(layer_size,kernel_initializer='random_normal') for layer_size in layer_sizes]
    self.layer_1 = tfkl.Dense(2,kernel_initializer='random_normal',activation='sigmoid')
    self.layer_2 = tfkl.Dense(1,kernel_initializer='random_normal')


  def call(self, inputs):
    """Computes the dot product between the input and kernel weights.

    Keyword arguments:
    inputs -- tensor encoding the visible block to be coarse-grained
    """
    x = self.layer_list[0](tf.reshape(inputs,inputs.shape[:-1]))   #(inputs)
    if len(self.layer_list) > 1:
        for i,layer in enumerate(self.layer_list[1:]):
            x = tf.nn.sigmoid(x)
            x = layer(x)
    return tf.reshape(x,x.shape+(1,)) # x
    
    return tf.reshape(x,x.shape+(1,))
    


class ConvGraphMultiple3(tf.keras.Model):#(tfkl.Layer):
  """Custom convolution layer to produce the (stochastic) coarse-graining map
  from visible degrees of freedom on a (netoworkx) graph. The difference to
  Conv2DSingle is that ll in *not* a tuple, but an int defining the radius of V, 
  all the configurations are otherwise one-dimensional. 
  
  The actual shape of the inputs is is: (implicitly) batch size, and then (size_V,1), where this 1 is there for the currently unused one-hot encoding.
  Don't confuse the actual shape, with the input_shape variable, which is given by (size_V,), and with the input_shape positional argument
  of the keras Reshape layer.
  """

  def __init__(self, hidden_dim, input_shape=(2,), layer_sizes = None, hidden_activations = None, hidden_activations_L2_reg = 0, init_rule=None):
    """Constructs the convolutional net.
    
    Attributes:
    ws -- weights of the kernel

    Methods: 
    call() -- call the network as a function
    """

    super(ConvGraphMultiple3, self).__init__()

    # CURRENT VERSION: IGNORE THE ONE_HOT DIMENSION. CUT IT OUT, THEN BRING BACK IN THE END.
    #in_reshape = tf.keras.layers.Reshape((872,), input_shape=(872,1))
    
    in_reshape = tf.keras.layers.Reshape(target_shape = input_shape, input_shape=(input_shape[0],1))
    out_reshape = tf.keras.layers.Reshape((hidden_dim,1))
    
    aux_layers = []
    for layer_size in layer_sizes[:-1]:
        aux_layers += [tfkl.Dense(layer_size,kernel_initializer='random_normal',activation=hidden_activations,activity_regularizer=tf.keras.regularizers.L2(hidden_activations_L2_reg))]
    aux_layers += [tfkl.Dense(layer_sizes[-1],kernel_initializer='random_normal')]
    self.CGlayers = tf.keras.Sequential([in_reshape]+aux_layers+[out_reshape])
    print(layer_sizes)
    print(hidden_activations)
    print("Hidden dim: ", hidden_dim)
    print('Ver 3')


  def call(self, inputs):
    """Computes the dot product between the input and kernel weights.

    Keyword arguments:
    inputs -- tensor encoding the visible block to be coarse-grained
    """
    
    return self.CGlayers(inputs)
    

class CoarseGrainer(tf.keras.Model):
  def __init__(self, ll: tuple=None, size_V: int=None,
              hidden_dim: int=1, visible_dim: int=1,
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
    hidden_dim (int) -- number of components of the coarse-grained variable H
    visible_dim (int) -- number of components of the original degrees of freedom
    conv_activation (str) -- (nonlinear) activation function to map H (default tanh)
    Nq (int) -- number of states for a Potts degree of freedom (default None)
    h_embed (bool) -- embed H into a (pseudo) discrete valued variable (default False)
    init_rule -- initial conditions for the weights of the convolution net
    relaxation_rate (float) -- Gumbel-softmax rate for exponential annealing schedule
    min_temperature (float) -- minimum value for the Gumbel-softmax relaxation parameter
    init_temperature (float) -- initial value for the Gumbel-softmax relaxation parameter
    use_logits (bool) -- switch for treating the convolved values as logits
    use_probs (bool) -- switch for treating the convolved values as probabilities
    l1_reg (float) -- L1 regularisation strength for filter weights (via extra_kwargs)
    orthogonality_reg (float) -- orthogonality regularisation strength (via extra_kwargs)

    Functions and methods:
    call() -- call function: V -> H
    regularisation_loss() -- computes L1 + orthogonality penalty on filter weights
    global_step() -- updates the iteration index locally
    tau() -- anneals the Gumbel-softmax temperature parameter using the global iteration step
    """

    super(CoarseGrainer, self).__init__()

    self.Nq = Nq
    self.l1_reg = extra_kwargs.get('l1_reg', 0.0)
    self.orthogonality_reg = extra_kwargs.get('orthogonality_reg', 0.0)
    self.use_STE = extra_kwargs.get('use_STE', False)

    self._global_step = 0  # intialise the global iteration step in training

    # Determine filter type
    use_windowed = extra_kwargs.get('use_windowed_filters', False)
    use_gaussian = extra_kwargs.get('use_gaussian_filters', False)

    if isinstance(size_V,int):
        if extra_kwargs['nonlinearCG'] is None or extra_kwargs['nonlinearCG']==[0]:
            self.coarse_grainer = ConvGraphSingle(hidden_dim, visible_dim=visible_dim,
                                input_shape=(size_V,), init_rule=init_rule)
        else:
            self.coarse_grainer = ConvGraphMultiple3(hidden_dim, (size_V,), extra_kwargs['nonlinearCG'],extra_kwargs['hidden_activations'],extra_kwargs['hidden_activations_L2_reg'])
            # TODO: handle multicomponent variables in ConvGraphMultiple3
    elif len(ll) == 2: # i.e. if dimensionality (d) is 2
        if use_gaussian:
            self.coarse_grainer = Conv2DGaussian(
                hidden_dim, visible_dim=visible_dim,
                input_shape=ll, init_rule=init_rule,
                init_centers=extra_kwargs.get('init_centers', None),
                target_width=extra_kwargs.get('target_width', 20.0),
                min_width=extra_kwargs.get('min_width', 10.0),
                max_width=extra_kwargs.get('max_width', 30.0),
                use_sigmoid_centers=extra_kwargs.get('use_sigmoid_centers', True),
                allow_bipartite=extra_kwargs.get('allow_bipartite', False))
        elif use_windowed:
            self.coarse_grainer = Conv2DWindowed(
                hidden_dim, visible_dim=visible_dim,
                input_shape=ll, init_rule=init_rule,
                init_centers=extra_kwargs.get('init_centers', None),
                max_width=extra_kwargs.get('max_width', 25.0),
                adaptive_width=extra_kwargs.get('adaptive_width', False),
                window_tau=extra_kwargs.get('window_tau', 1.0),
                window_type=extra_kwargs.get('window_type', 'rectangular'),
                use_sigmoid_centers=extra_kwargs.get('use_sigmoid_centers', False))
        else:
            self.coarse_grainer = Conv2DSingle(hidden_dim, visible_dim=visible_dim,
                                              input_shape=ll, init_rule=init_rule)
    elif len(ll) == 3: # if d=3
        self.coarse_grainer = Conv3DSingle(hidden_dim, visible_dim=visible_dim,
                                          input_shape=ll, init_rule=init_rule)


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

        elif self.use_STE:
          # Straight-through estimator: forward pass discretises, backward passes gradients through
          self.method = 'STE quantisation'
          self.ste_layer = tfkl.Lambda(lambda x: _ste_sign(x))
          self._Λ = tf.keras.Sequential(
              [self.coarse_grainer, tfkl.Activation(conv_activation),
               self.ste_layer, tfkl.Flatten()])

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
          #self._Λ = tf.keras.Sequential([self.coarse_grainer,self.embedder])  # TODO: squeeze the output?
          self._Λ = tf.keras.Sequential([self.coarse_grainer,self.embedder])  

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

  def regularisation_loss(self):
    """Computes the regularisation loss on the coarse-graining filter weights.

    Returns L1 sparsity penalty + soft orthogonality penalty on the
    filter weights of the coarse_grainer layer. Returns 0 if both
    l1_reg and orthogonality_reg are zero or if the coarse_grainer
    does not have a ws attribute (e.g. nonlinear CG).
    """

    if (self.l1_reg == 0 and self.orthogonality_reg == 0):
      return tf.constant(0.0)
    if not hasattr(self.coarse_grainer, 'ws'):
      return tf.constant(0.0)

    return filter_regularisation_loss(
        self.coarse_grainer.ws,
        l1_reg=self.l1_reg,
        orthogonality_reg=self.orthogonality_reg)

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
