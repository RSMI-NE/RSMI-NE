import numpy as np
import matplotlib.pylab as plt
#from tqdm import tqdm
from tqdm.notebook import tqdm
import VBMI_estimators

class VBMI():
    """Interface for mutual information estimation by maximising variational lower bounds.

    :Author: 
        Doruk Efe Gökmen
    Date: 10/01/2021
    """
    
    def __init__(self, batch_size, layers=2, embed_dim=16, hidden_dim=64, activation='relu',
                 iterations=600, shuffle=0, learning_rate=5e-3, bound='infonce'):
        """
        Mutual information estimation by maximising variational lower bounds 
        represented by neural network ansätze.

        Attributes:
        batch_size (int) -- total number of samples in the dataset
        layers (int) -- number of hidden layers of the MLPs (default 2)
        embed_dim (int) -- embedding dimension of the separable ansatz (default 16)
        hidden_dim (int) -- hidden dimension of the dense layers in the MLP (default 64)
        activation (str) -- activation function of the neurons (default 'relu')
        iterations (int) -- number of epochs for training (default 600)
        shuffle (int) -- size of shuffled blocks of sample data (default 0: no shuffling)
        learning_rate (float) -- (default 5e-3)
        bound (str) -- type of mutual information lower-bound for estimation (default InfoNCE)

        Methods:
        InfoNCE() -- InfoNCE lower-bound
        DV() -- Donsker-Varadhan estimator
        """

        self.batch_size = batch_size
        self.iterations = iterations
        self.bound = bound

        self.critic_params = {
            'layers': layers,
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'activation': activation,
        }

        self.opt_params = {
            'batch_size': batch_size,
            'iterations': iterations,
            'shuffle': shuffle,
            'learning_rate': learning_rate,
        }

    def InfoNCE(self, x, y):
        """ Returns the InfoNCE estimate for I(X:Y).

        Keyword arguments:
        x, y -- sample datasets
        """

        return VBMI_estimators.train_estimator(x, y, self.critic_params, self.opt_params, self.bound)

    def DV(self, x, y):
        """ Returns the DV estimate for I(X:Y).

        Keyword arguments:
        x, y -- sample datasets
        """

        return VBMI_estimators.train_estimator(x, y, self.critic_params, self.opt_params, self.bound)


