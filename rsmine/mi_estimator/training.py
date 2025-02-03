# Authors: Doruk Efe Gökmen
# Date: 03/02/2025

# NOTE: filename used to be `VBMI_estimators.py` but was changed to `training.py`

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm

from rsmine.mi_estimator.lowerbound_library import lowerbounds
from rsmine.mi_estimator.critics_baselines import SeparableCritic


log2 = np.log(2)

def train_estimator(X, Y, critic_params, opt_params, bound="infonce"):
    """Main training loop to estimate MI.

    Keyword arguments:
    X -- full dataset for the random variable X (torch.Tensor or np.ndarray)
    Y -- full dataset for the random variable Y (torch.Tensor or np.ndarray)
    critic_params (dict) -- set of parameters for the ansatz/critic function
    opt_params (dict) -- set of parameters for the optimiser
    bound (str) -- mutual information lower-bound (default InfoNCE)
    """

    # initialize critic
    f_ansatz = SeparableCritic(**critic_params)
    optimizer = torch.optim.Adam(f_ansatz.parameters(), lr=opt_params["learning_rate"])

    # If input data is a NumPy array, convert it to torch.Tensor
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float)
    if not torch.is_tensor(Y):
        Y = torch.tensor(Y, dtype=torch.float)

    num_samples = X.shape[0]
    dataset = TensorDataset(Y, X) 
    loader = DataLoader(
        dataset, batch_size=opt_params["batch_size"], shuffle=opt_params["shuffle"]
    )

    total_iterations = opt_params["iterations"] * (num_samples // opt_params["batch_size"])
    pbar = tqdm(total=total_iterations, desc="")
    estimates = []

    for epoch in range(opt_params["iterations"]):
        for y, x in loader:
            optimizer.zero_grad() # reset gradients
            mi = lowerbounds[bound](x, y, f_ansatz)
            cost = -mi
            cost.backward()
            optimizer.step()

            estimates.append(mi.item())
            pbar.set_description(f"I={mi.item()/log2:.2f} bits")
            pbar.update(1)

    return np.array(estimates)
