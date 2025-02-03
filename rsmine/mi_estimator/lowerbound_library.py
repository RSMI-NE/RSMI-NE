# Authors: Smayan Khanna and Doruk Efe Gökmen
# Date: 03/02/2025

# NOTE filename used to be `VBMI_bounds.py` but was changed to `lowerbound_library.py`

import torch
from rsmine.mi_estimator.utils import logmeanexp_offdiag, logmeanexp_masked


def dv_upper_lower_bound(x, y, f_ansatz):
    """Donsker-Varadhan estimator for I(X:Y).

    Args:
        x (torch.tensor): full sample dataset for random variable X
        y (torch.tensor): full sample dataset for random variable Y
        f_ansatz (function): critic function
    """

    scores = f_ansatz(x, y)
    batch_size = scores.shape[0]

    positive_mask = torch.eye(batch_size, dtype=bool)

    joint_term = torch.mean(torch.masked_select(scores, positive_mask), dim=-1)
    return joint_term - logmeanexp_offdiag(scores)


def infonce_lower_bound(x, y, f_ansatz):
    """InfoNCE replicated estimator for I(X:Y) (van den Oord et al. 2018).
    
    Author: Smayan Khanna
    
    This code was inspired by the implementation in the TensorFlow Probability library.

    Args:
        x (torch.tensor): full sample dataset for random variable X
        y (torch.tensor): full sample dataset for random variable Y
        f_ansatz (function): critic function
    """
    
    scores = f_ansatz(x, y)
    batch_size = scores.shape[0]

    positive_mask = torch.eye(batch_size, dtype=bool)

    log_n = torch.log(torch.tensor(scores.shape[-1], dtype=scores.dtype))

    joint_term = torch.mean(torch.masked_select(scores, positive_mask), dim=-1)
    marginal_term = torch.mean(torch.logsumexp(scores, dim=-1), dim=-1) - log_n

    return joint_term - marginal_term


def nwj_lower_bound(x, y, f_ansatz):
    """Nguyen-Wainwright-Jordan lower bound for I(X:Y).
    Equivalent to TUBA bound with consant baseline.

    Args:
        x (torch.tensor): full sample dataset for random variable X
        y (torch.tensor): full sample dataset for random variable Y
        f_ansatz (function): critic function
    """

    scores = f_ansatz(x, y)
    batch_size = scores.shape[0]

    positive_mask = torch.eye(batch_size, dtype=bool)
    negative_mask = ~positive_mask

    joint_term = torch.mean(torch.masked_select(scores, positive_mask), dim=-1)
    
    marginal_term = torch.exp(logmeanexp_masked(scores, negative_mask, axis=[-2, -1]) - 1.0)

    return joint_term - marginal_term


def tuba_lower_bound(x, y, f_ansatz, a_ansatz):
    """Tractable unnormalized Barber-Agakov lower bound for I(X:Y).

    Args:
        x (torch.tensor): full sample dataset for random variable X
        y (torch.tensor): full sample dataset for random variable Y
        f_ansatz (function): critic function
        a_ansatz (function): baseline function
    """
    #TODO: implement
    raise NotImplementedError("TUBA lower bound not implemented yet.")


def mine_lower_bound(x, y, f_ansatz, a_ansatz):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        f_ansatz (_type_): _description_
        a_ansatz (_type_): _description_
    """
    # TODO: implement
    raise NotImplementedError("MINE bound not implemented yet.")


lowerbounds = {
    "infonce": infonce_lower_bound,
    "dv": dv_upper_lower_bound,
    "nwj": nwj_lower_bound,
    "tuba": tuba_lower_bound,
    "mine": mine_lower_bound,
}
