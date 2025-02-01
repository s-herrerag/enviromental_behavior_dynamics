#########################
# Helpers for the ABMs
#########################

import numpy as np
from scipy.stats import truncnorm, uniform, norm  # Import other distributions as needed
from scipy.optimize import minimize_scalar


#### Transform rankings to percentages ------------------------

def transform_percentage(max_x,x):
    if max_x == 1:
        return 0
    return 100 * (max_x-x) / (max_x-1)

### Distribution of initial consumption ------------------------

def get_distribution(dist_type, mu=55, sigma=15, lower=10, upper=100, **kwargs):
    """
    Factory function to create different distribution objects.

    Parameters:
    - dist_type (str): Type of distribution ('truncnorm', 'uniform', 'normal', etc.)
    - mu (float): Mean of the distribution (used for normal and truncnorm)
    - sigma (float): Standard deviation (used for normal and truncnorm)
    - lower (float): Lower bound (used for truncnorm and uniform)
    - upper (float): Upper bound (used for truncnorm and uniform)
    - **kwargs: Additional keyword arguments for specific distributions

    Returns:
    - A scipy.stats distribution object
    """
    if dist_type == 'truncnorm':
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        return truncnorm(a, b, loc=mu, scale=sigma)
    elif dist_type == 'uniform':
        return uniform(loc=lower, scale=upper - lower)
    elif dist_type == 'normal':
        return norm(loc=mu, scale=sigma)
    # Add more distributions as needed
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")
