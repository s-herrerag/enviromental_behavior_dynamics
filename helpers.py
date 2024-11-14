#########################
# Helpers for the ABMs
#########################

import numpy as np
from scipy.stats import truncnorm, uniform, norm  # Import other distributions as needed
from scipy.optimize import minimize_scalar




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



### Estimation of the mode ------------------------
def calculate_mode_hist_midpoint(array, bins=10):
    """
    Calculate the mode of an array by finding the midpoint of the histogram bin with the highest frequency.

    Parameters:
    - array (list or numpy array): The input data.
    - bins (int): Number of bins to use for the histogram.

    Returns:
    - mode (float): The estimated mode as the midpoint of the most frequent bin.
    """
    if not array:
        return None  # Handle empty array

    counts, bin_edges = np.histogram(array, bins=bins)
    max_bin_index = np.argmax(counts)
    mode = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
    return mode


### Maximization of utility ------------------------

def g_anti(x):
    return x

def g_pro(x):
    return -x

def g_neutral(x):
    return 0

def maximize_utility(x_hat, g, lambda1, lambda2, s_i):
    # Define the original function u(x) using the passed g(x)
    """
    Find the value of x that maximizes the utility function u(x) = lambda1 * x + lambda2 * s_i + (1 - lambda1 - lambda2) * misalignment_cost,
    where misalignment_cost = -(x - x_hat)**2 + g(x).

    Parameters:
    - x_hat (float): The believed estimate of the distribution
    - g (callable): The function g(x) that is used in the misalignment cost
    - lambda1 (float): The weight of the linear term in the utility
    - lambda2 (float): The weight of the social influence term in the utility
    - s_i (float): The status

    Returns:
    - x_max (float): The value of x that maximizes the utility

    Raises:
    - ValueError: If the optimization fails
    """
    def u(x):
        misalignment_cost = - (x - x_hat)**2 + g(x)
        return lambda1 * x + lambda2 * s_i + (1 - lambda1 - lambda2) * misalignment_cost
    def neg_u(x):
        return -u(x)
    result = minimize_scalar(neg_u, bounds=(0, 20000), method='bounded')
    if result.success:
        x_max = result.x
        return x_max
    else:
        raise ValueError("Optimization failed: " + result.message)


