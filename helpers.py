#########################
# Helpers for the ABMs
#########################

import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import minimize_scalar


### Distribution of initial consumption ------------------------
mu = 55   # Mean
sigma = 15  # Standard deviation
lower, upper = 10, 100 # Lower and upper bounds

# Calculate the a and b parameters for truncnorm
a, b = (lower - mu) / sigma, (upper - mu) / sigma

# Truncated normal distribution
trunc_normal_dist = truncnorm(a, b, loc=mu, scale=sigma)

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
    return -x

def g_pro(x):
    return x

def g_neutral(x):
    return 0

def maximize_utility(alpha, x_hat, g):
    """
    Finds the maximum of f(x) = alpha * x - (x - x_hat)^2 - g(x) numerically.
    
    Parameters:
    - alpha (float): The coefficient of x in the function.
    - x_hat (float): The reference value in the quadratic term.
    - g (callable): A function representing g(x). It should take a single argument x.
    
    Returns:
    - x_max (float): The value of x that maximizes f(x).
    - f_max (float): The maximum value of f(x).
    
    Raises:
    - ValueError: If the optimization fails.
    """
    # Define the original function f(x) using the passed g(x)
    def f(x):
        return alpha * x - (x - x_hat)**2 - g(x)
    
    # Since scipy.optimize minimizes, define the negative of f(x)
    def neg_f(x):
        return -f(x)
    
    # Perform the optimization
    result = minimize_scalar(neg_f)
    
    if result.success:
        x_max = result.x
        return x_max
    else:
        raise ValueError("Optimization failed: " + result.message)


