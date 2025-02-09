#########################
# Helpers for the ABMs
#########################

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
    
### Plot and save main results ------------------------

def plot_all_agent_graphs(agent_data, color_dict, output_folder="plots", show_plots=True):
    """
    Plots and saves several graphs for the given agent data.

    Parameters:
        agent_data (DataFrame): DataFrame containing columns such as 'Step', 'Utility', 'Consumption', 'AgentID', and 'Group'.
        color_dict (dict): Dictionary mapping group names to colors.
        N (int): Total number of agents (used for normalizing counts to percentages).
        output_folder (str): Directory where the plots will be saved.
        show_plots (bool): If True, calls plt.show() to display the plots.
    """
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # -------------------------------
    # Plot 1: All agents' utilities over time
    plt.figure(figsize=(7, 5))
    sns.lineplot(
        data=agent_data,
        units="AgentID",
        x='Step',
        y='utility',
        hue='group',
        legend='full',
        linewidth=0.6,
        alpha=0.4,
        estimator=None,
        palette=color_dict
    )
    plt.title("All Agents' Utilities Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Utility (All agents)")
    plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "individual_utilities.pdf"), bbox_inches='tight', dpi=300)
    if show_plots:
        plt.show()
    plt.close()
    
    # -------------------------------
    # Plot 2: All agents' consumptions over time
    plt.figure(figsize=(7, 5))
    sns.lineplot(
        data=agent_data,
        units="AgentID",
        x='Step',
        y='consumption',
        hue='group',
        legend='full',
        linewidth=0.6,
        alpha=0.4,
        estimator=None,
        palette=color_dict
    )
    plt.title("All Agents' Consumptions Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Consumption (All agents)")
    plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "individual_consumptions.pdf"), bbox_inches='tight', dpi=300)
    if show_plots:
        plt.show()
    plt.close()
    
    # -------------------------------
    # Plot 3: Average utilities over time
    plt.figure(figsize=(7, 5))
    sns.lineplot(
        data=agent_data,
        units=None,
        x='Step',
        y='utility',
        hue='group',
        legend='full',
        linewidth=0.8,
        alpha=1,
        estimator="average",
        palette=color_dict
    )
    plt.title("Agents' Average Utilities Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Utility (Avg)")
    plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "average_utilities.pdf"), bbox_inches='tight', dpi=300)
    if show_plots:
        plt.show()
    plt.close()
    
    # -------------------------------
    # Plot 4: Average consumptions over time
    plt.figure(figsize=(7, 5))
    sns.lineplot(
        data=agent_data,
        units=None,
        x='Step',
        y='consumption',
        hue='group',
        legend='full',
        linewidth=0.6,
        alpha=1,
        estimator="average",
        palette=color_dict
    )
    plt.title("Agents' Average Consumptions Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Consumption (Avg)")
    plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "average_consumptions.pdf"), bbox_inches='tight', dpi=300)
    if show_plots:
        plt.show()
    plt.close()
    
    # -------------------------------
    # Plot 5: Group shares over time

    # N
    N = agent_data['AgentID'].nunique()

    # Count the number of individuals in each group at each Step
    group_shares = agent_data.groupby(['Step', 'group'])['AgentID'].count().reset_index()
    # Normalize the counts within each Step to get shares (percentage)
    group_shares['Share'] = group_shares['AgentID'].apply(lambda x: x * 100 / N)
    # Pivot the data to ensure every Step has an entry for each group (fill missing with 0)
    pivoted_shares = group_shares.pivot(index='Step', columns='group', values='Share').fillna(0)
    
    # Extract the series for each group. If a group is missing, default to zeros.
    steps_stack = pivoted_shares.index.tolist()
    anti_stack = pivoted_shares.get('Anti - environment', pd.Series(0, index=pivoted_shares.index)).tolist()
    neutral_stack = pivoted_shares.get('Neutral', pd.Series(0, index=pivoted_shares.index)).tolist()
    pro_stack = pivoted_shares.get('Pro - environment', pd.Series(0, index=pivoted_shares.index)).tolist()
    
    plt.figure(figsize=(7, 5))
    plt.stackplot(
        steps_stack,
        anti_stack,
        neutral_stack,
        pro_stack,
        labels=['Anti - environment', 'Neutral', 'Pro - environment'],
        colors=[
            color_dict.get('Anti - environment', 'blue'),
            color_dict.get('Neutral', 'gray'),
            color_dict.get('Pro - environment', 'green')
        ]
    )
    plt.title("Size of Groups Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Share (%)")
    plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "group_shares_noeffort.pdf"), bbox_inches='tight', dpi=300)
    if show_plots:
        plt.show()
    plt.close()

