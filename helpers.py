#########################
# Helpers for the ABMs
#########################
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import truncnorm, uniform, norm
from scipy.optimize import minimize_scalar

def transform_percentage(max_x, x):
    if max_x == 1:
        return 0
    return 100 * (max_x - x) / (max_x - 1)

def get_distribution(dist_type, mu=55, sigma=15, lower=10, upper=100, **kwargs):
    if dist_type == 'truncnorm':
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        return truncnorm(a, b, loc=mu, scale=sigma)
    elif dist_type == 'uniform':
        return uniform(loc=lower, scale=upper - lower)
    elif dist_type == 'normal':
        return norm(loc=mu, scale=sigma)
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

def plot_all_agent_graphs(agent_data, color_dict, output_folder="plots", show_plots=True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Plot 1: Individual utilities
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
    
    # Plot 2: Individual consumptions
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
    
    # Plot 3: Average utilities
    plt.figure(figsize=(7, 5))
    sns.lineplot(
        data=agent_data,
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
    
    # Plot 4: Average consumptions
    plt.figure(figsize=(7, 5))
    sns.lineplot(
        data=agent_data,
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
    
    # Plot 5: Group shares
    N = agent_data['AgentID'].nunique()
    group_shares = agent_data.groupby(['Step', 'group'])['AgentID'].count().reset_index()
    group_shares['Share'] = group_shares['AgentID'] * 100 / N
    pivoted_shares = group_shares.pivot(index='Step', columns='group', values='Share').fillna(0)
    
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
