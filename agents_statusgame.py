#######################################
# Status seeking agents
#######################################
import mesa
import numpy as np
from scipy import stats
import networkx as nx 
from helpers import get_distribution, transform_percentage

# Distribution for initial consumption
consumption_dist = get_distribution(dist_type="uniform", lower=0, upper=100)

class statusgame_agent(mesa.Agent):
    """
    Agent with only status and identity.
    """
    def __init__(self, model, lambda_s=1, gamma=1, wait_gamma=False,
                 shock = 0, period_shock = 0, 
                 weights = [1/3, 1/3, 1/3]):
        
        super().__init__(model)
        self.lambda_s = lambda_s
        self.initgamma = gamma
        self.shock = shock
        self.period_shock = period_shock
        self.wait_gamma = wait_gamma

        # Randomly assign initial group
        self.assigned_group = self.model.random.choices(
            ["Pro - environment", "Neutral", "Anti - environment"],
            weights=weights
        )[0]
        self.consumption = consumption_dist.rvs(size=1)[0]
        self.status = None

    def calculate_beliefs(self):
        # Use precomputed neighbor sets if available.
        if hasattr(self.model, "neighbor_sets"):
            neighbors_list = list(self.model.neighbor_sets[self.unique_id])
        else:
            neighbors_list = list(self.model.G.neighbors(self.unique_id))
        neighbors_consumption = [self.model.agents_dict[node].consumption for node in neighbors_list]
        arr = np.array(neighbors_consumption + [self.consumption])
        self.belief_max = arr.max()
        self.belief_min = arr.min()
        self.belief_median = np.percentile(arr, 50)

    def calculate_status_alternative(self, consumption=None, update=True):
        consumption_i = self.consumption if consumption is None else consumption
        # Use precomputed neighbor sets if available.
        if hasattr(self.model, "neighbor_sets"):
            neighbors_i = set(self.model.neighbor_sets[self.unique_id])
        else:
            neighbors_i = set(self.model.G.neighbors(self.unique_id))
        ranking_percentages = []

        n_common = []
        for j_id in neighbors_i:
            j_agent = self.model.agents_dict[j_id]
            neighbors_j = set(self.model.G.neighbors(j_id))
            common_nodes = (neighbors_i.union({self.unique_id})).intersection(neighbors_j)
            common_nodes_list = list(common_nodes)
            n_common.append(len(common_nodes_list))
            try:
                index_self = common_nodes_list.index(self.unique_id)
            except ValueError:
                continue
            common_consumptions = [
                consumption_i if node_id == self.unique_id
                else self.model.agents_dict[node_id].consumption
                for node_id in common_nodes_list
            ]
            if j_agent.assigned_group == "Pro - environment":
                rankings = stats.rankdata(common_consumptions, method="average")
            elif j_agent.assigned_group == "Anti - environment":
                rankings = stats.rankdata([-val for val in common_consumptions], method="average")
            elif j_agent.assigned_group == "Neutral":
                if len(common_consumptions) < 3:
                   rankings = [1] * len(common_consumptions)
                else:
                    median_value = np.median(common_consumptions)
                    differences = np.abs(np.array(common_consumptions) - median_value)
                    rankings = stats.rankdata(differences, method="average")
            max_rank = np.max(rankings)
            self_rank = rankings[index_self]
            rank_percentage = transform_percentage(max_rank, self_rank)
            ranking_percentages.append(rank_percentage)

        avg_status = np.mean(ranking_percentages) if ranking_percentages else 0
        if update:
            self.status = avg_status
            try:
                n_common = [0 if x is None else x for x in n_common]
            except TypeError:
                n_common = [0]
            self.n_common = np.mean(n_common)
        else:
            return avg_status
        
    def gamma_introduce(self):
        if self.wait_gamma:
            self.gamma = 0
            if self.model.steps >= self.model.memory:
                self.gamma = self.initgamma
                self.wait_gamma = False
        else:
            self.gamma = self.initgamma
        
    def calculate_field_of_action(self):
        # Distances
        l_pro = np.abs(self.consumption - self.belief_min) 
        l_anti = np.abs(self.consumption - self.belief_max) 
        l_neutral = np.abs(self.consumption - self.belief_median)
        
        # Fields
        field_pro = self.gamma * l_pro
        field_anti = self.gamma * l_anti
        field_neutral = self.gamma * l_neutral

        self.field_of_action = {"Pro - environment":field_pro, 
                                "Anti - environment":field_anti, 
                                "Neutral":field_neutral}

    def choose_consumption_alternative(self):
        # Define the ideal action for each agent, which depends on the field of action
        self.consumption_pro = self.consumption - self.field_of_action["Pro - environment"] - 1 
        self.consumption_anti = self.consumption + self.field_of_action["Anti - environment"] + 1
        
        # Calculate the difference between the median and the current consumption
        diff = self.belief_median - self.consumption
        # If the difference is within the allowed field, move exactly to the median.
        # Otherwise, move by the maximum allowed amount in the appropriate direction.
        if np.abs(diff) <= self.field_of_action["Neutral"]:
            self.consumption_neutral = (self.belief_median) + self.random.uniform(-1,1) # Prevents too many ties
        else:
            # Determine the direction: +1 if we need to increase, -1 if decrease.
            step_direction = 1 if diff > 0 else -1
            self.consumption_neutral = self.consumption + step_direction * self.field_of_action["Neutral"]

        # Calculate the status for each alternative
        status_pro = self.calculate_status_alternative(consumption=self.consumption_pro, update=False)
        status_anti = self.calculate_status_alternative(consumption=self.consumption_anti, update=False)
        status_neutral = self.calculate_status_alternative(consumption=self.consumption_neutral, update=False)
        self.status_pro = status_pro
        self.status_anti = status_anti
        self.status_neutral = status_neutral

        # Calculate utility
        if self.assigned_group == "Pro - environment":
            ideal = self.consumption_pro
        elif self.assigned_group == "Anti - environment":
            ideal = self.consumption_anti
        else:
            ideal = self.consumption_neutral

        self.u_pro = self.lambda_s * status_pro - (1 - self.lambda_s) * np.abs(self.consumption_pro - ideal)

        # Add shock
        if self.model.steps >= self.period_shock:
            self.u_pro = self.u_pro * 1 + np.abs(self.u_pro) * self.shock # Ensure that the shock is positive
        
        self.u_anti = self.lambda_s * status_anti - (1 - self.lambda_s) * np.abs(self.consumption_anti - ideal)
        self.u_neutral = self.lambda_s * status_neutral - (1 - self.lambda_s) * np.abs(self.consumption_neutral - ideal)

        utilities = [self.u_pro, self.u_anti, self.u_neutral] 
        highest_utilities = np.max(utilities)
        best_options = np.where(np.array(utilities) == highest_utilities)[0]

        if len(best_options) == 1:
            highest_utility = best_options[0]
        else:
            if ideal == self.consumption_pro:
                inertia_option_index = 0
            elif ideal == self.consumption_anti:
                inertia_option_index = 1
            else:
                inertia_option_index = 2
            if inertia_option_index in best_options:
                highest_utility = inertia_option_index
            else:
                highest_utility = self.model.random.choice(best_options)
        
        # Store info
        self.highest_utility = highest_utility
        self.utilities = utilities

        # Method to choose consumption based on the highest utility

    def update_consumption(self):

        if self.highest_utility == 0:
            self.consumption = self.consumption_pro
        elif self.highest_utility == 1:
            self.consumption = self.consumption_anti
        else:
            self.consumption = self.consumption_neutral

        self.highest_utility = self.highest_utility
        self.utility = self.utilities[self.highest_utility]

    def update_group(self):
        if self.highest_utility == 0:
            self.assigned_group = "Pro - environment"
        elif self.highest_utility == 1:
            self.assigned_group = "Anti - environment"
        else:
            self.assigned_group = "Neutral"