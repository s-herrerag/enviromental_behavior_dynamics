#######################################
# Status seeking agents
#######################################
import mesa
import numpy as np
from scipy import stats
import networkx as nx
from helpers import get_distribution, transform_percentage, rankdata_average

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

        self.is_leader = False
        self.pays_attention = False

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
        
    def gamma_introduce(self):
        if self.wait_gamma:
            self.gamma = 0
            if self.model.steps > self.model.memory * 2:
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

        self.field_of_action = {
            "Pro - environment": field_pro, 
            "Anti - environment": field_anti, 
            "Neutral": field_neutral
        }
    
    def rank_from_perspective(self, consumption_list, self_index, group_perspective):
        """
        Ranks consumption_list from viewpoint of 'group_perspective'.
        Returns transform_percentage(...) of the agent at self_index.
        """
        if group_perspective == "Pro - environment":
            # Use our custom rankdata_average instead of stats.rankdata
            rankings = rankdata_average(consumption_list)
        elif group_perspective == "Anti - environment":
            # Negate to rank higher consumption as lower
            rankings = rankdata_average([-val for val in consumption_list])
        else:  # "Neutral"
            if len(consumption_list) < 3:
                # Degenerate ranking if only 1 or 2 in the set
                rankings = [1] * len(consumption_list)
            else:
                median_val = np.median(consumption_list)
                diffs = np.abs(np.array(consumption_list) - median_val)
                rankings = rankdata_average(diffs)

        max_rank = np.max(rankings)
        self_rank = rankings[self_index]
        return transform_percentage(max_rank, self_rank)
    
    def compute_leader_status(self, hypothetical_consumption):
        """
        Computes agent i's status from perspective of *all* leaders in the model,
        weighting each leader's perspective by 1/(1 + dist(leader, i)).
        We now fetch BFS distances from self.model.distance_from_leader,
        rather than calling nx.shortest_path_length repeatedly.
        """
        leaders = [a for a in self.model.agents_list if a.is_leader]
        if not leaders:
            return 0  # No leaders => no contribution

        # i plus i's neighbors:
        if hasattr(self.model, "neighbor_sets"):
            my_neighbors = self.model.neighbor_sets[self.unique_id]
        else:
            my_neighbors = self.model.G.neighbors(self.unique_id)
        group_i = list(my_neighbors.union({self.unique_id}))

        sum_w = 0
        sum_status_w = 0

        for L in leaders:
            # Distance from i to L via cached BFS
            dist_iL = self.model.distance_from_leader.get((L.unique_id, self.unique_id), None)
            if dist_iL is None:
                # If no path, skip or treat as effectively 0 weight
                # (assuming your graph is connected, you can skip this check)
                continue
            w_L = 1 / (1 + dist_iL)

            # Build consumption array
            c_list = []
            for node in group_i:
                if node == self.unique_id:
                    c_list.append(hypothetical_consumption)
                else:
                    c_list.append(self.model.agents_dict[node].consumption)

            # Rank from L's perspective
            self_index = group_i.index(self.unique_id)
            rank_pct = self.rank_from_perspective(c_list, self_index, L.assigned_group)

            sum_status_w += w_L * rank_pct
            sum_w += w_L

        if sum_w > 0:
            return sum_status_w / sum_w
        else:
            return 0
    
    def calculate_status_alternative(self, consumption=None, update=True):
        consumption_i = self.consumption if consumption is None else consumption

        # Precompute neighbors_i_incl_self once
        if hasattr(self.model, "neighbor_sets"):
            neighbors_i = self.model.neighbor_sets[self.unique_id]
        else:
            neighbors_i = set(self.model.G.neighbors(self.unique_id))
        neighbors_i_incl_self = neighbors_i.union({self.unique_id})

        ranking_percentages = []

        for j_id in neighbors_i:
            j_agent = self.model.agents_dict[j_id]

            # Intersection with j's neighbors
            neighbors_j = self.model.neighbor_sets[j_id] if hasattr(self.model, "neighbor_sets") else set(self.model.G.neighbors(j_id))
            common_nodes_list = list(neighbors_i_incl_self.intersection(neighbors_j))

            try:
                index_self = common_nodes_list.index(self.unique_id)
            except ValueError:
                # If self isn't in the intersection, skip
                continue

            # Build consumption array for common_nodes_list
            common_consumptions = []
            for node_id in common_nodes_list:
                if node_id == self.unique_id:
                    common_consumptions.append(consumption_i)
                else:
                    common_consumptions.append(self.model.agents_dict[node_id].consumption)

            # Rank from j_agent's perspective
            if j_agent.assigned_group == "Pro - environment":
                rankings = rankdata_average(common_consumptions)
            elif j_agent.assigned_group == "Anti - environment":
                rankings = rankdata_average([-val for val in common_consumptions])
            else:  # "Neutral"
                if len(common_consumptions) < 3:
                    rankings = [1] * len(common_consumptions)
                else:
                    median_value = np.median(common_consumptions)
                    differences = np.abs(np.array(common_consumptions) - median_value)
                    rankings = rankdata_average(differences)

            max_rank = np.max(rankings)
            self_rank = rankings[index_self]
            rank_percentage = transform_percentage(max_rank, self_rank)
            ranking_percentages.append(rank_percentage)

        avg_status = np.mean(ranking_percentages) if ranking_percentages else 0

        # Potential effect of leadership
        if self.pays_attention:
            leader_status = self.compute_leader_status(consumption_i)
            final_status = self.model.beta * avg_status + (1 - self.model.beta) * leader_status
        else:
            final_status = avg_status

        if update:
            self.status = final_status
        else:
            return final_status

    def choose_consumption_alternative(self):
        # Determine "ideal" moves
        self.consumption_pro = self.consumption - self.field_of_action["Pro - environment"] - 1 
        self.consumption_anti = self.consumption + self.field_of_action["Anti - environment"] + 1

        diff = self.belief_median - self.consumption
        if np.abs(diff) <= self.field_of_action["Neutral"]:
            self.consumption_neutral = (self.belief_median) + self.random.uniform(-1,1)
        else:
            step_direction = 1 if diff > 0 else -1
            self.consumption_neutral = self.consumption + step_direction * self.field_of_action["Neutral"]

        # Calculate status for each hypothetical consumption
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
        if self.model.steps >= self.period_shock:
            self.u_pro = self.u_pro * 1 + np.abs(self.u_pro) * self.shock
        
        self.u_anti = self.lambda_s * status_anti - (1 - self.lambda_s) * np.abs(self.consumption_anti - ideal)
        self.u_neutral = self.lambda_s * status_neutral - (1 - self.lambda_s) * np.abs(self.consumption_neutral - ideal)

        utilities = [self.u_pro, self.u_anti, self.u_neutral] 
        highest_utilities = np.max(utilities)
        best_options = np.where(np.array(utilities) == highest_utilities)[0]

        if len(best_options) == 1:
            highest_utility = best_options[0]
        else:
            # Tie-breaking by "inertia" preference
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
        
        self.highest_utility = highest_utility
        self.utilities = utilities

    def update_consumption(self):
        if self.highest_utility == 0:
            self.consumption = self.consumption_pro
        elif self.highest_utility == 1:
            self.consumption = self.consumption_anti
        else:
            self.consumption = self.consumption_neutral

        self.utility = self.utilities[self.highest_utility]

    def update_group(self):
        if self.highest_utility == 0:
            self.assigned_group = "Pro - environment"
        elif self.highest_utility == 1:
            self.assigned_group = "Anti - environment"
        else:
            self.assigned_group = "Neutral"
