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
    def __init__(self, model, lambda_s=1, status_strategy="nontie"):
        super().__init__(model)
        self.lambda_s = lambda_s
        self.status_strategy = status_strategy
        # Randomly assign initial group
        self.assigned_group = self.model.random.choices(
            ["Pro - environment", "Neutral", "Anti - environment"],
            weights=[1/3, 1/3, 1/3]
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

    def calculate_status_classical(self, identities_neighbors=None, neighbors_consumption=None, consumption=None, update=True):
        if identities_neighbors is None or neighbors_consumption is None:
            if hasattr(self.model, "neighbor_sets"):
                neighbors_list = list(self.model.neighbor_sets[self.unique_id])
            else:
                neighbors_list = list(self.model.G.neighbors(self.unique_id))
            identities_neighbors = [self.model.agents_dict[node].assigned_group for node in neighbors_list]
            neighbors_consumption = [self.model.agents_dict[node].consumption for node in neighbors_list]
            self.identities_neighbors = identities_neighbors
            self.neighbors_consumption = neighbors_consumption

        consumption = self.consumption if consumption is None else consumption
        arr = np.array(neighbors_consumption + [consumption])
        belief_max = arr.max()
        belief_min = arr.min()
        belief_median = np.percentile(arr, 50)

        # Compute rankings for each scenario.
        rankings_pro = stats.rankdata(arr, method="dense")
        rpro_i = transform_percentage(rankings_pro.max(), rankings_pro[-1])
        rankings_anti = stats.rankdata(-arr, method="dense")
        ranti_i = transform_percentage(rankings_anti.max(), rankings_anti[-1])
        if len(arr) < 3:
            rankings_neutral = [1] * len(arr)
        else:
            rankings_neutral = stats.rankdata(np.abs(arr - belief_median), method="dense")
        rneutral_i = transform_percentage(max(rankings_neutral), rankings_neutral[-1])

        total_neighbors = len(identities_neighbors) if identities_neighbors else 1
        wpro_i = identities_neighbors.count("Pro - environment") / total_neighbors
        wanti_i = identities_neighbors.count("Anti - environment") / total_neighbors
        wneutral_i = identities_neighbors.count("Neutral") / total_neighbors

        status = rpro_i * wpro_i + ranti_i * wanti_i + rneutral_i * wneutral_i

        if update:
            self.status = status
            self.belief_max = belief_max
            self.belief_min = belief_min
            self.belief_median = belief_median
            self.rpro_i = rpro_i
            self.ranti_i = ranti_i
            self.rneutral_i = rneutral_i
            self.total_neighbors = total_neighbors
            self.identities_neighbors = identities_neighbors
            self.array_consumptions = arr
            self.rankings_pro = rankings_pro
            self.rankings_anti = rankings_anti
            self.rankings_neutral = rankings_neutral
        else:
            return status

    def choose_consumption_classical(self):
        if self.status_strategy == "nontie":
            self.consumption_pro = self.belief_min - 1
            self.consumption_anti = self.belief_max + 1
            self.consumption_neutral = self.belief_median
        elif self.status_strategy == "tie":
            self.consumption_pro = self.belief_min
            self.consumption_anti = self.belief_max
            self.consumption_neutral = self.belief_median

        status_pro = self.calculate_status_classical(consumption=self.consumption_pro,
                                                     identities_neighbors=self.identities_neighbors,
                                                     neighbors_consumption=self.neighbors_consumption,
                                                     update=False)
        status_anti = self.calculate_status_classical(consumption=self.consumption_anti,
                                                      identities_neighbors=self.identities_neighbors,
                                                      neighbors_consumption=self.neighbors_consumption,
                                                      update=False)
        status_neutral = self.calculate_status_classical(consumption=self.consumption_neutral,
                                                         identities_neighbors=self.identities_neighbors,
                                                         neighbors_consumption=self.neighbors_consumption,
                                                         update=False)
        self.status_pro = status_pro
        self.status_anti = status_anti
        self.status_neutral = status_neutral

        if self.assigned_group == "Pro - environment":
            ideal = self.consumption_pro
        elif self.assigned_group == "Anti - environment":
            ideal = self.consumption_anti
        else:
            ideal = self.consumption_neutral

        self.u_pro = self.lambda_s * status_pro - (1 - self.lambda_s) * np.abs(self.consumption_pro - ideal)
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

        if highest_utility == 0:
            self.consumption = self.consumption_pro
        elif highest_utility == 1:
            self.consumption = self.consumption_anti
        else:
            self.consumption = self.consumption_neutral

        self.highest_utility = highest_utility
        self.utility = utilities[highest_utility]

    def update_group(self):
        if self.highest_utility == 0:
            self.assigned_group = "Pro - environment"
        elif self.highest_utility == 1:
            self.assigned_group = "Anti - environment"
        else:
            self.assigned_group = "Neutral"

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
                rankings = stats.rankdata(common_consumptions, method="dense")
            elif j_agent.assigned_group == "Anti - environment":
                rankings = stats.rankdata([-val for val in common_consumptions], method="dense")
            elif j_agent.assigned_group == "Neutral":
                if len(common_consumptions) < 3:
                    rankings = [1] * len(common_consumptions)
                else:
                    median_value = np.median(common_consumptions)
                    differences = np.abs(np.array(common_consumptions) - median_value)
                    rankings = stats.rankdata(differences, method="dense")
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

    def choose_consumption_alternative(self):
        if self.status_strategy == "nontie":
            self.consumption_pro = self.belief_min - 1
            self.consumption_anti = self.belief_max + 1
            self.consumption_neutral = self.belief_median
            status_pro = self.calculate_status_alternative(consumption=self.consumption_pro, update=False)
            status_anti = self.calculate_status_alternative(consumption=self.consumption_anti, update=False)
            status_neutral = self.calculate_status_alternative(consumption=self.consumption_neutral, update=False)
        elif self.status_strategy == "tie":
            self.consumption_pro = self.belief_min
            self.consumption_anti = self.belief_max
            self.consumption_neutral = self.belief_median
            status_pro = self.calculate_status_alternative(consumption=self.consumption_pro, update=False)
            status_anti = self.calculate_status_alternative(consumption=self.consumption_anti, update=False)
            status_neutral = self.calculate_status_alternative(consumption=self.consumption_neutral, update=False)

        self.status_pro = status_pro
        self.status_anti = status_anti
        self.status_neutral = status_neutral

        if self.assigned_group == "Pro - environment":
            ideal = self.consumption_pro
        elif self.assigned_group == "Anti - environment":
            ideal = self.consumption_anti
        else:
            ideal = self.consumption_neutral

        self.u_pro = self.lambda_s * status_pro - (1 - self.lambda_s) * np.abs(self.consumption_pro - ideal)
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

        if highest_utility == 0:
            self.consumption = self.consumption_pro
        elif highest_utility == 1:
            self.consumption = self.consumption_anti
        else:
            self.consumption = self.consumption_neutral

        self.highest_utility = highest_utility
        self.utility = utilities[highest_utility]
