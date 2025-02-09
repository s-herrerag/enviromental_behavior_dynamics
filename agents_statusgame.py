#######################################
# Status seeking agents
#######################################

# Libraries ------------------------
import mesa
import numpy as np
from scipy import stats
import networkx as nx

# Helpers ------------------------
from helpers import get_distribution
from helpers import transform_percentage

### Agent class ------------------------

consumption_dist = get_distribution(dist_type="uniform", lower=0, upper=100)

class statusgame_agent(mesa.Agent):
    """
    Agent with only status and identity
    """
    def __init__(self, model, lambda_s = 1, status_strategy = "nontie"):
        super().__init__(model)

        # Store parameters
        self.lambda_s = lambda_s
        self.status_strategy = status_strategy

        # Assign initial group randomly
        self.assigned_group = self.model.random.choices(["Pro - environment", "Neutral", "Anti - environment"], weights=[1/3, 1/3, 1/3])[0]
        # Assign initial consumption and status
        self.consumption = consumption_dist.rvs(size=1)[0]
        self.status = None

        # Store in node the information
        self.model.G.add_node(self.unique_id)

    def calculate_beliefs(self):
        neighbors_list = [n for n in self.model.G.neighbors(self.unique_id)]
        neighbors_consumption = [self.model.agents_dict[node].consumption for node in neighbors_list]
        array_consumptions = np.array(neighbors_consumption + [self.consumption]) 
        self.belief_max = np.max(array_consumptions)
        self.belief_min = np.min(array_consumptions)
        self.belief_median = np.percentile(array_consumptions, 50)

    def calculate_status_classical(self, identities_neighbors = None, neighbors_consumption = None, consumption = None, update = True):
         """
         Each agent must look into their entry in the network. Then they compute their status. 
         In this run, status is calculated with the following steps: 
         1) Estimate the consumption of each neighbor
         2) Build the three rankings: pro, anti, neutral. 
         3) Find own rank. 
         4) Multiply status in each rank by the normalized number of neighbors in that rank.
         """
         if identities_neighbors is None or neighbors_consumption is None:
            neighbors_list = [n for n in self.model.G.neighbors(self.unique_id)]
            identities_neighbors = [self.model.agents_dict[node].assigned_group for node in neighbors_list]
            neighbors_consumption = [self.model.agents_dict[node].consumption for node in neighbors_list] 
            # And add as attribute
            self.identities_neighbors = identities_neighbors
            self.neighbors_consumption = neighbors_consumption

         if consumption is None:
             consumption = self.consumption

         array_consumptions = np.array(neighbors_consumption + [consumption])

         #References
         belief_max = np.max(array_consumptions)
         belief_min = np.min(array_consumptions)
         belief_median = np.percentile(array_consumptions, 50)
        
         #Ranks
         # Pro
         rankings_pro = stats.rankdata(array_consumptions, method="dense")
         rpro_i = transform_percentage(max(rankings_pro), rankings_pro[-1])
         # Anti 
         rankings_anti = stats.rankdata(-array_consumptions, method="dense")
         ranti_i = transform_percentage(max(rankings_anti), rankings_anti[-1])

         # Neutral 
         # Neutral requires at least three items
         if len(array_consumptions) < 3:
            rankings_neutral = [1,1]
         else:
            rankings_neutral = stats.rankdata(np.abs(array_consumptions-belief_median), method="dense")
         rneutral_i = transform_percentage(max(rankings_neutral), rankings_neutral[-1])

         #Weights
         total_neighbors = len(identities_neighbors)
         wpro_i = identities_neighbors.count("Pro - environment")/total_neighbors
         wanti_i = identities_neighbors.count("Anti - environment")/total_neighbors
         wneutral_i = identities_neighbors.count("Neutral")/total_neighbors

         # Status
         status = rpro_i*wpro_i + ranti_i*wanti_i +rneutral_i*wneutral_i

         # Update if needed
         if update == True:
             self.status = status
             self.belief_max = belief_max
             self.belief_min = belief_min
             self.belief_median = belief_median
             self.rpro_i = rpro_i
             self.ranti_i = ranti_i
             self.rneutral_i = rneutral_i
             #self.wpro_i = wpro_i
             #self.wanti_i = wanti_i
             #self.wneutral_i = wneutral_i
             self.total_neighbors = total_neighbors
             self.identities_neighbors = identities_neighbors
             self.array_consumptions = array_consumptions
             self.rankings_neutral = rankings_neutral
             self.rankings_pro = rankings_pro
             self.rankings_anti = rankings_anti
         else:
            return status
    
    
    def choose_consumption_classical(self):
        if self.status_strategy == "nontie":
            self.consumption_pro = self.belief_min - 1
            self.consumption_anti = self.belief_max + 1
            self.consumption_neutral = self.belief_median

            # Calculate status for each alternative
            status_pro = self.calculate_status_classical(consumption = self.consumption_pro, 
                                                         identities_neighbors=self.identities_neighbors, 
                                                         neighbors_consumption=self.neighbors_consumption, update=False)
            

            status_anti = self.calculate_status_classical(consumption = self.consumption_anti, 
                                                         identities_neighbors=self.identities_neighbors, 
                                                         neighbors_consumption=self.neighbors_consumption, update=False)
            status_neutral = self.calculate_status_classical(consumption = self.consumption_neutral, 
                                                         identities_neighbors=self.identities_neighbors, 
                                                         neighbors_consumption=self.neighbors_consumption, update=False)
            
        elif self.status_strategy == "tie":
            self.consumption_pro = self.belief_min
            self.consumption_anti = self.belief_max
            self.consumption_neutral = self.belief_median
            
            # Calculate status for each alternative
            status_pro = self.calculate_status_classical(consumption = self.consumption_pro, 
                                                         identities_neighbors=self.identities_neighbors,
                                                         neighbors_consumption=self.neighbors_consumption, update=False)
            status_anti = self.calculate_status_classical(consumption = self.consumption_anti, 
                                                         identities_neighbors=self.identities_neighbors,
                                                         neighbors_consumption=self.neighbors_consumption, update=False)
            status_neutral = self.calculate_status_classical(consumption = self.consumption_neutral, 
                                                         identities_neighbors=self.identities_neighbors,
                                                         neighbors_consumption=self.neighbors_consumption, update=False)
        
        self.status_pro = status_pro
        self.status_anti = status_anti
        self.status_neutral = status_neutral

        ## Compare believed utilities in each group
        # Compute ideal 

        if self.assigned_group == "Pro - environment":
            ideal = self.consumption_pro
        elif self.assigned_group == "Anti - environment":
            ideal = self.consumption_anti
        else:
            ideal = self.consumption_neutral

        # Compute utility
        self.u_pro = self.lambda_s * status_pro - (1-self.lambda_s) * np.abs(self.consumption_pro - ideal)
        self.u_anti = self.lambda_s * status_anti - (1-self.lambda_s) * np.abs(self.consumption_anti - ideal)
        self.u_neutral = self.lambda_s * status_neutral - (1-self.lambda_s) * np.abs(self.consumption_neutral - ideal)

        # Choose consumption
        utilities = [self.u_pro, self.u_anti, self.u_neutral]
        highest_utilities = np.max(utilities)
        best_options = np.where(utilities == highest_utilities)[0]

        # The only important thing is to define the tiebreaking rules. I will assume agents have inertia if two / more options have the same utility
        # When it is impossible to keep up one's behavior, randomly choose. 

        if len(best_options) == 1:
            highest_utility = best_options[0]
        else:
            # Inertia
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

        # Finally, choose consumption
        if highest_utility == 0:
            self.consumption = self.consumption_pro
        elif highest_utility == 1:
            self.consumption = self.consumption_anti
        else:
            self.consumption = self.consumption_neutral
        
        # Store as attribute the highest utility
        self.highest_utility = highest_utility
        self.utility = utilities[highest_utility]

    def update_group(self):
        # Update group
        if self.highest_utility == 0:
            self.assigned_group = "Pro - environment"
        elif self.highest_utility == 1:
            self.assigned_group = "Anti - environment"
        else:
            self.assigned_group = "Neutral"


    # There is an alternative way of defining the status ------------------------
    def calculate_status_alternative(self, consumption=None, update=True):
        """
        Alternative status calculation.
        
        Parameters:
            consumption (float, optional): The consumption value to use for agent i (self).
                                           If not provided, self.consumption is used.
            update (bool): Whether to update self.status with the computed value.

        For each neighbor j, the agent:
          1) Identifies the set of common nodes between itself and neighbor j. (Here, we include self in the common set.)
          2) Retrieves the consumption values for the common nodes. For self, the provided consumption value
             is used if given.
          3) Uses neighbor j’s identity (assigned group) to rank the common consumption values:
             - "Pro - environment": Direct ranking.
             - "Anti - environment": Ranking of the negative of the consumption values.
             - "Neutral": Ranking based on the absolute difference from the median of the common set.
          4) Finds its own rank in j’s ranking and transforms it into a percentage.
          5) After processing all neighbors, averages these percentages to set its status.
        """
        # Use the provided consumption for agent i, if given; otherwise, use self.consumption.
        consumption_i = self.consumption if consumption is None else consumption

        # Get agent i's neighbors (as a set of IDs)
        neighbors_i = set(self.model.G.neighbors(self.unique_id))
        ranking_percentages = []
        
        # Iterate over each neighbor j
        for j_id in neighbors_i:
            j_agent = self.model.agents_dict[j_id]
            # Get neighbor j's neighbors (as a set)
            neighbors_j = set(self.model.G.neighbors(j_id))
            # Define common nodes as those in both: (neighbors of i plus self) and neighbors of j.
            common_nodes = (neighbors_i.union({self.unique_id})).intersection(neighbors_j)
            common_nodes_list = list(common_nodes)
            
            # Build the list of consumption values for the common nodes.
            # For self, use consumption_i (which might be different from self.consumption).
            common_consumptions = [
                consumption_i if node_id == self.unique_id 
                else self.model.agents_dict[node_id].consumption
                for node_id in common_nodes_list
            ]
            
            # Find self's index in the common nodes list.
            try:
                index_self = common_nodes_list.index(self.unique_id)
            except ValueError:
                # This should not occur, but skip this neighbor if it does.
                continue
            
            # Rank the consumption values according to neighbor j's identity.
            if j_agent.assigned_group == "Pro - environment":
                # Direct ranking: lower consumption gets a lower rank.
                rankings = stats.rankdata(common_consumptions, method="dense")
            elif j_agent.assigned_group == "Anti - environment":
                # Reverse ranking: higher consumption gets a lower rank.
                rankings = stats.rankdata([-val for val in common_consumptions], method="dense")
            elif j_agent.assigned_group == "Neutral":
                # Neutral ranking: use the absolute difference from the median.
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
        
        # Compute the average ranking percentage.
        if ranking_percentages:
            avg_status = np.mean(ranking_percentages)
        else:
            # If no neighbors or valid rankings are found, default to 0.
            print("No valid rankings found for agent ", self.unique_id)
            avg_status = 0
        
        if update:
            self.status = avg_status
        else:
            return avg_status
        
    def choose_consumption_alternative(self):
        if self.status_strategy == "nontie":
            self.consumption_pro = self.belief_min - 1
            self.consumption_anti = self.belief_max + 1
            self.consumption_neutral = self.belief_median

            # Calculate status for each alternative
            status_pro = self.calculate_status_alternative(consumption = self.consumption_pro, update=False)
            

            status_anti = self.calculate_status_alternative(consumption = self.consumption_anti, update=False)
            status_neutral = self.calculate_status_alternative(consumption = self.consumption_neutral, update=False)
            
        elif self.status_strategy == "tie":
            self.consumption_pro = self.belief_min
            self.consumption_anti = self.belief_max
            self.consumption_neutral = self.belief_median
            
            # Calculate status for each alternative
            status_pro = self.calculate_status_alternative(consumption = self.consumption_pro, update=False)
            status_anti = self.calculate_status_alternative(consumption = self.consumption_anti, update=False)
            status_neutral = self.calculate_status_alternative(consumption = self.consumption_neutral, update=False)
        
        self.status_pro = status_pro
        self.status_anti = status_anti
        self.status_neutral = status_neutral

        ## Compare believed utilities in each group
        # Compute ideal 

        if self.assigned_group == "Pro - environment":
            ideal = self.consumption_pro
        elif self.assigned_group == "Anti - environment":
            ideal = self.consumption_anti
        else:
            ideal = self.consumption_neutral

        # Compute utility
        self.u_pro = self.lambda_s * status_pro - (1-self.lambda_s) * np.abs(self.consumption_pro - ideal)
        self.u_anti = self.lambda_s * status_anti - (1-self.lambda_s) * np.abs(self.consumption_anti - ideal)
        self.u_neutral = self.lambda_s * status_neutral - (1-self.lambda_s) * np.abs(self.consumption_neutral - ideal)

        # Choose consumption
        utilities = [self.u_pro, self.u_anti, self.u_neutral]
        highest_utilities = np.max(utilities)
        best_options = np.where(utilities == highest_utilities)[0]

        # The only important thing is to define the tiebreaking rules. I will assume agents have inertia if two / more options have the same utility
        # When it is impossible to keep up one's behavior, randomly choose. 

        if len(best_options) == 1:
            highest_utility = best_options[0]
        else:
            # Inertia
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

        # Finally, choose consumption
        if highest_utility == 0:
            self.consumption = self.consumption_pro
        elif highest_utility == 1:
            self.consumption = self.consumption_anti
        else:
            self.consumption = self.consumption_neutral
        
        # Store as attribute the highest index
        self.highest_utility = highest_utility
        self.utility = utilities[highest_utility]



        
        