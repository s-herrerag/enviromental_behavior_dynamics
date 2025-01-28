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
        self.assigned_group = self.model.random.choice(
            ["Pro - environment", "Neutral", "Anti - environment"]
        )
        # Assign initial consumption
        self.consumption = consumption_dist.rvs(size=1)[0]

        # Store in node the information
        self.model.G[self.unique_id]["group"] = self.assigned_group
        self.model.G[self.unique_id]["consumption"] = self.consumption

    def calculate_status_classical(self, identities_neighbors = None, neighbors_consumption = None, consumption = None):
         """
         Each agent must look into their entry in the network. Then they compute their status. 
         In this run, status is calculated with the following steps: 
         1) Estimate the consumption of each neighbor
         2) Build the three rankings: pro, anti, neutral. 
         3) Find own rank. 
         4) Multiply status in each rank by the normalized number of neighbors in that rank.
         """
         if identities_neighbors is None or neighbors_consumption is None:
            identities_neighbors = [self.model.G[node]["group"] for node in self.model.G.neighbors(self.unique_id)]
            neighbors_consumption = [self.model.G[node]["consumption"] for node in self.model.G.neighbors(self.unique_id)]
            # And add as attribute
            self.identities_neighbors = identities_neighbors
            self.neighbors_consumption = neighbors_consumption

         if consumption is None:
             consumption = self.consumption

         array_consumptions = neighbors_consumption + [consumption]

         #References
         self.belief_max = max(array_consumptions)
         self.belief_min = min(array_consumptions)
         self.belief_median = np.percentile(array_consumptions, 50)
        
         #Ranks
         rpro = 100 - stats.percentileofscore(array_consumptions, array_consumptions)
         self.rpro_i = rpro[-1]
         ranti = stats.percentileofscore(array_consumptions, array_consumptions)
         self.ranti_i = ranti[-1]
         rneutral = 100 - stats.percentileofscore(np.abs(array_consumptions-self.belief_median), np.abs(array_consumptions-self.belief_median))
         self.rneutral_i = rneutral[-1]

         #Weights
         total_neighbors = len(identities_neighbors)
         self.wpro_i = identities_neighbors.count("Pro - environment")/total_neighbors
         self.wanti_i = identities_neighbors.count("Anti - environment")/total_neighbors
         self.wneutral_i = identities_neighbors.count("Neutral")/total_neighbors

         # Status
         status = self.rpro_i*self.wpro_i + self.ranti_i*self.wanti_i + self.rneutral_i*self.wneutral_i

         # Update if needed
         if identities_neighbors is None or neighbors_consumption is None:
             self.status = status
             
         return status
    
    
    def choose_consumption_classical(self):
        if self.status_strategy == "nontie":
            consumption_pro = self.belief_min - 1
            consumption_anti = self.belief_max + 1
            consumption_neutral = self.belief_median

            # Calculate status for each alternative
            status_pro = self.calculate_status_classical(consumption = consumption_pro, 
                                                         identities_neighbors=self.identities_neighbors, 
                                                         neighbors_consumption=self.neighbors_consumption)
            status_anti = self.calculate_status_classical(consumption = consumption_anti, 
                                                         identities_neighbors=self.identities_neighbors, 
                                                         neighbors_consumption=self.neighbors_consumption)
            status_neutral = self.calculate_status_classical(consumption = consumption_neutral, 
                                                         identities_neighbors=self.identities_neighbors, 
                                                         neighbors_consumption=self.neighbors_consumption)
        
        elif self.status_strategy == "tie":
            consumption_pro = self.belief_min
            consumption_anti = self.belief_max
            consumption_neutral = self.belief_median
            
            # Calculate status for each alternative
            status_pro = self.calculate_status_classical(consumption = consumption_pro, 
                                                         identities_neighbors=self.identities_neighbors,
                                                         neighbors_consumption=self.neighbors_consumption)
            status_anti = self.calculate_status_classical(consumption = consumption_anti, 
                                                         identities_neighbors=self.identities_neighbors,
                                                         neighbors_consumption=self.neighbors_consumption)
            status_neutral = self.calculate_status_classical(consumption = consumption_neutral, 
                                                         identities_neighbors=self.identities_neighbors,
                                                         neighbors_consumption=self.neighbors_consumption)

        ## Compare believed utilities in each group
        # Compute ideal 

        if self.assigned_group == "Pro - environment":
            ideal = consumption_pro
        elif self.assigned_group == "Anti - environment":
            ideal = consumption_anti
        else:
            ideal = consumption_neutral

        # Compute utility
        u_pro = self.lambda_s * status_pro - (1-self.lambda_s) * np.abs(consumption_pro - ideal)
        u_anti = self.lambda_s * status_anti - (1-self.lambda_s) * np.abs(consumption_anti - ideal)
        u_neutral = self.lambda_s * status_neutral - (1-self.lambda_s) * np.abs(consumption_neutral - ideal)

        # Choose consumption
        utilities = [u_pro, u_anti, u_neutral]
        highest_utility = np.argmax(utilities)

        if highest_utility == 0:
            self.consumption = consumption_pro
        elif highest_utility == 1:
            self.consumption = consumption_anti
        else:
            self.consumption = consumption_neutral
        

        def update_group(self):

            # Update group
            if self.consumption < self.belief_min:
                self.assigned_group = "Pro - environment"
            elif self.consumption > self.belief_max:
                self.assigned_group = "Anti - environment"
            else:
                self.assigned_group = "Neutral"