#######################################
# Status seeking and coordination game
#######################################

### Libraries ------------------------
import mesa
import numpy as np
from mesa.datacollection import DataCollector
from scipy import stats
import math
import networkx as nx
from mesa.time import BaseScheduler

### Helpers ------------------------
from helpers import get_distribution
from helpers import g_pro, g_neutral, g_anti, maximize_utility

### Agents ------------------------

consumption_dist = get_distribution(dist_type="uniform", lower=0, upper=100)

class status_identity_agent(mesa.Agent):
    """
    Agent with only status and identity
    """
    #Initialize agent
    def __init__(self, model, 
                 lambda_s = 1, status_strategy = "nontie"):
        super().__init__(model)

        # Store parameters
        self.lambda_s = lambda_s
        self.status_strategy = status_strategy

        # Assign initial group randomly
        self.assigned_group = self.model.random.choice(
            ["Pro - environment", "Neutral", "Anti - environment"]
        )
        # Assign initial consumption
        self.history = [self.assigned_group]
        c0 = consumption_dist.rvs(size=1)[0]
        self.history = [c0]










class status_identity_model(mesa.Model):
        def __init__(self, N):
            """
            Create N agents and manipulate a graph where they will interact
            """
            self.num_agents = N
            
            # Simple scheduler - you can switch to other schedulers if needed
            self.schedule = BaseScheduler(self)
            
            # Create an empty Graph in which agents will be nodes
            self.G = nx.Graph()

            # Create agents, add them to scheduler, and add them as nodes in the graph
            for i in range(self.num_agents):
                agent = PairAgent(i, self)
                self.schedule.add(agent)
                self.G.add_node(agent)


# En cada step del modelooo, armar los edges