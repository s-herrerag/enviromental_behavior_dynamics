#########################################
# Model class of the status identity game
#########################################
import mesa
import numpy as np
from mesa.datacollection import DataCollector
from scipy import stats
import math
import networkx as nx

from agents_statusgame import statusgame_agent

class statusgame_model(mesa.Model):
    """
    Create N agents and manipulate a graph where they will interact
    """
    def __init__(self, N, seed=None, lambda_s = 1, status_strategy = "nontie", memory = 10):
        super().__init__(seed=seed)
        self.num_agents = N
        self.memory = memory #Param for deleting edges

        # Create an empty Graph in which agents will be nodes
        self.G = nx.Graph()
        
        # Create agents 
        statusgame_agent.create_agents(model=self, n=N, lambda_s = lambda_s, status_strategy = status_strategy)
        
        # Have a list and dict of agents to modify G and identify neighbors
        self.agents_list = self.agents[:]

        self.agents_dict = {}
        for agent in self.agents:
            self.agents_dict[agent.unique_id] = agent

        # Pair record
        self.history_pairs = {}

        # Have to create initial pairs
        pairs = []
        for i in range(0, self.num_agents, 2):
            a1 = self.agents_list[i].unique_id
            a2 = self.agents_list[i+1].unique_id
            pairs.append((a1, a2))
        
        self.history_pairs[self.steps] = pairs
        self.G.add_edges_from(pairs)

        # Create datacollector and collect initial data
        self.datacollector = DataCollector(model_reporters={"network": lambda m: m.G}, 
                                           agent_reporters={"status": "status", "consumption": "consumption", "group": "assigned_group", "total_neighbors": "total_neighbors", 
                                                            "identities_neighbors": "identities_neighbors", "array_consumptions": "array_consumptions",
                                                            "belief_min": "belief_min", "belief_median": "belief_median", "belief_max": "belief_max",
                                                            "rankings_pro": "rankings_pro", "rankings_anti": "rankings_anti", "rankings_neutral": "rankings_neutral",
                                                            "rpro_i": "rpro_i", "ranti_i": "ranti_i", "rneutral_i": "rneutral_i",
                                                            "wpro_i": "wpro_i", "wanti_i": "wanti_i", "wneutral_i": "wneutral_i",
                                                            "u_pro": "u_pro", "u_anti": "u_anti", "u_neutral": "u_neutral", 
                                                            "status_pro" : "status_pro", "status_anti" : "status_anti", "status_neutral" : "status_neutral"})
        
        self.datacollector.collect(self)

        

    def step(self):

        # 1) Control agents -------------
        # Simultaneously activate agents 
        self.agents.do("calculate_status_classical")
        self.agents.do("calculate_status_alternative")
        self.agents.do("choose_consumption_alternative")
        self.agents.do("choose_consumption_classical")
        self.agents.do("update_group")

        # 2) Control netweork ------------
        # Create random pairs to add as edges to the network
        self.random.shuffle(self.agents_list)

        pairs = []
        for i in range(0, self.num_agents, 2):
            a1 = self.agents_list[i].unique_id
            a2 = self.agents_list[i+1].unique_id
            pairs.append((a1, a2))
        
        self.history_pairs[self.steps] = pairs # We will only have 'memory' steps saved in each period
        self.G.add_edges_from(pairs) # For now, this is an undirected / unweighted graph

        # Then remove old edges
        if self.steps > self.memory:
            self.G.remove_edges_from(self.history_pairs[self.steps - self.memory])
            self.history_pairs.pop(self.steps - self.memory)
        
        # 3) Collect data -------------
        self.datacollector.collect(self)

