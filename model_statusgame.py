#########################################
# Model class of the status identity game
#########################################
import mesa
import numpy as np
from mesa.datacollection import DataCollector
from scipy import stats
import math
import networkx as nx
import itertools

from agents_statusgame import statusgame_agent

class statusgame_model(mesa.Model):
    """
    Create N agents and manipulate a graph where they will interact
    """
    def __init__(self, N, seed=None, lambda_s = 1, status_strategy = "nontie", memory = 10, create_network = True, p = 1/10):
        super().__init__(seed=seed)
        self.num_agents = N
        self.memory = memory #Param for deleting edges

        # Create agents 
        statusgame_agent.create_agents(model=self, n=N, lambda_s = lambda_s, status_strategy = status_strategy)
        
        # Have a list and dict of agents to modify G and identify neighbors
        self.agents_list = self.agents[:]

        self.agents_dict = {}
        for agent in self.agents:
            self.agents_dict[agent.unique_id] = agent

        # Pair record
        self.history_pairs = {}

        # Have to create initial edges
        unique_ids = [agent.unique_id for agent in self.agents]
        if create_network:
            mapping = dict(enumerate(unique_ids))
            G_numeric = nx.fast_gnp_random_graph(N, p)
            self.G = nx.relabel_nodes(G_numeric, mapping)
            initial_edges = list(self.G.edges())
            for i, e in enumerate(initial_edges):
                self.history_pairs[i] = [e]   
        else:
            self.G = nx.Graph()
            self.G.add_nodes_from(unique_ids)

        # Add additional pairs in step 0
        pairs = []
        for i in range(0, self.num_agents, 2):
            a1 = self.agents_list[i].unique_id
            a2 = self.agents_list[i+1].unique_id
            if not self.G.has_edge(a1, a2):
            # It's a genuinely new edge, so add it
                pairs.append((a1, a2))
        try:
            self.history_pairs[self.steps]
        except KeyError:
            self.history_pairs[self.steps] = []
        
        self.history_pairs[self.steps].extend(pairs)
        self.G.add_edges_from(pairs)

        # Create datacollector and collect initial data
        self.datacollector = DataCollector(model_reporters={"network": lambda m: m.G}, 
                                           agent_reporters={"group": "assigned_group", "status": "status", "consumption": "consumption", "utility": "utility",
                                                            "belief_min": "belief_min", "belief_median": "belief_median", "belief_max": "belief_max",
                                                            "u_pro": "u_pro", "u_anti": "u_anti", "u_neutral": "u_neutral", 
                                                            "status_pro" : "status_pro", "status_anti" : "status_anti", "status_neutral" : "status_neutral"})
        
        self.datacollector.collect(self)

        # Batch
        self.running = True

        
    def step(self):

        # 1) Control agents -------------
        # Simultaneously activate agents 

        self.agents.do("calculate_beliefs")
        self.agents.do("calculate_status_alternative")
        self.agents.do("choose_consumption_alternative")
        
        self.agents.do("update_group")

        # 2) Control netweork ------------
        # Create random pairs to add as edges to the network
        self.random.shuffle(self.agents_list)

        pairs = []
        for i in range(0, self.num_agents, 2):
            a1 = self.agents_list[i].unique_id
            a2 = self.agents_list[i+1].unique_id
            if not self.G.has_edge(a1, a2):
            # It's a genuinely new edge, so add it
                pairs.append((a1, a2))
        
        try:
            self.history_pairs[self.steps]
        except KeyError:
            self.history_pairs[self.steps] = []
        
        self.history_pairs[self.steps].extend(pairs)

        self.G.add_edges_from(pairs) # For now, this is an undirected / unweighted graph

        # Then remove old edges
        if self.steps >= self.memory:
            self.G.remove_edges_from(self.history_pairs[self.steps - self.memory])
            self.history_pairs.pop(self.steps - self.memory)
        
        # 3) Collect data -------------
        self.datacollector.collect(self)

