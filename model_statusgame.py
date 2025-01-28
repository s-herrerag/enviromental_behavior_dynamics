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
        
        # Create agents 
        statusgame_agent.create_agents(model=self, n=N, seed = seed, lambda_s = lambda_s, status_strategy = status_strategy)
        
        # Create an empty Graph in which agents will be nodes
        self.agents_list = self.agents[:]
        self.G = nx.Graph()
        self.G.add_nodes_from(self.agents_list)

        # Pair record
        self.history_pairs = {}

    def step(self):
        # Create random pairs to add as edges to the network
        self.random.shuffle(self.agents_list)

        pairs = []
        for i in range(0, self.num_agents, 2):
            a1 = self.agents_list[i]
            a2 = self.agents_list[i+1]
            pairs.append((a1, a2))
        
        self.history_pairs[self.steps] = pairs # We will only have 'memory' steps saved in each period
        
        # Add them to the network
        self.G.add_edges_from(pairs) # For now, this is an undirected / unweighted graph

        # Then remove old edges
        if self.steps > self.memory:
            self.G.remove_edges_from(self.history_pairs[self.steps - self.memory])
            self.history_pairs.pop(self.steps - self.memory)

        # Simultaneously activate agents
        self.agents.do("calculate_status")
        #self.agents.do("choose_consumption")

