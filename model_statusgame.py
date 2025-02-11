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
    Create N agents and manipulate a graph where they will interact.
    """
    def __init__(self, N, seed=None, lambda_s=1, status_strategy="nontie", memory=10, create_network=True, p=1/10):
        super().__init__(seed=seed)
        self.num_agents = N
        self.memory = memory  # parameter for deleting edges
        
        # Create agents (agents are added to self.agents by statusgame_agent.create_agents)
        statusgame_agent.create_agents(model=self, n=N, lambda_s=lambda_s, status_strategy=status_strategy)
        
        # Store agents in a list and build a dict for fast lookup
        self.agents_list = list(self.agents)
        self.agents_dict = {agent.unique_id: agent for agent in self.agents}
        
        # Initialize history record
        self.history_pairs = {}
        
        # Create initial network
        unique_ids = [agent.unique_id for agent in self.agents]
        if create_network:
            mapping = dict(enumerate(unique_ids))
            G_numeric = nx.fast_gnp_random_graph(N, p)
            self.G = nx.relabel_nodes(G_numeric, mapping)
            initial_edges = list(self.G.edges())
            # Instead of storing each edge with a separate key, we store them all under step 0.
            self.history_pairs[0] = initial_edges[:]  # copy initial edges
        else:
            self.G = nx.Graph()
            self.G.add_nodes_from(unique_ids)
        
        # Add additional pairs in step 0
        pairs = [(self.agents_list[i].unique_id, self.agents_list[i+1].unique_id)
                 for i in range(0, self.num_agents - self.num_agents % 2, 2)
                 if not self.G.has_edge(self.agents_list[i].unique_id, self.agents_list[i+1].unique_id)]
        # Use setdefault to initialize history_pairs for step 0 if not already there
        self.history_pairs.setdefault(0, []).extend(pairs)
        self.G.add_edges_from(pairs)
        
        # Create datacollector and collect initial data
        self.datacollector = DataCollector(
            #model_reporters={"network": lambda m: m.G},
            agent_reporters={
                "group": "assigned_group", 
                "status": "status", 
                "consumption": "consumption", 
                "utility": "utility",
                "belief_min": "belief_min", 
                "belief_median": "belief_median", 
                "belief_max": "belief_max",
                "u_pro": "u_pro", 
                "u_anti": "u_anti", 
                "u_neutral": "u_neutral", 
                "status_pro": "status_pro", 
                "status_anti": "status_anti", 
                "status_neutral": "status_neutral"
            }
        )
        self.datacollector.collect(self)
        
        self.running = True

    def step(self):
        # 1) Simultaneously activate agents 
        self.agents.do("calculate_beliefs")
        self.agents.do("calculate_status_alternative")
        self.agents.do("choose_consumption_alternative")
        self.agents.do("update_group")

        # 2) Update network: random pairing
        self.random.shuffle(self.agents_list)
        pairs = [(self.agents_list[i].unique_id, self.agents_list[i+1].unique_id)
                 for i in range(0, self.num_agents - self.num_agents % 2, 2)
                 if not self.G.has_edge(self.agents_list[i].unique_id, self.agents_list[i+1].unique_id)]
        # Use setdefault instead of try/except
        self.history_pairs.setdefault(self.steps, []).extend(pairs)
        self.G.add_edges_from(pairs)
        
        # Remove old edges if memory limit reached
        if self.steps >= self.memory:
            old_pairs = self.history_pairs.pop(self.steps - self.memory, [])
            self.G.remove_edges_from(old_pairs)
        
        # 3) Collect data
        self.datacollector.collect(self)
