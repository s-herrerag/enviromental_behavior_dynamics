#########################################
# Model class of the status identity game
#########################################
import mesa
import numpy as np
from mesa.datacollection import DataCollector
import networkx as nx
from agents_statusgame import statusgame_agent

class statusgame_model(mesa.Model):
    """
    Create N agents and manipulate a graph where they will interact.
    """
    def __init__(self, N, seed=None, lambda_s=1, status_strategy="nontie",
                 memory=10, create_network=True, p=1/10):
        super().__init__(seed=seed)
        self.num_agents = N
        self.memory = memory  # Parameter for deleting edges

        # Create agents
        statusgame_agent.create_agents(model=self, n=N, lambda_s=lambda_s, status_strategy=status_strategy)

        # Build list and dict of agents
        self.agents_list = self.agents[:]  # shallow copy of agents list
        self.agents_dict = {agent.unique_id: agent for agent in self.agents}

        # Record of edge creation by step
        self.history_pairs = {}

        # Create initial edges
        unique_ids = np.array([agent.unique_id for agent in self.agents_list])
        if create_network:
            mapping = dict(enumerate(unique_ids))
            G_numeric = nx.fast_gnp_random_graph(N, p)
            self.G = nx.relabel_nodes(G_numeric, mapping)
            initial_edges = list(self.G.edges())
            # Store each initial edge in its own history entry (key = index)
            for i, e in enumerate(initial_edges):
                self.history_pairs[i] = [e]
        else:
            self.G = nx.Graph()
            self.G.add_nodes_from(unique_ids)

        # Add additional pairs in step 0
        # Use vectorized pairing: zip pairs from the ordered list of unique_ids
        candidate_pairs = list(zip(unique_ids[0::2], unique_ids[1::2]))
        # Filter candidate pairs to add only those not already present
        additional_pairs = [pair for pair in candidate_pairs if not self.G.has_edge(pair[0], pair[1])]
        # Ensure there is a list for the current step (self.steps is 0 at initialization)
        if self.steps not in self.history_pairs:
            self.history_pairs[self.steps] = []
        self.history_pairs[self.steps].extend(additional_pairs)
        self.G.add_edges_from(additional_pairs)

        # Auxiliary set for fast edge membership checking
        self.edge_set = set(self.G.edges())

        # Network reporters
        self.average_degree = sum(dict(self.G.degree()).values()) / self.num_agents
        self.connected_components = nx.number_connected_components(self.G)
        largest_cc = max(nx.connected_components(self.G), key=len)
        G_largest = self.G.subgraph(largest_cc)
        self.max_path_length = nx.diameter(G_largest)


        # Create datacollector and collect initial data
        self.datacollector = DataCollector(
            model_reporters={"average_degree": "average_degree",
                             "connected_components": "connected_components",
                             "max_path_length": "max_path_length"},
            agent_reporters={
                "group": "assigned_group",
                "avg_common": "n_common",
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
        # Precompute neighbor sets once per step for fast agent lookups.
        self.neighbor_sets = {agent.unique_id: set(self.G.neighbors(agent.unique_id))
                              for agent in self.agents_list}

        # 1) Activate agents simultaneously.
        self.agents.do("calculate_beliefs")
        self.agents.do("calculate_status_alternative")
        self.agents.do("choose_consumption_alternative")
        self.agents.do("update_group")

        # 2) Update network: Create random new pairs.
        unique_ids = np.array([agent.unique_id for agent in self.agents_list])
        self.random.shuffle(unique_ids)  # Uses the model’s random generator.
        candidate_pairs = list(zip(unique_ids[0::2], unique_ids[1::2]))
        # Filter out pairs that already exist (check both orders in the auxiliary set)
        new_pairs = [pair for pair in candidate_pairs
                     if (pair not in self.edge_set and (pair[1], pair[0]) not in self.edge_set)]
        # Record new pairs in the history for the current step.
        if self.steps not in self.history_pairs:
            self.history_pairs[self.steps] = []
        self.history_pairs[self.steps].extend(new_pairs)
        self.G.add_edges_from(new_pairs)
        self.edge_set.update(new_pairs)

        # Remove old edges if memory limit is reached.
        if self.steps >= self.memory:
            old_key = self.steps - self.memory
            if old_key in self.history_pairs:
                old_pairs = self.history_pairs.pop(old_key)
                self.G.remove_edges_from(old_pairs)
                for pair in old_pairs:
                    if pair in self.edge_set:
                        self.edge_set.remove(pair)
                    elif (pair[1], pair[0]) in self.edge_set:
                        self.edge_set.remove((pair[1], pair[0]))

        # Update network metrics
        self.average_degree = sum(dict(self.G.degree()).values()) / self.num_agents
        self.connected_components = nx.number_connected_components(self.G)
        largest_cc = max(nx.connected_components(self.G), key=len)
        G_largest = self.G.subgraph(largest_cc)
        self.max_path_length = nx.diameter(G_largest)

        # 3) Collect data.
        self.datacollector.collect(self)
