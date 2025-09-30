"""
ABM Comparison Script:
Compare the evolution of one agent’s decisions when (i) all agents update their consumption/group 
according to individual incentives (full simulation) versus (ii) when all other agents remain fixed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import mesa

# Import the original model and agent definitions.
# (Ensure that model_statusgame.py, agents_statusgame.py, and helpers.py are in your PYTHONPATH.)
from model_statusgame import statusgame_model
from agents_statusgame import statusgame_agent

# ------------------------------------------------------------------------------
# Define a subclass of the original model that runs a "one-active" simulation.
# Only the agent with id equal to one_active_agent_id will update; the others keep their initial values.
# ------------------------------------------------------------------------------
class OneActiveStatusGameModel(statusgame_model):
    def __init__(self, N, one_active_agent_id, **kwargs):
        # Call the original initialization so that agents, network, and data collector are set up
        super().__init__(N, **kwargs)
        # Mark agents: only the chosen agent is active.
        for agent in self.agents:
            agent.active = (agent.unique_id == one_active_agent_id)

    def step(self):
        # Precompute neighbor sets (as in the original model)
        self.neighbor_sets = {agent.unique_id: set(self.G.neighbors(agent.unique_id))
                              for agent in self.agents_list}

        # 1) Update only active agents.
        # (Inactive agents simply keep their initial consumption, group, etc.)
        for agent in self.agents:
            if getattr(agent, 'active', True):  # default True if not set
                agent.calculate_beliefs()
                agent.calculate_status_alternative()
                agent.calculate_field_of_action()
                agent.choose_consumption_alternative()
                agent.update_group()
                
        # 2) Update network: Create new pairs with homophily.
        # Get the current unique_ids array (for reference)
        all_ids = np.array([agent.unique_id for agent in self.agents_list])

        # Build available sets per group from the current agents.
        pro_ids    = set(a.unique_id for a in self.agents.select(lambda a: a.assigned_group == "Pro - environment"))
        anti_ids   = set(a.unique_id for a in self.agents.select(lambda a: a.assigned_group == "Anti - environment"))
        neutral_ids= set(a.unique_id for a in self.agents.select(lambda a: a.assigned_group == "Neutral"))
        available_all = set(all_ids)  # all available agents for pairing

        # 2) Update network: Create new edges based on homophily + fraction of j’s neighbors.

        new_pairs = []
        all_ids = np.array([agent.unique_id for agent in self.agents_list])
        self.random.shuffle(all_ids)  # randomize order for fairness

        for i_id in all_ids:
            # For clarity, agent i meets exactly one "primary" partner j.
            i_group = self.agents_dict[i_id].assigned_group
            
            # Partition by group:
            if i_group == "Pro - environment":
                same_group_set = pro_ids - {i_id}
                diff_group_set = (anti_ids | neutral_ids)
            elif i_group == "Anti - environment":
                same_group_set = anti_ids - {i_id}
                diff_group_set = (pro_ids | neutral_ids)
            else:  # "Neutral"
                same_group_set = neutral_ids - {i_id}
                diff_group_set = (pro_ids | anti_ids)
            
            # Decide which group pool to use (homophily):
            if self.random.random() < self.rho and same_group_set:
                candidate_pool = list(same_group_set)
            elif diff_group_set:
                candidate_pool = list(diff_group_set)
            else:
                continue  # no one is available in that group

            # Randomly pick j from the chosen pool:
            j_id = self.random.choice(candidate_pool)

            # Create an edge (i, j) if it doesn't exist:
            if (i_id, j_id) not in self.edge_set and (j_id, i_id) not in self.edge_set:
                new_pairs.append((i_id, j_id))
                self.edge_set.add((i_id, j_id))

            # Now agent i also meets a fraction alpha of j’s neighbors:
            j_neighbors = self.neighbor_sets[j_id]
            # Compute how many of j’s neighbors to meet:
            n_meet = int(np.floor(self.alpha * len(j_neighbors)))
            if n_meet > 0:
                # Randomly pick that many out of j’s neighbors:
                chosen_neighbors = self.random.sample(list(j_neighbors), n_meet)
                for nbr_id in chosen_neighbors:
                    # Add (i, nbr_id) if it doesn’t exist:
                    if (i_id, nbr_id) not in self.edge_set and (nbr_id, i_id) not in self.edge_set:
                        new_pairs.append((i_id, nbr_id))
                        self.edge_set.add((i_id, nbr_id))

        # Add the new edges to the graph and record them:
        self.G.add_edges_from(new_pairs)
        if self.steps not in self.history_pairs:
            self.history_pairs[self.steps] = []
        self.history_pairs[self.steps].extend(new_pairs)

        # 3) Remove edges older than “memory” steps:
        if self.steps >= self.memory:
            old_key = self.steps - self.memory
            if old_key in self.history_pairs:
                old_edges = self.history_pairs.pop(old_key)
                self.G.remove_edges_from(old_edges)
                # Also remove them from edge_set for consistency:
                for e in old_edges:
                    if e in self.edge_set:
                        self.edge_set.remove(e)
                    elif (e[1], e[0]) in self.edge_set:
                        self.edge_set.remove((e[1], e[0]))

    # 3) Collect data.

        # Update network metrics.
        self.dense_degree = sum(dict(self.G.degree()).values()) / self.num_agents
        self.connected_components = nx.number_connected_components(self.G)
        largest_cc = max(nx.connected_components(self.G), key=len)
        G_largest = self.G.subgraph(largest_cc)
        self.max_path_length = nx.diameter(G_largest)
        self.cluster_coefficient = nx.average_clustering(self.G)

        # 3) Collect data.
        self.datacollector.collect(self)
