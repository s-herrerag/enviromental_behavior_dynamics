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
                
        # 2) Network update (identical to the original model)
        all_ids = np.array([agent.unique_id for agent in self.agents_list])
        self.random.shuffle(all_ids)  # shuffle for randomness

        # Build available sets per group from the current agents.
        # (We use the agent selection based on assigned_group as in the original model.)
        pro_ids    = set(a.unique_id for a in self.agents if a.assigned_group == "Pro - environment")
        anti_ids   = set(a.unique_id for a in self.agents if a.assigned_group == "Anti - environment")
        neutral_ids= set(a.unique_id for a in self.agents if a.assigned_group == "Neutral")
        available_all = set(all_ids)

        new_pairs = []
        available_list = list(available_all)
        self.random.shuffle(available_list)  # randomize order of pairing attempts

        while len(available_list) >= 2:
            agent_id = available_list.pop(0)
            if agent_id not in available_all:
                continue
            agent_group = self.agents_dict[agent_id].assigned_group
            # Determine available partners by group.
            if agent_group == "Pro - environment":
                same_group_avail = pro_ids - {agent_id}
                diff_group_avail = (anti_ids | neutral_ids) & available_all
            elif agent_group == "Anti - environment":
                same_group_avail = anti_ids - {agent_id}
                diff_group_avail = (pro_ids | neutral_ids) & available_all
            else:  # "Neutral"
                same_group_avail = neutral_ids - {agent_id}
                diff_group_avail = (pro_ids | anti_ids) & available_all

            # Decide whether to pick from the same group (homophily) or a different group.
            if self.random.random() < self.rho and same_group_avail:
                candidate_pool = list(same_group_avail & available_all)
            elif diff_group_avail:
                candidate_pool = list(diff_group_avail)
            elif same_group_avail:
                candidate_pool = list(same_group_avail & available_all)
            else:
                continue

            partner_id = self.random.choice(candidate_pool)
            # Verify that there is no existing edge; try a few times if needed.
            max_attempts = 10
            attempts = 0
            while ((agent_id, partner_id) in self.edge_set or 
                   (partner_id, agent_id) in self.edge_set or 
                   agent_id == partner_id) and attempts < max_attempts:
                partner_id = self.random.choice(candidate_pool)
                attempts += 1
            if attempts == max_attempts:
                continue

            # Record the new pair.
            new_pairs.append((agent_id, partner_id))
            self.edge_set.add((agent_id, partner_id))
            # Remove both agents from available pools.
            available_all.discard(agent_id)
            available_all.discard(partner_id)
            if agent_id in available_list:
                available_list.remove(agent_id)
            if partner_id in available_list:
                available_list.remove(partner_id)
            pro_ids.discard(agent_id); anti_ids.discard(agent_id); neutral_ids.discard(agent_id)
            pro_ids.discard(partner_id); anti_ids.discard(partner_id); neutral_ids.discard(partner_id)

        # Add the new pairs to the graph and record them.
        self.G.add_edges_from(new_pairs)
        if self.steps not in self.history_pairs:
            self.history_pairs[self.steps] = []
        self.history_pairs[self.steps].extend(new_pairs)

        # Remove old edges if the memory limit is reached.
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

        # Update network metrics.
        self.dense_degree = sum(dict(self.G.degree()).values()) / self.num_agents
        self.connected_components = nx.number_connected_components(self.G)
        largest_cc = max(nx.connected_components(self.G), key=len)
        G_largest = self.G.subgraph(largest_cc)
        self.max_path_length = nx.diameter(G_largest)
        self.cluster_coefficient = nx.average_clustering(self.G)

        # 3) Collect data.
        self.datacollector.collect(self)
