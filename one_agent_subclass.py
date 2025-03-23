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

        new_pairs = []
        # Build a list of all currently available agents.
        available_list = list(available_all)
        self.random.shuffle(available_list)

        while len(available_list) >= 2:
            agent_id = available_list.pop(0)
            # Skip if agent_id was removed already (already paired).
            if agent_id not in available_all:
                continue

            # Decide which mechanism to use:
            if self.random.random() < self.alpha:
                #
                #  (A) Match with friend-of-friend
                #
                # Get the agent's direct neighbors (already computed in neighbor_sets).
                friends = self.neighbor_sets[agent_id]
                # Build the set of friends-of-friends.
                friend_of_friends = set()
                for f_id in friends:
                    friend_of_friends.update(self.neighbor_sets[f_id])
                # Remove direct friends and the agent itself from that set.
                friend_of_friends.discard(agent_id)
                friend_of_friends -= friends
                # Restrict to currently unpaired agents.
                candidate_pool = list(friend_of_friends & available_all)

            else:
                #
                #  (B) Match randomly, inside the group with prob rho or outside with prob (1-rho)
                #
                agent_group = self.agents_dict[agent_id].assigned_group
                if agent_group == "Pro - environment":
                    same_group_avail = pro_ids - {agent_id}
                    diff_group_avail = (anti_ids | neutral_ids) & available_all
                elif agent_group == "Anti - environment":
                    same_group_avail = anti_ids - {agent_id}
                    diff_group_avail = (pro_ids | neutral_ids) & available_all
                else:  # "Neutral"
                    same_group_avail = neutral_ids - {agent_id}
                    diff_group_avail = (pro_ids | anti_ids) & available_all

                # Decide whether to pick from the same or a different group.
                if self.random.random() < self.rho and same_group_avail:
                    candidate_pool = list(same_group_avail & available_all)
                elif diff_group_avail:
                    candidate_pool = list(diff_group_avail)
                elif same_group_avail:
                    candidate_pool = list(same_group_avail & available_all)
                else:
                    candidate_pool = []

            if not candidate_pool:
                # No valid partner found for this agent — skip.
                continue

            # Pick the partner, making sure we don't duplicate edges.
            max_attempts = 10
            attempts = 0
            partner_id = None

            while attempts < max_attempts:
                candidate = self.random.choice(candidate_pool)
                attempts += 1
                # Verify there's no existing edge to candidate.
                if (agent_id, candidate) not in self.edge_set and (candidate, agent_id) not in self.edge_set:
                    partner_id = candidate
                    break

            # If no valid partner was found, skip this agent.
            if partner_id is None:
                continue

            # Record the pair and update the sets.
            new_pairs.append((agent_id, partner_id))
            self.edge_set.add((agent_id, partner_id))

            # Remove paired agents from the available pools.
            available_all.discard(agent_id)
            available_all.discard(partner_id)
            if agent_id in available_list:
                available_list.remove(agent_id)
            if partner_id in available_list:
                available_list.remove(partner_id)
            pro_ids.discard(agent_id);  anti_ids.discard(agent_id);  neutral_ids.discard(agent_id)
            pro_ids.discard(partner_id);  anti_ids.discard(partner_id);  neutral_ids.discard(partner_id)

        # Finally, add the new pairs to the network and record them.
        self.G.add_edges_from(new_pairs)
        if self.steps not in self.history_pairs:
            self.history_pairs[self.steps] = []
        self.history_pairs[self.steps].extend(new_pairs)

        # Remove old edges if the memory limit is reached (unchanged from your previous code).
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
