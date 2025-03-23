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
    def __init__(self, N, seed=None, lambda_s=1,
                 memory=10, create_network="erdos_renyi", p=1/10, gamma = 1, k = 5, rho = 1/3, 
                 seed_consumption = False, alpha = 1/2):
        super().__init__(seed=seed)
        self.num_agents = N
        self.memory = memory  # Parameter for deleting edges
        self.rho = rho
        self.alpha = alpha

        if seed_consumption:
            np.random.seed(seed)

        # Create agents
        statusgame_agent.create_agents(model=self, n=N, lambda_s=lambda_s, gamma=gamma)

        # Build list and dict of agents
        self.agents_list = self.agents[:]  # shallow copy of agents list
        self.agents_dict = {agent.unique_id: agent for agent in self.agents}

        # Record of edge creation by step
        self.history_pairs = {}

        # Create initial edges
        unique_ids = np.array([agent.unique_id for agent in self.agents_list])
        if create_network == "erdos_renyi":
            mapping = dict(enumerate(unique_ids))
            G_numeric = nx.fast_gnp_random_graph(N, p)
            self.G = nx.relabel_nodes(G_numeric, mapping)
            initial_edges = list(self.G.edges())
            # Store each initial edge in its own history entry (key = index)
            for i, e in enumerate(initial_edges):
                self.history_pairs[i] = [e]
        
        elif create_network == "watts_strogatz":
            mapping = dict(enumerate(unique_ids))
            G_numeric = nx.watts_strogatz_graph(N, k=k, p=p)
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
        self.dense_degree = sum(dict(self.G.degree()).values()) / self.num_agents
        self.connected_components = nx.number_connected_components(self.G)
        largest_cc = max(nx.connected_components(self.G), key=len)
        G_largest = self.G.subgraph(largest_cc)
        self.max_path_length = nx.diameter(G_largest)
        self.cluster_coefficient = nx.average_clustering(self.G)


        # Create datacollector and collect initial data
        self.datacollector = DataCollector(
            model_reporters={"dense_degree": "dense_degree",
                             "connected_components": "connected_components",
                             "max_path_length": "max_path_length", 
                             "cluster_coefficient": "cluster_coefficient"},
            agent_reporters={
                "group": "assigned_group",
                "avg_common": "n_common",
                "status": "status",
                "field_of_action": "field_of_action",
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
        # Precompute neighbor sets for all agents (used by agent methods)
        self.neighbor_sets = {agent.unique_id: set(self.G.neighbors(agent.unique_id))
                              for agent in self.agents_list}

        # 1) Activate agents simultaneously.
        self.agents.do("calculate_beliefs")
        self.agents.do("calculate_status_alternative")
        self.agents.do("calculate_field_of_action")
        self.agents.do("choose_consumption_alternative")
        self.agents.do("update_group")

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
                #  Match with friend-of-friend
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
                #  Match randomly, inside the group with prob rho or outside with prob (1-rho)
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


        # Update network metrics.
        self.dense_degree = sum(dict(self.G.degree()).values()) / self.num_agents
        self.connected_components = nx.number_connected_components(self.G)
        largest_cc = max(nx.connected_components(self.G), key=len)
        G_largest = self.G.subgraph(largest_cc)
        self.max_path_length = nx.diameter(G_largest)
        self.cluster_coefficient = nx.average_clustering(self.G)

        # 3) Collect data.
        self.datacollector.collect(self)

