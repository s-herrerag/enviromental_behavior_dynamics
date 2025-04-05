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
                 memory=10, create_network="erdos_renyi", 
                 p=1/10, 
                 gamma = 1, 
                 k = 5, 
                 rho = 1/3, 
                 seed_consumption = False, 
                 alpha = 1/2, 
                 weights = [1/3, 1/3, 1/3],
                 shock = 0, 
                 period_shock = 0, 
                 wait_gamma = False, 
                 fraction_leaders=0,
                 fraction_attention=0,
                 beta=0.5):
        super().__init__(seed=seed)
        self.num_agents = N
        self.memory = memory  # Parameter for deleting edges
        self.rho = rho
        self.alpha = alpha

        self.fraction_leaders = fraction_leaders
        self.fraction_attention = fraction_attention
        self.beta = beta

        if seed_consumption:
            np.random.seed(seed)

        # Create agents
        statusgame_agent.create_agents(model=self, n=N, 
                                       lambda_s=lambda_s, gamma=gamma, 
                                       weights=weights, shock=shock, period_shock=period_shock, 
                                       wait_gamma=wait_gamma)

        # Build list and dict of agents
        self.agents_list = self.agents[:]  # shallow copy of agents list
        self.agents_dict = {agent.unique_id: agent for agent in self.agents}

        # Mark a fraction of them as leaders
        num_leaders = int(np.floor(self.num_agents * self.fraction_leaders))
        leader_candidates = self.random.sample(self.agents_list, num_leaders)
        for ag in leader_candidates:
            ag.is_leader = True

        # Mark a fraction of them as "paying attention"
        num_attention = int(np.floor(self.num_agents * self.fraction_attention))
        attention_candidates = self.random.sample(self.agents_list, num_attention)
        for ag in attention_candidates:
            ag.pays_attention = True

        # Record of edge creation by step
        self.history_pairs = {}

        # Create initial edges
        unique_ids = np.array([agent.unique_id for agent in self.agents_list])
        if create_network == "erdos_renyi":
            mapping = dict(enumerate(unique_ids))
            G_numeric = nx.fast_gnp_random_graph(N, p, seed = seed)
            self.G = nx.relabel_nodes(G_numeric, mapping)
            initial_edges = list(self.G.edges())
            # Test with all in 0
            self.history_pairs[0] = initial_edges   
        
        elif create_network == "watts_strogatz":
            mapping = dict(enumerate(unique_ids))
            G_numeric = nx.watts_strogatz_graph(N, k=k, p=p, seed = seed)
            self.G = nx.relabel_nodes(G_numeric, mapping)
            initial_edges = list(self.G.edges())
            # Test with all in 0
            self.history_pairs[0] = initial_edges
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
                "status_neutral": "status_neutral", 
                "degree": lambda a: a.model.G.degree(a.unique_id)
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
        self.agents.do("gamma_introduce")
        self.agents.do("calculate_field_of_action")
        self.agents.do("choose_consumption_alternative")
        self.agents.do("update_consumption")
        self.agents.do("update_group")

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
            if i_id in j_neighbors:
                j_neighbors = j_neighbors - {i_id}
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

        # Update network metrics.
        self.dense_degree = sum(dict(self.G.degree()).values()) / self.num_agents
        self.connected_components = nx.number_connected_components(self.G)
        largest_cc = max(nx.connected_components(self.G), key=len)
        G_largest = self.G.subgraph(largest_cc)
        self.max_path_length = nx.diameter(G_largest)
        self.cluster_coefficient = nx.average_clustering(self.G)

        # 3) Collect data.
        self.datacollector.collect(self)

