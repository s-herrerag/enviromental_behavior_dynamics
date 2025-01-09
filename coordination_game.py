################################
# Emissions / Coordination Game
################################

### Libraries ------------------------

import mesa
import numpy as np
from mesa.datacollection import DataCollector
from scipy import stats
import math

### Helpers ------------------------
from helpers import get_distribution
from helpers import calculate_mode_hist_midpoint
from helpers import g_pro, g_neutral, g_anti, maximize_utility

# Choose consumption distribution
consumption_dist = get_distribution(dist_type="uniform", lower=5, upper=100)

### Agents ------------------------
class coordination_agent(mesa.Agent):
    """
    An agent with a defined category, utility function, and step method.
    """

    def __init__(self, unique_id, model, lambda1=1/3, lambda2=1/3, alpha=1, beta=2/3,
                 steps_convincement=10):
        super().__init__(unique_id, model)

        # Assign initial group randomly
        self.assigned_group = self.model.random.choice(["Pro - environment", "Neutral", "Anti - environment"])
        # Assign initial consumption
        c0 = consumption_dist.rvs(size=1)[0]
        self.history = [c0]

        # Initialize observations and beliefs
        self.others_actions = []
        self.others_identities = []

        # Store parameters
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.steps_convincement = steps_convincement
        self.alpha = alpha
        self.beta = beta

        # Initialize alter ego and related histories
        self.alterego = self.assigned_group
        self.alternative_utilities = []
        self.alter_s_i_history = []
        self.alter_rankings = {'rpro': [], 'ranti': [], 'rneutral': []}

        # Placeholder for status component
        self.s_i = 0

        # Initialize effort and incoming efforts
        self.effort = 0
        self.incoming_efforts = []

        # Bias
        self.bias = 1 #self.model.random.uniform(0.5, 1)

        # Initialize utility history
        self.utilities = []

        # Initialize rankings histories
        self.rpro_history = []
        self.ranti_history = []
        self.rneutral_history = []

        # Initialize efforts given and received histories
        self.efforts_given = []
        self.efforts_received = []

        # Initialize other agent IDs history
        self.other_agent_ids = []

    def calculate_status(self, identity, rpro, ranti, rneutral, alpha, share):
        if identity == "Pro - environment":
            theta_pro = 0.5
            theta_anti = 0.25
        elif identity == "Anti - environment":
            theta_pro = 0.25
            theta_anti = 0.5
        else:  # Neutral
            theta_pro = 1/3
            theta_anti = 1/3
        theta_neutral = 1 - theta_pro - theta_anti

        individual_status = theta_pro * rpro + theta_anti * ranti + theta_neutral * rneutral
        status = alpha * individual_status + (1 - alpha) * share

        return status

    def utility(self, identity, consumption, status):
        if identity == "Pro - environment":
            x_hat = self.min_believed_consumption
            g_value = g_pro(consumption)
        elif identity == "Anti - environment":
            x_hat = self.max_believed_consumption
            g_value = g_anti(consumption)
        else:  # Neutral
            x_hat = self.mode_believed_consumption
            g_value = g_neutral(consumption)

        misalignment_cost = - (consumption - x_hat)**2 + g_value
        u = (self.lambda1 * consumption +
             self.lambda2 * status +
             (1 - self.lambda1 - self.lambda2) * misalignment_cost)
        return u

    def consumption_selection(self, identity, status):
        if identity == "Pro - environment":
            x_hat = self.min_believed_consumption
            g_func = g_pro
        elif identity == "Anti - environment":
            x_hat = self.max_believed_consumption
            g_func = g_anti
        else:  # Neutral
            x_hat = self.mode_believed_consumption
            g_func = g_neutral
        consumption = maximize_utility(
            x_hat=x_hat,
            g=g_func,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            s_i=status
        )
        return consumption

    def step(self):
        # Reset effort at the beginning of the step
        self.effort = 0

        # Get the other agent
        for (ag1, ag2) in self.model.pairs:
            if ag1 == self:
                other_agent = ag2
                break
            elif ag2 == self:
                other_agent = ag1
                break

        # Update sample of identities
        self.others_identities.append(other_agent.assigned_group)

        # Store the other agent's ID
        self.other_agent_ids.append(other_agent.unique_id)

        # See the consumption of all acquaintances (including the other agent) - Reset others actions
        self.others_actions = []
        for i in self.other_agent_ids:
            last_action = self.model.schedule.agents[i].history[-1] * self.bias
            self.others_actions.append(last_action)

        steps_taken = len(self.others_identities)
        share_pro = self.others_identities.count("Pro - environment") / steps_taken
        share_neutral = self.others_identities.count("Neutral") / steps_taken
        share_anti = self.others_identities.count("Anti - environment") / steps_taken

        # Update beliefs based on observed actions
        self.min_believed_consumption = min(self.others_actions) 
        self.max_believed_consumption = max(self.others_actions)
        mode_believed = calculate_mode_hist_midpoint(self.others_actions, bins=10)
        if mode_believed is None:
            mode_believed = self.history[-1]
        self.mode_believed_consumption = mode_believed 
        
        # Include own last consumption
        all_consumptions = self.others_actions + [self.history[-1]]
        consumptions_array = np.array(all_consumptions)

        # Calculate rankings before efforts
        rpro_i = 100 - stats.percentileofscore(consumptions_array, self.history[-1])
        ranti_i = stats.percentileofscore(consumptions_array, self.history[-1])
        mode_diff = np.abs(consumptions_array - self.mode_believed_consumption)
        mode_ranks = 1 / (mode_diff + 1e-6)
        self_mode_rank = mode_ranks[-1]
        rneutral_i = stats.percentileofscore(mode_ranks, self_mode_rank)

        # Append unadjusted rankings to history
        self.rpro_history.append(rpro_i)
        self.ranti_history.append(ranti_i)
        self.rneutral_history.append(rneutral_i)

        # Initialize adjusted rankings
        adjusted_rpro_i = rpro_i
        adjusted_ranti_i = ranti_i
        adjusted_rneutral_i = rneutral_i

        # Process incoming efforts and adjust rankings
        total_effort_received = 0
        for eff in self.incoming_efforts:
            group_name, effort_amount = eff
            total_effort_received += effort_amount
            if group_name == "Pro - environment":
                adjusted_rpro_i += effort_amount
            elif group_name == "Anti - environment":
                adjusted_ranti_i += effort_amount
            elif group_name == "Neutral":
                adjusted_rneutral_i += effort_amount
        # Clear incoming efforts after processing
        self.incoming_efforts = []

        # Store adjusted rankings for history
        self.alter_rankings['rpro'].append(adjusted_rpro_i)
        self.alter_rankings['ranti'].append(adjusted_ranti_i)
        self.alter_rankings['rneutral'].append(adjusted_rneutral_i)

        # Store total effort received
        self.efforts_received.append(total_effort_received)

        # Calculate status with adjusted rankings
        if self.assigned_group == "Pro - environment":
            share_own = 100*share_pro
        elif self.assigned_group == "Anti - environment":
            share_own = 100*share_anti
        else:
            share_own = 100*share_neutral

        self.s_i = self.calculate_status(
            self.assigned_group,
            rpro=adjusted_rpro_i,
            ranti=adjusted_ranti_i,
            rneutral=adjusted_rneutral_i,
            alpha=self.alpha,
            share=share_own
        )

        # Choose consumption
        consumption = self.consumption_selection(self.assigned_group, self.s_i)

        # Compute initial utility
        initial_utility = self.utility(self.assigned_group, consumption, self.s_i)
        utility = initial_utility  # Initialize utility

        # Determine if agent will exert effort to convince other_agent
        if other_agent.assigned_group != self.assigned_group:
            # Alternative share and rank if other_agent joins
            if self.assigned_group == "Pro - environment":
                alt_share_own = 100 * (self.others_identities.count("Pro - environment") + 1) / (steps_taken + 1)
                r_own = adjusted_rpro_i
            elif self.assigned_group == "Anti - environment":
                alt_share_own = 100 * (self.others_identities.count("Anti - environment") + 1) / (steps_taken + 1)
                r_own = adjusted_ranti_i
            else:
                alt_share_own = 100 * (self.others_identities.count("Neutral") + 1) / (steps_taken + 1)
                r_own = adjusted_rneutral_i

            alt_si = self.calculate_status(
                self.assigned_group,
                rpro=adjusted_rpro_i,
                ranti=adjusted_ranti_i,
                rneutral=adjusted_rneutral_i,
                alpha=self.alpha,
                share=alt_share_own
            )
            alt_utility = self.utility(self.assigned_group, consumption, alt_si)

            # Imaginary probability of turning others
            p_turning = 1 - math.exp(-self.beta * r_own)

            if p_turning == 0:
                self.effort = 0
                utility = initial_utility
            else:
                if alt_utility - initial_utility > (self.beta * r_own) / p_turning:
                    utility = alt_utility - (self.beta * r_own)
                    self.effort = self.beta * r_own
                    # Send effort to other_agent
                    other_agent.incoming_efforts.append((self.assigned_group, self.effort))
                else:
                    utility = initial_utility
                    self.effort = 0
        else:
            self.effort = 0

        # Store effort given
        self.efforts_given.append(self.effort)

        # Update utilities and history
        self.utilities.append(utility)
        self.history.append(consumption)

        # Calculate alter ego status
        if self.alterego == "Pro - environment":
            share_alterego = share_pro
        elif self.alterego == "Anti - environment":
            share_alterego = share_anti
        else:
            share_alterego = share_neutral

        alter_s_i = self.calculate_status(
            self.alterego,
            rpro=adjusted_rpro_i,
            ranti=adjusted_ranti_i,
            rneutral=adjusted_rneutral_i,
            alpha=self.alpha,
            share=share_alterego
        )
        self.alter_s_i_history.append(alter_s_i)

        # Compute consumption and utility for alter ego
        consumption_alter = self.consumption_selection(self.alterego, alter_s_i)
        utility_alter = self.utility(self.alterego, consumption_alter, alter_s_i)
        self.alternative_utilities.append(utility_alter)

        # Agents may change their group
        if steps_taken % self.steps_convincement == 0 and steps_taken >= self.steps_convincement:
            # Compute own average utility over the last steps
            own_avg_utility = np.mean(self.utilities[-self.steps_convincement:])
            # Compute average utility for alter ego
            alterego_avg_utility = np.mean(self.alternative_utilities[-self.steps_convincement:])

            # If alter ego has higher utility, switch
            if alterego_avg_utility > own_avg_utility:
                self.assigned_group = self.alterego

            # Compute average rankings over the last steps
            avg_rpro = np.mean(self.alter_rankings['rpro'][-self.steps_convincement:])
            avg_ranti = np.mean(self.alter_rankings['ranti'][-self.steps_convincement:])
            avg_rneutral = np.mean(self.alter_rankings['rneutral'][-self.steps_convincement:])

            # Determine new alter ego based on highest average ranking
            status_scores = {
                "Pro - environment": avg_rpro,
                "Anti - environment": avg_ranti,
                "Neutral": avg_rneutral
            }
            # Set alter ego to the identity with the highest average ranking
            self.alterego = max(status_scores, key=status_scores.get)

            # Reset alternative utilities and alter ego histories
            self.alternative_utilities = []
            self.alter_s_i_history = []
            self.alter_rankings = {'rpro': [], 'ranti': [], 'rneutral': []}

class coordination_model(mesa.Model):
    """
    A model with a number of agents.
    """

    def __init__(self, N, lambda1=1/3, lambda2=1/3, steps_convincement=10,
                 alpha=1, beta=2/3):
        super().__init__()
        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)

        for i in range(self.num_agents):
            a = coordination_agent(
                i, self,
                lambda1=lambda1,
                lambda2=lambda2,
                steps_convincement=steps_convincement,
                alpha=alpha,
                beta=beta
            )
            self.schedule.add(a)

        # Initialize DataCollector
        self.datacollector = DataCollector(
            agent_reporters={
                "Group": "assigned_group",
                "Consumption": lambda a: a.history[-1],
                "Utility": lambda a: a.utilities[-1] if a.utilities else None,
                "rpro": lambda a: a.rpro_history[-1] if a.rpro_history else None,
                "ranti": lambda a: a.ranti_history[-1] if a.ranti_history else None,
                "rneutral": lambda a: a.rneutral_history[-1] if a.rneutral_history else None,
                "EffortGiven": lambda a: a.efforts_given[-1] if a.efforts_given else None,
                "EffortReceived": lambda a: a.efforts_received[-1] if a.efforts_received else None,
                "OtherAgentID": lambda a: a.other_agent_ids[-1] if a.other_agent_ids else None,
                "Alterego": "alterego", 
                "AlterUtility": lambda a: a.alternative_utilities[-1] if a.alternative_utilities else None,
            }
        )

    def step(self):
        # Shuffle agents to randomize pairings
        agents_list = self.schedule.agents[:]
        self.random.shuffle(agents_list)

        # Pair agents in consecutive pairs
        self.pairs = []
        for i in range(0, len(agents_list), 2):
            a1 = agents_list[i]
            a2 = agents_list[i+1]
            self.pairs.append((a1, a2))

        # Each agent will know its paired other_agent from self.pairs.
        self.schedule.step()

        # Collect data after all agents have moved
        self.datacollector.collect(self)


## Permitir consumo negativo





### Other definition of status-seeking behavior: Replication

# Example distribution for initial consumption (as in your original code)
consumption_dist = get_distribution(dist_type="uniform", lower=5, upper=100)

class status_seeking_agent(mesa.Agent):
    """
    An agent that chooses consumption by looking at others' statuses
    and copying the highest-status consumption.
    """

    def __init__(self, unique_id, model,
                 lambda1=1/3, lambda2=1/3, alpha=1, beta=2/3,
                 steps_convincement=10):
        super().__init__(unique_id, model)

        # Assign initial group randomly
        self.assigned_group = self.model.random.choice(
            ["Pro - environment", "Neutral", "Anti - environment"]
        )
        # Assign initial consumption
        c0 = consumption_dist.rvs(size=1)[0]
        self.history = [c0]

        # Observations and beliefs
        self.others_actions = []
        self.others_identities = []

        # Model parameters
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.steps_convincement = steps_convincement
        self.alpha = alpha
        self.beta = beta

        # Alter ego and related tracking
        self.alterego = self.assigned_group
        self.alternative_utilities = []
        self.alter_s_i_history = []
        self.alter_rankings = {'rpro': [], 'ranti': [], 'rneutral': []}

        # Status component
        self.s_i = 0

        # Effort mechanics
        self.effort = 0
        self.incoming_efforts = []

        # Bias
        self.bias = 1  # or self.model.random.uniform(0.5, 1)

        # Utility history
        self.utilities = []

        # Rankings histories
        self.rpro_history = []
        self.ranti_history = []
        self.rneutral_history = []

        # Efforts
        self.efforts_given = []
        self.efforts_received = []

        # List of other agents (IDs) we've encountered
        self.other_agent_ids = []

    def calculate_status(self, identity, rpro, ranti, rneutral, alpha, share):
        """Compute full status given identity, ranks, alpha, and share."""
        if identity == "Pro - environment":
            theta_pro = 0.5
            theta_anti = 0.25
        elif identity == "Anti - environment":
            theta_pro = 0.25
            theta_anti = 0.5
        else:  # Neutral
            theta_pro = 1/3
            theta_anti = 1/3
        theta_neutral = 1 - theta_pro - theta_anti

        individual_status = theta_pro * rpro + theta_anti * ranti + theta_neutral * rneutral
        status = alpha * individual_status + (1 - alpha) * share
        return status

    def calculate_individual_status(self, identity, rpro, ranti, rneutral):
        """Compute the portion of status that depends on rpro/ranti/rneutral."""
        if identity == "Pro - environment":
            theta_pro = 0.5
            theta_anti = 0.25
        elif identity == "Anti - environment":
            theta_pro = 0.25
            theta_anti = 0.5
        else:  # Neutral
            theta_pro = 1/3
            theta_anti = 1/3
        theta_neutral = 1 - theta_pro - theta_anti
        return theta_pro * rpro + theta_anti * ranti + theta_neutral * rneutral

    def utility(self, identity, consumption, status):
        """
        Original utility function, though you mentioned wanting
        to remove the linear consumption term. 
        Keep or modify as needed.
        """
        if identity == "Pro - environment":
            x_hat = self.min_believed_consumption
            g_value = g_pro(consumption)
        elif identity == "Anti - environment":
            x_hat = self.max_believed_consumption
            g_value = g_anti(consumption)
        else:  # Neutral
            x_hat = self.mode_believed_consumption
            g_value = g_neutral(consumption)

        misalignment_cost = - (consumption - x_hat)**2 + g_value
        # If you want to remove direct consumption from utility, remove self.lambda1 * consumption below:
        u = (self.lambda1 * consumption +
             self.lambda2 * status +
             (1 - self.lambda1 - self.lambda2) * misalignment_cost)
        return u

    def consumption_selection(self, identity, status):
        """
        Original 'maximize_utility' approach if you want to keep it around.
        """
        if identity == "Pro - environment":
            x_hat = self.min_believed_consumption
            g_func = g_pro
        elif identity == "Anti - environment":
            x_hat = self.max_believed_consumption
            g_func = g_anti
        else:  # Neutral
            x_hat = self.mode_believed_consumption
            g_func = g_neutral

        consumption = maximize_utility(
            x_hat=x_hat,
            g=g_func,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            s_i=status
        )
        return consumption

    def compute_rankings(self, consumption_value):
        """
        Given a single consumption_value, compute rpro, ranti, and rneutral
        from THIS agent's perspective (self.others_actions + own last consumption).
        """
        consumptions_array = np.array(self.others_actions + [self.history[-1]])
        
        # rpro = 100 - percentile rank (higher consumption => higher 'Anti' rank => lower 'Pro' rank)
        rpro_i = 100 - stats.percentileofscore(consumptions_array, consumption_value)

        # ranti = percentile rank
        ranti_i = stats.percentileofscore(consumptions_array, consumption_value)

        # rneutral: based on how close the consumption is to the mode
        mode_diff = np.abs(consumptions_array - self.mode_believed_consumption)
        mode_ranks = 1 / (mode_diff + 1e-6)
        mode_rank = 1 / (np.abs(consumption_value - self.mode_believed_consumption) + 1e-6)
        rneutral_i = stats.percentileofscore(mode_ranks, mode_rank)
    
        return rpro_i, ranti_i, rneutral_i

    def status_selection(self, other_agents, perspective_identity=None):
        """
        - other_agents: a list of agent IDs (integers).
        - perspective_identity: if provided, we use this identity
          to compute the hypothetical status for each other agent's consumption.
          If None, we use that other agent's actual assigned_group.
        """
        other_status = []
        for agent_id in other_agents:
            consumption_val = self.model.schedule.agents[agent_id].history[-1]

            # Compute the rpro/ranti/rneutral from THIS agent's perspective
            rpro, ranti, rneutral = self.compute_rankings(consumption_val)

            if perspective_identity is not None:
                identity_to_use = perspective_identity
            else:
                identity_to_use = self.model.schedule.agents[agent_id].assigned_group

            # Calculate the partial status for that other agent
            # (i.e., ignoring alpha * share, if you're only interested in
            # picking the 'highest individual_status' or in pure rank ordering)
            other_agent_status = self.calculate_individual_status(
                identity_to_use,
                rpro,
                ranti,
                rneutral
            )
            other_status.append(other_agent_status)
        
        # If we have no other agents, just default
        if not other_status:
            return self.history[-1]

        # Select the agent with the highest status
        highest_status_idx = np.argmax(other_status)
        highest_status_agent_id = other_agents[highest_status_idx]

        # Return that agent's consumption
        return self.model.schedule.agents[highest_status_agent_id].history[-1]

    def step(self):
        """
        The main step for the agent. Picks a partner from self.model.pairs,
        updates beliefs, chooses new consumption, tries to 'convince' partner, etc.
        """

        # Reset effort
        self.effort = 0

        # Identify who I'm paired with this step
        for (ag1, ag2) in self.model.pairs:
            if ag1 == self:
                other_agent = ag2
                break
            elif ag2 == self:
                other_agent = ag1
                break

        # Add the partner's identity & ID
        self.others_identities.append(other_agent.assigned_group)
        self.other_agent_ids.append(other_agent.unique_id)

        # Gather last actions from everyone I've encountered
        self.others_actions = []
        for i in self.other_agent_ids:
            last_action = self.model.schedule.agents[i].history[-1] * self.bias
            self.others_actions.append(last_action)

        # Compute shares for each identity from the agents I've encountered
        steps_taken = len(self.others_identities)
        share_pro = self.others_identities.count("Pro - environment") / steps_taken
        share_neutral = self.others_identities.count("Neutral") / steps_taken
        share_anti = self.others_identities.count("Anti - environment") / steps_taken

        # Update beliefs about min, max, mode
        self.min_believed_consumption = min(self.others_actions) 
        self.max_believed_consumption = max(self.others_actions)
        mode_believed = calculate_mode_hist_midpoint(self.others_actions, bins=10)
        if mode_believed is None:
            mode_believed = self.history[-1]
        self.mode_believed_consumption = mode_believed

        # Include own last consumption in the array
        all_consumptions = self.others_actions + [self.history[-1]]
        consumptions_array = np.array(all_consumptions)

        # Compute agent's own ranks
        rpro_i = 100 - stats.percentileofscore(consumptions_array, self.history[-1])
        ranti_i = stats.percentileofscore(consumptions_array, self.history[-1])
        mode_diff = np.abs(consumptions_array - self.mode_believed_consumption)
        mode_ranks = 1 / (mode_diff + 1e-6)
        self_mode_rank = mode_ranks[-1]
        rneutral_i = stats.percentileofscore(mode_ranks, self_mode_rank)

        # Store the unadjusted ranks
        self.rpro_history.append(rpro_i)
        self.ranti_history.append(ranti_i)
        self.rneutral_history.append(rneutral_i)

        # Adjusted ranks (for incoming efforts)
        adjusted_rpro_i = rpro_i
        adjusted_ranti_i = ranti_i
        adjusted_rneutral_i = rneutral_i

        # Process any incoming efforts
        total_effort_received = 0
        for eff in self.incoming_efforts:
            group_name, effort_amount = eff
            total_effort_received += effort_amount
            if group_name == "Pro - environment":
                adjusted_rpro_i += effort_amount
            elif group_name == "Anti - environment":
                adjusted_ranti_i += effort_amount
            else:  # "Neutral"
                adjusted_rneutral_i += effort_amount

        # Clear efforts
        self.incoming_efforts = []

        # Store the adjusted ranks
        self.alter_rankings['rpro'].append(adjusted_rpro_i)
        self.alter_rankings['ranti'].append(adjusted_ranti_i)
        self.alter_rankings['rneutral'].append(adjusted_rneutral_i)

        # Effort bookkeeping
        self.efforts_received.append(total_effort_received)

        # Compute final status
        if self.assigned_group == "Pro - environment":
            share_own = 100 * share_pro
        elif self.assigned_group == "Anti - environment":
            share_own = 100 * share_anti
        else:
            share_own = 100 * share_neutral

        self.s_i = self.calculate_status(
            self.assigned_group,
            rpro=adjusted_rpro_i,
            ranti=adjusted_ranti_i,
            rneutral=adjusted_rneutral_i,
            alpha=self.alpha,
            share=share_own
        )

        # --- 1) Choose "normal" consumption based on copying highest status among known agents
        # Use *my* list of known agent IDs and compute status with their *actual* identity:
        consumption = self.status_selection(other_agents=self.other_agent_ids, perspective_identity=self.assigned_group)

        # Compute utility
        initial_utility = self.utility(self.assigned_group, consumption, self.s_i)
        utility = initial_utility

        # --- 2) Attempt to convince partner if they're a different identity
        if other_agent.assigned_group != self.assigned_group:
            # Alternative share if partner changes
            if self.assigned_group == "Pro - environment":
                alt_share_own = 100 * (self.others_identities.count("Pro - environment") + 1) / (steps_taken + 1)
                r_own = adjusted_rpro_i
            elif self.assigned_group == "Anti - environment":
                alt_share_own = 100 * (self.others_identities.count("Anti - environment") + 1) / (steps_taken + 1)
                r_own = adjusted_ranti_i
            else:
                alt_share_own = 100 * (self.others_identities.count("Neutral") + 1) / (steps_taken + 1)
                r_own = adjusted_rneutral_i

            alt_si = self.calculate_status(
                self.assigned_group,
                rpro=adjusted_rpro_i,
                ranti=adjusted_ranti_i,
                rneutral=adjusted_rneutral_i,
                alpha=self.alpha,
                share=alt_share_own
            )
            alt_utility = self.utility(self.assigned_group, consumption, alt_si)

            # Probability of turning others
            p_turning = 1 - math.exp(-self.beta * r_own)

            if p_turning == 0:
                self.effort = 0
            else:
                # Minimal condition for paying effort
                if alt_utility - initial_utility > (self.beta * r_own) / p_turning:
                    utility = alt_utility - (self.beta * r_own)
                    self.effort = self.beta * r_own
                    other_agent.incoming_efforts.append((self.assigned_group, self.effort))
                else:
                    self.effort = 0
        else:
            self.effort = 0

        self.efforts_given.append(self.effort)
        self.utilities.append(utility)
        self.history.append(consumption)

        # --- 3) Compute "alter ego" consumption & utility
        # Here we pass perspective_identity=self.alterego, 
        # but still the same list of agent IDs
        alter_s_i = self.calculate_status(
            self.alterego,
            rpro=adjusted_rpro_i,
            ranti=adjusted_ranti_i,
            rneutral=adjusted_rneutral_i,
            alpha=self.alpha,
            share=(share_pro if self.alterego == "Pro - environment"
                   else share_anti if self.alterego == "Anti - environment"
                   else share_neutral)
        )
        self.alter_s_i_history.append(alter_s_i)

        consumption_alter = self.status_selection(
            other_agents=self.other_agent_ids,
            perspective_identity=self.alterego
        )
        utility_alter = self.utility(self.alterego, consumption_alter, alter_s_i)
        self.alternative_utilities.append(utility_alter)

        # --- 4) Identity switching logic every 'steps_convincement' steps
        if steps_taken % self.steps_convincement == 0 and steps_taken >= self.steps_convincement:
            # Compare mean of own utilities vs. alter ego utilities
            own_avg_utility = np.mean(self.utilities[-self.steps_convincement:])
            alterego_avg_utility = np.mean(self.alternative_utilities[-self.steps_convincement:])

            if alterego_avg_utility > own_avg_utility:
                self.assigned_group = self.alterego

            # Recompute average adjusted ranks
            avg_rpro = np.mean(self.alter_rankings['rpro'][-self.steps_convincement:])
            avg_ranti = np.mean(self.alter_rankings['ranti'][-self.steps_convincement:])
            avg_rneutral = np.mean(self.alter_rankings['rneutral'][-self.steps_convincement:])

            status_scores = {
                "Pro - environment": avg_rpro,
                "Anti - environment": avg_ranti,
                "Neutral": avg_rneutral
            }
            # New alter ego is whichever identity has highest rank
            self.alterego = max(status_scores, key=status_scores.get)

            # Reset these for the next round
            self.alternative_utilities = []
            self.alter_s_i_history = []
            self.alter_rankings = {'rpro': [], 'ranti': [], 'rneutral': []}


class status_seeking_model(mesa.Model):
    """
    A model with a number of agents.
    """

    def __init__(self, N, lambda1=1/3, lambda2=1/3, steps_convincement=10,
                 alpha=1, beta=2/3):
        super().__init__()
        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)

        for i in range(self.num_agents):
            a = status_seeking_agent(
                i, self,
                lambda1=lambda1,
                lambda2=lambda2,
                steps_convincement=steps_convincement,
                alpha=alpha,
                beta=beta
            )
            self.schedule.add(a)

        # Initialize DataCollector
        self.datacollector = DataCollector(
            agent_reporters={
                "Group": "assigned_group",
                "Consumption": lambda a: a.history[-1],
                "Utility": lambda a: a.utilities[-1] if a.utilities else None,
                "rpro": lambda a: a.rpro_history[-1] if a.rpro_history else None,
                "ranti": lambda a: a.ranti_history[-1] if a.ranti_history else None,
                "rneutral": lambda a: a.rneutral_history[-1] if a.rneutral_history else None,
                "EffortGiven": lambda a: a.efforts_given[-1] if a.efforts_given else None,
                "EffortReceived": lambda a: a.efforts_received[-1] if a.efforts_received else None,
                "OtherAgentID": lambda a: a.other_agent_ids[-1] if a.other_agent_ids else None,
                "Alterego": "alterego", 
                "AlterUtility": lambda a: a.alternative_utilities[-1] if a.alternative_utilities else None,
            }
        )

    def step(self):
        # Shuffle agents to randomize pairings
        agents_list = self.schedule.agents[:]
        self.random.shuffle(agents_list)

        # Pair agents in consecutive pairs
        self.pairs = []
        for i in range(0, len(agents_list), 2):
            a1 = agents_list[i]
            a2 = agents_list[i+1]
            self.pairs.append((a1, a2))

        # Each agent will know its paired other_agent from self.pairs.
        self.schedule.step()

        # Collect data after all agents have moved
        self.datacollector.collect(self)