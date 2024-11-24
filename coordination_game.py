################################
# Emissions / Coordination Game
################################

"""
There are three types of agents:
1. Pro-environmental agents
2. Anti-environmental agents
3. Neutral agents

Agents face a different game depending on their identities.

All agents are playing a coordination game on consumption.

Pro-environmental agents know what their consumption means in terms of emissions - g(c) = e -.
Same goes for the anti-environmental agents.
"""

### Libraries ------------------------

import mesa
import random
import numpy as np
from mesa.datacollection import DataCollector
from scipy import stats
import math

### Helpers ------------------------
from helpers import get_distribution
from helpers import calculate_mode_hist_midpoint
from helpers import g_pro, g_neutral, g_anti, maximize_utility

# Choose consumption (later to be included as a parameter, if promising)
consumption_dist = get_distribution(dist_type="uniform", lower=5, upper=100)

### Agents ------------------------
class coordination_agent(mesa.Agent):
    """
    An agent must have:
    1) a defined category
    2) a utility function
    3) a step in the game
    """

    def __init__(self, unique_id, model, lambda1=1/3, lambda2=1/3, alpha=1, beta=2/3,
                 steps_convincement=10):
        """
        Initializes a new instance of the coordination_agent class.

        Args:
            unique_id (int): The unique identifier for the agent.
            model (Model): The model the agent belongs to.
            lambda1 (float): Weight for consumption in utility.
            lambda2 (float): Weight for status in utility.
            alpha (float): Weight for individual status in overall status calculation.
            beta (float): Effort parameter in convincing others.
            steps_convincement (int): Number of steps after which agents consider switching groups.

        Returns:
            None
        """
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        # Assign the first action, which is a sample drawn from a distribution.
        self.history = []
        self.assigned_group = random.choice(["Pro - environment", "Neutral", "Anti - environment"])
        c0 = consumption_dist.rvs(size=1)[0]
        self.history.append(c0)

        # In the basic scenario, agents use sample learning
        self.others_actions = []

        # They also store the identities of others
        self.others_identities = []

        # Store parameters
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.steps_convincement = steps_convincement
        self.alpha = alpha
        self.beta = beta

        # Create a dictionary for alternative identities
        groups = ["Pro - environment", "Neutral", "Anti - environment"]
        other_groups = [g for g in groups if g != self.assigned_group]
        self.alteregos = {g: {'consumption': [], 'utility': [], 's_i': []} for g in other_groups}

        # Placeholder for status component
        self.s_i = 0

        # Initialize effort and incoming efforts
        self.effort = 0
        self.incoming_efforts = []

        # Bias
        self.bias = random.uniform(0.5, 1.5)

        # Also create an empty list for utilities
        self.utilities = []

    def calculate_status(self, identity, rpro, ranti, rneutral, alpha, share):
        """
        Calculates the status of the agent.

        Parameters:
        - identity (str): The agent's identity group.
        - rpro (float): Pro-environmental ranking.
        - ranti (float): Anti-environmental ranking.
        - rneutral (float): Neutral ranking.
        - alpha (float): Weight for individual status in overall status calculation.
        - share (float): Share of agents in the same group.

        Returns:
        - status (float): The calculated status value.
        """
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
        """
        Calculates the utility of the agent based on the outcome or a belief about the game.

        Parameters:
        - identity (str): The agent's identity group.
        - consumption (float): The agent's consumption choice.
        - status (float): The agent's status.

        Returns:
        - u (float): The calculated utility value.
        """
        if identity == "Pro - environment":
            x_hat = self.min_believed_consumption
            g_value = g_pro(consumption)

        elif identity == "Anti - environment":
            x_hat = self.max_believed_consumption
            g_value = g_anti(consumption)

        else:
            x_hat = self.mode_believed_consumption
            g_value = g_neutral(consumption)

        misalignment_cost = - (consumption - x_hat)**2 + g_value
        u = self.lambda1 * consumption + self.lambda2 * status + (1 - self.lambda1 - self.lambda2) * misalignment_cost
        return u

    def consumption_selection(self, identity, status):
        """
        Agents choose consumption to maximize their utility.

        Parameters:
        - identity (str): The agent's identity group.
        - status (float): The agent's status.

        Returns:
        - consumption (float): The chosen consumption level.
        """
        if identity == "Pro - environment":
            x_hat = self.min_believed_consumption
            g_func = g_pro
        elif identity == "Anti - environment":
            x_hat = self.max_believed_consumption
            g_func = g_anti
        else:
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
        """
        Executes one step of the coordination game for the current agent.
        """
        # Reset effort at the beginning of the step
        self.effort = 0

        # Proceed with the rest of the step function
        # Each agent observes another agent's last consumption
        other_agent = self.random.choice(self.model.schedule.agents)
        other_last_consumption = other_agent.history[-1]
        self.others_actions.append(other_last_consumption)

        # Update also the sample of identities (and believed shares)
        self.others_identities.append(other_agent.assigned_group)

        steps_taken = len(self.others_identities)
        share_pro = self.others_identities.count("Pro - environment") / steps_taken
        share_neutral = self.others_identities.count("Neutral") / steps_taken
        share_anti = self.others_identities.count("Anti - environment") / steps_taken

        # Update beliefs based on observed actions
        self.min_believed_consumption = min(self.others_actions) * self.bias
        self.max_believed_consumption = max(self.others_actions) * self.bias
        mode_believed = calculate_mode_hist_midpoint(self.others_actions, bins=10)
        if mode_believed is None:
            mode_believed = self.history[-1]  # Use own consumption if no observations
        self.mode_believed_consumption = mode_believed * self.bias

        ## If status is included:
        # Include own last consumption
        all_consumptions = self.others_actions + [self.history[-1]]
        consumptions_array = np.array(all_consumptions)

        # Pro-environmental ranking: lower consumption gets higher rank
        rpro_i = 100 - stats.percentileofscore(consumptions_array, self.history[-1])

        # Anti-environmental ranking: higher consumption gets higher rank
        ranti_i = stats.percentileofscore(consumptions_array, self.history[-1])

        # Neutral ranking: proximity to mode consumption
        mode_diff = np.abs(consumptions_array - self.mode_believed_consumption)
        mode_ranks = 1 / (mode_diff + 1e-6)
        self_mode_rank = mode_ranks[-1]
        rneutral_i = stats.percentileofscore(mode_ranks, self_mode_rank)

        # Calculate s_i
        if self.assigned_group == "Pro - environment":
            share_own = share_pro
        elif self.assigned_group == "Anti - environment":
            share_own = share_anti
        else:
            share_own = share_neutral

        self.s_i = self.calculate_status(self.assigned_group, rpro=rpro_i, ranti=ranti_i,
                                         rneutral=rneutral_i, alpha=self.alpha, share=share_own)

        # Choose consumption
        consumption = self.consumption_selection(self.assigned_group, self.s_i)

        # Compute initial utility
        initial_utility = self.utility(self.assigned_group, consumption, self.s_i)

        utility = initial_utility  # Initialize utility

        # Determine if agent will exert effort to convince other_agent
        if other_agent.assigned_group != self.assigned_group:
            # Alternative share and rank if other_agent joins
            if self.assigned_group == "Pro - environment":
                alt_share_own = (self.others_identities.count("Pro - environment") + 1) / (steps_taken + 1)
                r_own = rpro_i
            elif self.assigned_group == "Anti - environment":
                alt_share_own = (self.others_identities.count("Anti - environment") + 1) / (steps_taken + 1)
                r_own = ranti_i
            else:
                alt_share_own = (self.others_identities.count("Neutral") + 1) / (steps_taken + 1)
                r_own = rneutral_i

            alt_si = self.calculate_status(self.assigned_group, rpro=rpro_i, ranti=ranti_i,
                                           rneutral=rneutral_i, alpha=self.alpha, share=alt_share_own)
            alt_utility = self.utility(self.assigned_group, consumption, alt_si)

            # Imaginary probability of turning others
            p_turning = 1 - math.exp(-self.beta * r_own)

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

        # For each alter ego, compute s_i, consumption, and utility
        for alterego in self.alteregos.keys():
            if alterego == "Pro - environment":
                share_alterego = share_pro
            elif alterego == "Anti - environment":
                share_alterego = share_anti
            else:
                share_alterego = share_neutral

            # Calculate alter_s_i
            alter_s_i = self.calculate_status(alterego, rpro=rpro_i, ranti=ranti_i,
                                              rneutral=rneutral_i, alpha=self.alpha, share=share_alterego)

            # Compute consumption_alter
            consumption_alter = self.consumption_selection(alterego, alter_s_i)

            # Compute utility_alter
            utility_alter = self.utility(alterego, consumption_alter, alter_s_i)

            # Append to self.alteregos[alterego]
            self.alteregos[alterego]['s_i'].append(alter_s_i)
            self.alteregos[alterego]['consumption'].append(consumption_alter)
            self.alteregos[alterego]['utility'].append(utility_alter)

        # Now process incoming efforts and add them to the corresponding alter ego utilities
        for eff in self.incoming_efforts:
            group_name, effort_amount = eff
            if group_name in self.alteregos:
                # Ensure there's at least one utility entry
                if self.alteregos[group_name]['utility']:
                    self.alteregos[group_name]['utility'][-1] += effort_amount
                else:
                    # Initialize utility if empty
                    self.alteregos[group_name]['utility'].append(effort_amount)
            else:
                pass  # Handle if group_name is not in alteregos
        # Clear incoming efforts after processing
        self.incoming_efforts = []

        # Add parameters
        self.utilities.append(utility)
        self.history.append(consumption)

        # Agents may change their group, depending on the mechanism of conversion
        if steps_taken % self.steps_convincement == 0 and steps_taken >= self.steps_convincement:
            # Compute own average utility over the last steps
            own_avg_utility = np.mean(self.utilities[-self.steps_convincement:])

            # Compute average utilities for each alter ego
            alterego_avg_utilities = {}
            for alterego in self.alteregos.keys():
                alterego_utilities = self.alteregos[alterego]['utility'][-self.steps_convincement:]
                alterego_avg_utility = np.mean(alterego_utilities)
                alterego_avg_utilities[alterego] = alterego_avg_utility

            # Find the alter ego with the highest average utility
            best_alterego = max(alterego_avg_utilities, key=alterego_avg_utilities.get)
            best_alterego_avg_utility = alterego_avg_utilities[best_alterego]

            # If best alter ego has higher utility than own, switch
            if best_alterego_avg_utility > own_avg_utility:
                self.assigned_group = best_alterego

                # Update self.alteregos to contain the new alter egos
                groups = ["Pro - environment", "Neutral", "Anti - environment"]
                other_groups = [g for g in groups if g != self.assigned_group]
                self.alteregos = {g: {'consumption': [], 'utility': [], 's_i': []} for g in other_groups}

class coordination_model(mesa.Model):
    """
    A model with some number of agents.
    """

    def __init__(self, N, lambda1=1/3, lambda2=1/3, steps_convincement=5,
                 alpha=1, beta=2/3):
        """
        Initializes the coordination model.

        Args:
            N (int): Number of agents in the model.
            lambda1 (float): Weight for consumption in utility.
            lambda2 (float): Weight for status in utility.
            steps_convincement (int): Number of steps after which agents consider switching groups.
            alpha (float): Weight for individual status in overall status calculation.
            beta (float): Effort parameter in convincing others.

        Returns:
            None
        """
        super().__init__()
        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)
        for i in range(self.num_agents):
            a = coordination_agent(i, self, lambda1=lambda1, lambda2=lambda2,
                                   steps_convincement=steps_convincement,
                                   alpha=alpha, beta=beta)
            self.schedule.add(a)

        # Initialize DataCollector
        self.datacollector = DataCollector(
            agent_reporters={
                "Group": "assigned_group",
                "Consumption": lambda a: a.history[-1],
                "Utility": lambda a: a.utilities[-1] if a.utilities else None,
            })

    def step(self):
        """
        Advance the model by one step.
        """
        # Collect data
        self.datacollector.collect(self)
        # The model's step will go here; for now, this will call the step method of each agent
        self.schedule.step()
