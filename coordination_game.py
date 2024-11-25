################################
# Emissions / Coordination Game
################################

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

        # Initialize alterego and alternative utilities
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
        self.bias = random.uniform(0.5, 1)

        # Also create an empty list for utilities
        self.utilities = []

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

        else:
            x_hat = self.mode_believed_consumption
            g_value = g_neutral(consumption)

        misalignment_cost = - (consumption - x_hat)**2 + g_value
        u = self.lambda1 * consumption + self.lambda2 * status + (1 - self.lambda1 - self.lambda2) * misalignment_cost
        return u

    def consumption_selection(self, identity, status):
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
        # Reset effort at the beginning of the step
        self.effort = 0

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

        # Initialize rankings with current values
        adjusted_rpro_i = rpro_i
        adjusted_ranti_i = ranti_i
        adjusted_rneutral_i = rneutral_i

        # Process incoming efforts and adjust rankings
        for eff in self.incoming_efforts:
            group_name, effort_amount = eff
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

        # Calculate s_i with adjusted rankings
        if self.assigned_group == "Pro - environment":
            share_own = share_pro
        elif self.assigned_group == "Anti - environment":
            share_own = share_anti
        else:
            share_own = share_neutral

        self.s_i = self.calculate_status(self.assigned_group, rpro=adjusted_rpro_i, ranti=adjusted_ranti_i,
                                         rneutral=adjusted_rneutral_i, alpha=self.alpha, share=share_own)

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
                r_own = adjusted_rpro_i
            elif self.assigned_group == "Anti - environment":
                alt_share_own = (self.others_identities.count("Anti - environment") + 1) / (steps_taken + 1)
                r_own = adjusted_ranti_i
            else:
                alt_share_own = (self.others_identities.count("Neutral") + 1) / (steps_taken + 1)
                r_own = adjusted_rneutral_i

            alt_si = self.calculate_status(self.assigned_group, rpro=adjusted_rpro_i, ranti=adjusted_ranti_i,
                                           rneutral=adjusted_rneutral_i, alpha=self.alpha, share=alt_share_own)
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

        # Add parameters
        self.utilities.append(utility)
        self.history.append(consumption)

        # Calculate alter_s_i
        if self.alterego == "Pro - environment":
            share_alterego = share_pro
        elif self.alterego == "Anti - environment":
            share_alterego = share_anti
        else:
            share_alterego = share_neutral

        alter_s_i = self.calculate_status(self.alterego, rpro=adjusted_rpro_i, ranti=adjusted_ranti_i,
                                            rneutral=adjusted_rneutral_i, alpha=self.alpha, share=share_alterego)
        self.alter_s_i_history.append(alter_s_i)

        # Compute consumption and utility for alterego
        consumption_alter = self.consumption_selection(self.alterego, alter_s_i)
        utility_alter = self.utility(self.alterego, consumption_alter, alter_s_i)
        self.alternative_utilities.append(utility_alter)

        # Agents may change their group, depending on the mechanism of conversion
        if steps_taken % self.steps_convincement == 0 and steps_taken >= self.steps_convincement:

            # Compute own average utility over the last steps
            own_avg_utility = np.mean(self.utilities[-self.steps_convincement:])
            # Compute average utility for alterego
            alterego_avg_utility = np.mean(self.alternative_utilities[-self.steps_convincement:])

            # If alterego has higher utility, switch
            if alterego_avg_utility > own_avg_utility:
                self.assigned_group = self.alterego

            # Compute average rankings over the last steps_convincement steps
            avg_rpro = np.mean(self.alter_rankings['rpro'][-self.steps_convincement:])
            avg_ranti = np.mean(self.alter_rankings['ranti'][-self.steps_convincement:])
            avg_rneutral = np.mean(self.alter_rankings['rneutral'][-self.steps_convincement:])

            # Determine alterego based on highest average ranking
            status_scores = {
                "Pro - environment": avg_rpro,
                "Anti - environment": avg_ranti,
                "Neutral": avg_rneutral
            }
            # Set alterego to the identity with the highest average ranking
            self.alterego = max(status_scores, key=status_scores.get)

            # Reset alternative utilities and alter_s_i_history
            self.alternative_utilities = []
            self.alter_s_i_history = []
            self.alter_rankings = {'rpro': [], 'ranti': [], 'rneutral': []}
        


class coordination_model(mesa.Model):
    """
    A model with some number of agents.
    """

    def __init__(self, N, lambda1=1/3, lambda2=1/3, steps_convincement=10,
                 alpha=1, beta=2/3):
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
        # Collect data
        self.datacollector.collect(self)
        # Advance the model by one step
        self.schedule.step()
