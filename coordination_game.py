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
        self.bias = self.model.random.uniform(0.5, 1)

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

        # Each agent observes another agent's last consumption
        other_agent = self.model.random.choice(self.model.schedule.agents)
        other_last_consumption = other_agent.history[-1]
        self.others_actions.append(other_last_consumption)

        # Update sample of identities
        self.others_identities.append(other_agent.assigned_group)

        # Store the other agent's ID
        self.other_agent_ids.append(other_agent.unique_id)

        steps_taken = len(self.others_identities)
        share_pro = self.others_identities.count("Pro - environment") / steps_taken
        share_neutral = self.others_identities.count("Neutral") / steps_taken
        share_anti = self.others_identities.count("Anti - environment") / steps_taken

        # Update beliefs based on observed actions
        self.min_believed_consumption = min(self.others_actions) * self.bias
        self.max_believed_consumption = max(self.others_actions) * self.bias
        mode_believed = calculate_mode_hist_midpoint(self.others_actions, bins=10)
        if mode_believed is None:
            mode_believed = self.history[-1]
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
            }
        )

    def step(self):
        # Collect data
        self.datacollector.collect(self)
        # Advance the model by one step
        self.schedule.step()