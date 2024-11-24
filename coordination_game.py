################################
# Emissions / coordination game
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
    2) an utility function
    3) a step in the game
    """

    def __init__(self, unique_id, model, lambda1 = 1/3, lambda2 = 1/3, convincement_type = "ticker", steps_convincement = 10):
        """
        Initializes a new instance of the coordination_agent class.

        Args:
            unique_id (int): The unique identifier for the agent.
            model (Model): The model the agent belongs to.
            alpha (float, optional): The consumption utility parameter. Defaults to 1.
            theta (float, optional): The discount rate of misalignment. Defaults to 1.

        Returns:
            None
        """
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        # Assign the first action, which is a sample drawn from a distribution, and has to be positive (chisq?).
        self.history = []
        self.assigned_group = random.choices(["Pro - environment", "Neutral" ,"Anti - environment"], weights=[1/3, 1/3, 1/3])[0]
        c0 = consumption_dist.rvs(size=1)[0]
        self.history.append(c0)

        # In the basic scenario, agents use sample learning
        self.others_actions = []

        # They also store the identities of others
        self.others_identities = []

        # Store parameters
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.convincement_type = convincement_type
        self.steps_convincement = steps_convincement

        # Define a threshold if convincement_type = "ticker"
        self.threshold_convincement =  0.51 # Agents try to be like the majority

        # Also create an empty list for utilities
        self.utilities = []

        # Finally, create a placeholder for alternative identities
        self.alterego = self.assigned_group
        self.alternative_utilities = []

        # Placeholder for status component
        self.s_i = 0
        self.alter_s_i = 0

        # Bias
        self.bias = random.uniform(0.85, 1.15)

    def calculate_status(self, identity, rpro, ranti, rneutral):
        """
        Calculates the status of the agent. 

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

        status = theta_pro * rpro + theta_anti * ranti + theta_neutral * rneutral

        return status
    
    def utility(self, identity, consumption, status):
        """
        Calculates the utility of the agent based on the outcome or a belief about the game.
        It depends on the identity (the game played), and needs that the agent has already a belief.

        Each agent tries to align with someone. Pro-environmental substract g(c) = e from utility (emissions), 
        and anti-environmental add g(c) = e to utility (emissions). 

        In the simplest case, g(c) = e.

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
        # Each agent observes another agent's last consumption
        other_agent = self.random.choice(self.model.schedule.agents)
        other_last_consumption = other_agent.history[-1]
        self.others_actions.append(other_last_consumption)

        # Update beliefs based on observed actions
        self.min_believed_consumption = min(self.others_actions) * self.bias
        self.max_believed_consumption = max(self.others_actions) * self.bias
        self.mode_believed_consumption = calculate_mode_hist_midpoint(self.others_actions, bins=10) * self.bias

        # Update also the sample of identities (and believed shares)
        self.others_identities.append(other_agent.assigned_group)

        steps_taken = len(self.others_identities)
        share_pro = self.others_identities.count("Pro - environment")/steps_taken
        share_neutral = self.others_identities.count("Neutral")/steps_taken
        share_anti = self.others_identities.count("Anti - environment")/steps_taken

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
        self.s_i = self.calculate_status(self.assigned_group, rpro=rpro_i, ranti=ranti_i, rneutral=rneutral_i)
        self.alter_s_i = self.calculate_status(self.alterego, rpro=rpro_i, ranti=ranti_i, rneutral=rneutral_i)
        
        # If convincement_type is 'ticker', set s_i = 0
        if self.convincement_type == 'ticker':
            self.s_i = 0

        ### Choose consumption and update utilities
        consumption = self.consumption_selection(self.assigned_group, self.s_i)
        utility = self.utility(self.assigned_group, consumption, self.s_i)
        self.utilities.append(utility)
        self.history.append(consumption)

        # Calculate utility for the alterego
        consumption_alter = self.consumption_selection(self.alterego, self.alter_s_i)
        utility_alter = self.utility(self.alterego, consumption_alter, self.alter_s_i)
        self.alternative_utilities.append(utility_alter)


        # Agents may change their group, depending on the mechanism of conversion
        if steps_taken % self.steps_convincement == 0:

            # First thing: if utilities are greater for the alter-ego, switch
            if np.mean(self.utilities[-self.steps_convincement:]) < np.mean(self.alternative_utilities[-self.steps_convincement]):
                self.assigned_group = self.alterego

            if self.convincement_type == "ticker":
                if share_pro >= self.threshold_convincement:
                    self.alterego = "Pro - environment"
                elif share_neutral >= self.threshold_convincement:
                    self.alterego = "Neutral"
                elif share_anti >= self.threshold_convincement:
                    self.alterego = "Anti - environment"

            elif self.convincement_type == "ranking_relatives":
                status_scores = {
                "Pro - environment": rpro_i,
                "Anti - environment": ranti_i,
                "Neutral": rneutral_i
                }
                # Set alterego to the identity with the highest status score
                self.alterego = max(status_scores, key=status_scores.get)
                pass 


class coordination_model(mesa.Model):
    """
    A model with some number of agents.
    """

    def __init__(self, N, lambda1=0.3, lambda2=0.3, convincement_type="ticker", steps_convincement=10):
        super().__init__()
        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)
        for i in range(self.num_agents):
            a = coordination_agent(i, self, lambda1=lambda1, lambda2=lambda2,
                                   convincement_type=convincement_type,
                                   steps_convincement=steps_convincement)
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
        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.schedule.step()










