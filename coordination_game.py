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

### Helpers ------------------------
from helpers import trunc_normal_dist
from helpers import calculate_mode_hist_midpoint
from helpers import g_pro, g_neutral, g_anti, maximize_utility


### Agents ------------------------
class coordination_agent(mesa.Agent):
    """
    An agent must have:
    1) a defined category
    2) an utility function
    3) a step in the game
    """

    def __init__(self, unique_id, model, alpha = 1, theta = 1, convincement_type = "ticker"):
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
        c0 = trunc_normal_dist.rvs(size=1)[0]
        self.history.append(c0)

        # In the basic scenario, agents use sample learning
        self.others_actions = []

        # Store parameters
        self.alpha = alpha
        self.theta = theta
        self.convincement_type = convincement_type

        # Also create an empty list for utilities
        self.utilities = []

    def utility(self, identity, consumption):
        """
        Calculates the utility of the agent based on the outcome or a belief about the game.
        It depends on the identity (the game played), and needs that the agent has already a belief.

        Each agent tries to align with someone. Pro-environmental substract g(c) = e from utility (emissions), 
        and anti-environmental add g(c) = e to utility (emissions). 

        In the simplest case, g(c) = e.

        """

        if identity == "Pro - environment":
              
            u = self.alpha * consumption - self.theta * (self.min_believed_consumption - consumption)**2 - consumption

        elif identity == "Anti - environment":
            u = self.alpha * consumption - self.theta * (self.max_believed_consumption - consumption)**2 + consumption

        else:
            u = self.alpha * consumption - self.theta * (self.mode_believed_consumption - consumption)**2

        return u
    
    def consumption_selection(self, identity):
        """
        Agents choose consumption to maximize their utility.
        """
        if identity == "Pro - environment":
            consumption = maximize_utility(self.alpha, self.min_believed_consumption, g_pro)

        elif identity == "Anti - environment":
            consumption = maximize_utility(self.alpha, self.max_believed_consumption, g_anti)

        else:
            consumption = maximize_utility(self.alpha, self.mode_believed_consumption, g_neutral)

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
        self.min_believed_consumption = min(self.others_actions)
        self.max_believed_consumption = max(self.others_actions)
        self.mode_believed_consumption = calculate_mode_hist_midpoint(self.others_actions, bins=10)

        # Choose consumption and update utility
        consumption = self.consumption_selection(self.assigned_group)
        utility = self.utility(self.assigned_group, consumption)
        self.utilities.append(utility)
        self.history.append(consumption)

class coordination_model(mesa.Model):
    """
    A model with some number of agents.
    """

    def __init__(self, N, alpha = 1, theta = 1, convincement_type = "ticker"):
        super().__init__()
        self.num_agents = N
        # Create scheduler and assign it to the model
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            a = coordination_agent(i, self, alpha = alpha, theta = theta, convincement_type = convincement_type)
            # Add the agent to the scheduler
            self.schedule.add(a)

    def step(self):
        """
        Advance the model by one step.
        """
        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.schedule.step()










