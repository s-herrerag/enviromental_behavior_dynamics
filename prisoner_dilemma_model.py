#######################################
# ABMs: Random matching with games
#######################################

import pandas as pd
import numpy as np
import mesa
import random
import matplotlib.pyplot as plt
import seaborn as sns

class PD_Agent(mesa.Agent):
    """An agent with fixed initial category."""

    def __init__(self, unique_id, model, tax, tax_type = "fixed"):
        """
        Initializes a new instance of the PD_Agent class.

        Args:
            unique_id (int): The unique identifier for the agent.
            model (Model): The model the agent belongs to.
            tax (float): The tax amount.
            tax_type (str, optional): The type of tax. Defaults to "fixed".

        Returns:
            None
        """
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        # Assign the first action, which is equal to the group.
        self.history = []
        self.assigned_group = random.randint(0,1)
        if self.assigned_group == 0:
            self.assigned_group = "Pro - environment"
            action = "Pro - environment action"
            
        else:
            self.assigned_group = "Anti - environment"
            action = "Anti - environment action"
        
        self.history.append(action)

        # Store the tax attribute in the agent instance
        self.tax = tax
        self.tax_type = tax_type
        
    
    def step(self):
        """
        Executes one step of the Prisoner's Dilemma game for the current agent.

        This function simulates a Prisoner's Dilemma game between the current agent and a randomly chosen other agent.
        It calculates the payoffs and choices for both agents based on their decisions.
        The utility function is calculated based on the assigned category and the number of anti-environmental actions.

        Parameters:
            self (PD_Agent): The current agent.
            other_agent (PD_Agent): The other agent in the game.

        Returns:
            None

        """
        # Utility function and the step is appending the choice to the history

        # Now a modification, players play a Prisoner's dilemma
        other_agent = self.random.choice(self.model.schedule.agents)
        
        def prisoner_dilemma(self, other_agent):
            """
            This function simulates a Prisoner's dilemma game between the current agent and a randomly chosen other agent.
            
            Parameters:
                other_agent (PD_Agent): The other agent in the game.
            
            Returns:
                dict: A dictionary containing the payoffs and choices for both agents.
            """

            # Payoffs: T (temptation), R (reward), P (punishment), S (sucker)
            T, R, P, S = 3, 2, 1, 0
            # Random match with another player
            
            # Play the game
            if self.assigned_group == "Pro - environment":
                self_choice = random.choices([True, False], weights=[0.8, 0.2])[0]
            
            else:
                self_choice = random.choices([True, False], weights=[0.2, 0.8])[0]
            
            if other_agent.assigned_group == "Pro - environment":
                other_agent_choice = random.choices([True, False], weights=[0.8, 0.2])[0]
            
            else:
                other_agent_choice = random.choices([True, False], weights=[0.2, 0.8])[0]

            # Determine outcomes based on decisions
            if self_choice and other_agent_choice:
                # Both cooperate: get R
                self_payoff = R
                other_agent_payoff = R
            elif self_choice and not other_agent_choice:
                # You cooperate, other defects: You get S, other gets T
                self_payoff = S
                other_agent_payoff = T
            elif not self_choice and other_agent_choice:
                # You defect, other cooperates: You get T, other gets S
                self_payoff = T
                other_agent_payoff = S
            else:
                # Both defect: get P
                self_payoff = P
                other_agent_payoff = P
            
            return {"self_payoff":self_payoff, "other_payoff":other_agent_payoff, "self_choice":self_choice, "other_choice":other_agent_choice}

        def utility_function(self, other_agent):
            """
            This function calculates the utility and choice of the current agent based on the assigned category and the number of anti-environmental actions.
            
            Parameters:
                self (PD_Agent): The current agent.
                other_agent (PD_Agent): The other agent in the game.
            
            Returns:
                dict: A dictionary containing the choice and utility of the current agent.
            """
            # First, look at the history of everyone
            histories_from_everyone = []
            for a in self.model.schedule.agents:
                histories_from_everyone.extend(a.history)
            
            # Now count the number of anti-environment actions, the nature discounts a fraction of this.
            anti_actions_count = histories_from_everyone.count("Anti - environment action")

            # Play the game
            payoffs_choices = prisoner_dilemma(self, other_agent)

            # Now, the utilities and the choice depend on the assigned category and the number of anti-environmental actions

            if payoffs_choices["self_choice"] == True:
                utility = payoffs_choices["self_payoff"] - (1/10**4)*anti_actions_count
                choice = "Pro - environment action"

            if payoffs_choices["self_choice"] == False:
                utility = payoffs_choices["self_payoff"] - (1/10**4)*anti_actions_count - self.tax
                choice = "Anti - environment action"
            
            return {"choice":choice, "utility":utility}
        

        #Run the function and step forward
        ran_function = utility_function(self, other_agent)
        self.choice = ran_function["choice"]
        self.history.append(self.choice)
        self.utility = ran_function["utility"]

        if self.tax_type == "incremental":
            self.tax = self.tax + 0.005

class PD_Model(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, tax, tax_type = "fixed"):
        super().__init__()
        self.num_agents = N
        # Create scheduler and assign it to the model
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            a = PD_Agent(i, self, tax, tax_type)
            # Add the agent to the scheduler
            self.schedule.add(a)

    def step(self):
        """Advance the model by one step."""
        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.schedule.step()


# Now, what if people followed a tit-for-tat strategy? #################################

class tit_for_tat_Agent(mesa.Agent):
    """An agent with fixed initial category and a tit-for-tat strategy."""

    def __init__(self, unique_id, model, tax, tax_type="fixed"):
        """
        Initializes a new instance of the PD_Agent class.
        """
        super().__init__(unique_id, model)
        
        self.history = []
        self.interaction_history = {}  # Store past interactions with other agents
        self.assigned_group = random.randint(0, 1)
        if self.assigned_group == 0:
            self.assigned_group = "Pro - environment"
            action = "Pro - environment action"
        else:
            self.assigned_group = "Anti - environment"
            action = "Anti - environment action"
        
        self.history.append(action)
        self.tax = tax
        self.tax_type = tax_type
    
    def tit_for_tat(self, other_agent):
        """
        Tit-for-tat strategy: cooperate on the first move, then copy the other agent's last move.
        """
        # If the agent has encountered the other agent before, mimic their last move
        if other_agent.unique_id in self.interaction_history:
            return self.interaction_history[other_agent.unique_id]
        else:
            if self.assigned_group == "Pro - environment": 
                return random.choices([True, False], weights=[0.8, 0.2])[0]

            if self.assigned_group == "Anti - environment": 
                return random.choices([True, False], weights=[0.2, 0.8])[0]

    def prisoner_dilemma(self, other_agent):
        """
        Simulates a Prisoner's dilemma game between the current agent and another agent.
        """
        # Payoffs: T (temptation), R (reward), P (punishment), S (sucker)
        T, R, P, S = 3, 2, 1, 0

        # Apply the tit-for-tat strategy
        self_choice = self.tit_for_tat(other_agent)
        other_agent_choice = other_agent.tit_for_tat(self)

        # Determine outcomes based on decisions
        if self_choice and other_agent_choice:
            self_payoff = R
            other_agent_payoff = R
        elif self_choice and not other_agent_choice:
            self_payoff = S
            other_agent_payoff = T
        elif not self_choice and other_agent_choice:
            self_payoff = T
            other_agent_payoff = S
        else:
            self_payoff = P
            other_agent_payoff = P
        
        # Update interaction history
        self.interaction_history[other_agent.unique_id] = other_agent_choice
        other_agent.interaction_history[self.unique_id] = self_choice

        return {"self_payoff": self_payoff, "other_payoff": other_agent_payoff, 
                "self_choice": self_choice, "other_choice": other_agent_choice}

    def utility_function(self, other_agent):
        """
        Calculates the utility of the current agent based on the assigned category and anti-environmental actions.
        """
        # First, look at the history of everyone
        histories_from_everyone = []
        for a in self.model.schedule.agents:
            histories_from_everyone.extend(a.history)
        
        anti_actions_count = histories_from_everyone.count("Anti - environment action")

        # Play the game
        payoffs_choices = self.prisoner_dilemma(other_agent)

        # Calculate utility based on choice and anti-environmental actions
        if payoffs_choices["self_choice"]:
            utility = payoffs_choices["self_payoff"] - (1 / 10**4) * anti_actions_count
            choice = "Pro - environment action"
        else:
            utility = payoffs_choices["self_payoff"] - (1 / 10**4) * anti_actions_count - self.tax
            choice = "Anti - environment action"
        
        return {"choice": choice, "utility": utility}
    
    def step(self):
        """
        Executes one step of the Prisoner's Dilemma game for the current agent.
        """
        other_agent = self.random.choice(self.model.schedule.agents)
        ran_function = self.utility_function(other_agent)
        self.choice = ran_function["choice"]
        self.history.append(self.choice)
        self.utility = ran_function["utility"]

        if self.tax_type == "incremental":
            self.tax += 0.005

class tit_for_tat_Model(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, tax, tax_type = "fixed"):
        super().__init__()
        self.num_agents = N
        # Create scheduler and assign it to the model
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            a = tit_for_tat_Agent(i, self, tax, tax_type)
            # Add the agent to the scheduler
            self.schedule.add(a)

    def step(self):
        """Advance the model by one step."""
        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.schedule.step()