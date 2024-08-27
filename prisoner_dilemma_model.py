#######################################
# ABMs: Random matching with games
#######################################

import pandas as pd
import numpy as np
import mesa
import random
import matplotlib.pyplot as plt
import seaborn as sns


#### Initial code for Prisoner's Dilemma: No choosing of strategy

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


#### Updated Prisoner's Dilemma
class PD_learning_Agent(mesa.Agent):
    """An agent with fixed initial category."""

    def __init__(self, unique_id, model, tax, tax_type = "fixed", learn_type = "complete_information", theta = 0):
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
        self.assigned_group = random.choices(["Pro - environment", "Anti - environment"], weights=[0.5, 0.5])[0]
        if self.assigned_group == "Pro - environment":
            action = random.choices(["Pro - environment action", "Anti - environment action"], weights=[0.8, 0.2])[0]
            
        else:
            action = random.choices(["Pro - environment action", "Anti - environment action"], weights=[0.2, 0.8])[0]
        
        self.history.append(action)

        # Store the tax attribute in the agent instance
        self.tax = tax
        self.tax_type = tax_type

        # Store the way of learning
        self.learn_type = learn_type

        #Store theta
        self.theta = theta
        
    
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

        def beliefs_update(self, other_agent, type = "complete_information"):
            """
            Updates the beliefs of an agent based on the type of information it has.

            Parameters:
                self (PD_Agent): The current agent.
                other_agent (PD_Agent): The other agent in the game.
                payoffs (dict): A dictionary containing the payoffs for each action.
                type (str, optional): The type of information the agent has. Defaults to "complete_information".

            Returns:
                float: The probability of cooperation.
            """
            if type == "no information":
                # People always know how many people cooperated in t-1
                last_actions = [i.history[-1] for i in self.model.schedule.agents]
                # People do not know about who are cooperators, so their best guess is based on the last actions of everyone
                p_cooperate = last_actions.count("Pro - environment action")/len(last_actions)
            
            else:
                # Here, people know the share of cooperators in the last round, and how many of them cooperated in t-1
                last_actions_groups = dict(zip([i.assigned_group for i in self.model.schedule.agents],[i.history[-1] for i in self.model.schedule.agents]))
                last_actions_groups = pd.DataFrame({"category": last_actions_groups.keys(), "last_action": last_actions_groups.values()})

                behaviors_environmentalists = last_actions_groups.loc[last_actions_groups["category"] == "Pro - environment"]
                behaviors_anti_environmentalists = last_actions_groups.loc[last_actions_groups["category"] == "Anti - environment"]

                # Now calculate the probabilities
                p_environmentalist = len(behaviors_environmentalists)/len(last_actions_groups)
                p_anti_environmentalist = 1 - p_environmentalist

                p_cooperate_environmentalist = behaviors_environmentalists.loc[behaviors_environmentalists["last_action"] == "Pro - environment action"].shape[0]/len(behaviors_environmentalists)
                p_cooperate_anti_environmentalist = behaviors_anti_environmentalists.loc[behaviors_anti_environmentalists["last_action"] == "Pro - environment action"].shape[0]/len(behaviors_anti_environmentalists)
            
            if type == "share of cooperators":
                # Here, people know the share of cooperators in the last round, and how many of them cooperated in t-1

                p_cooperate = p_cooperate_environmentalist*p_environmentalist + p_cooperate_anti_environmentalist*p_anti_environmentalist

            elif type == "complete_information":
                # In this case, the player knows the category of its opponent
                if other_agent.assigned_group == "Pro - environment":
                    p_cooperate = p_cooperate_environmentalist
                else:
                    p_cooperate = p_cooperate_anti_environmentalist
            
            return p_cooperate    

        def expected_utility(self, beliefs, payoffs):

            # Cooperate
            if self.assigned_group == "Pro - environment":
                expected_utility_cooperation = payoffs["R"]*beliefs + payoffs["S"]*(1-beliefs) + self.theta
            else:
                expected_utility_cooperation = payoffs["R"]*beliefs + payoffs["S"]*(1-beliefs)
            
            #Defect
            expected_utility_defect = payoffs["T"]*beliefs + payoffs["P"]*(1-beliefs)

            return {"expected_utility_cooperation": expected_utility_cooperation, "expected_utility_defect": expected_utility_defect}
        
        def prisoner_dilemma(self, other_agent):
            """
            This function simulates a Prisoner's dilemma game between the current agent and a randomly chosen other agent.
            
            Parameters:
                other_agent (PD_Agent): The other agent in the game.
            
            Returns:
                dict: A dictionary containing the payoffs and choices for both agents.
            """

            # Payoffs: T (temptation), R (reward), P (punishment), S (sucker)
            payoffs_prisoner_dilemma = {"T": 3, "R": 2, "P": 1, "S": 0}
            # Random match with another player
            
            # Play the game:
            # 1) Update the behavior of the agents
            self_belief = beliefs_update(self, other_agent, type = self.learn_type)
            other_belief = beliefs_update(other_agent, self, type = other_agent.learn_type)

            # 2) Calculate the expected payoffs
            # 2.1) Self
            self_payoff_cooperate = expected_utility(self, self_belief, payoffs_prisoner_dilemma)["expected_utility_cooperation"]
            self_payoff_defect = expected_utility(self, self_belief, payoffs_prisoner_dilemma)["expected_utility_defect"]
            # 2.2) Other
            other_agent_payoff_cooperate = expected_utility(other_agent, other_belief, payoffs_prisoner_dilemma)["expected_utility_cooperation"]
            other_agent_payoff_defect = expected_utility(other_agent, other_belief, payoffs_prisoner_dilemma)["expected_utility_defect"]

            # 3) Determine the choice
            # 3.1) Self
            if self_payoff_cooperate >= self_payoff_defect:
                self_choice = True
            else:
                self_choice = False
            # 3.2) Other
            if other_agent_payoff_cooperate >= other_agent_payoff_defect:
                other_agent_choice = True
            else:
                other_agent_choice = False

            # 4) Determine the payoffs
            # Determine outcomes based on decisions
            if self_choice and other_agent_choice:
                # Both cooperate: get R
                self_payoff = payoffs_prisoner_dilemma["R"]
                other_agent_payoff = payoffs_prisoner_dilemma["R"]
            elif self_choice and not other_agent_choice:
                # You cooperate, other defects: You get S, other gets T
                self_payoff = payoffs_prisoner_dilemma["S"]
                other_agent_payoff = payoffs_prisoner_dilemma["T"]
            elif not self_choice and other_agent_choice:
                # You defect, other cooperates: You get T, other gets S
                self_payoff = payoffs_prisoner_dilemma["T"]
                other_agent_payoff = payoffs_prisoner_dilemma["S"]
            else:
                # Both defect: get P
                self_payoff = payoffs_prisoner_dilemma["P"]
                other_agent_payoff = payoffs_prisoner_dilemma["P"]
            
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

            elif payoffs_choices["self_choice"] == False:
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


class PD_learning_Model(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, tax, tax_type = "fixed", learn_type = "complete_information", theta = 0):
        super().__init__()
        self.num_agents = N
        # Create scheduler and assign it to the model
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            a = PD_learning_Agent(i, self, tax, tax_type, learn_type=learn_type, theta=theta)
            # Add the agent to the scheduler
            self.schedule.add(a)

    def step(self):
        """Advance the model by one step."""
        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.schedule.step()
