# Author: Marlos C. Machado

import utils
import random
import numpy as np
from abc import ABC, abstractmethod


class Agent(ABC):
    """
    Superclass all the other agents have to extend.
    """
    def __init__(self, env):
        self.env = env
        self.q = np.zeros((env.get_num_states(), env.get_total_num_actions()))
        self.episode_count = 0
        self.total_undisc_return = 0.0
        self.current_undisc_return = 0.0
        super().__init__()

    def epsilon_greedy(self, q_values, epsilon):
        sampled = random.uniform(0.0, 1.0)
        action_set = list(self.env.get_action_set(self.env.get_current_state()))
        if sampled < epsilon:
            return random.choice(action_set)
        else:
            return utils.argmax(q_values[action_set]) # I'm breaking ties randomly

    def get_avg_undisc_return(self):
        return self.total_undisc_return/self.episode_count

    def get_current_undisc_return(self):
        return self.current_undisc_return

    @abstractmethod
    def step(self):
        pass
