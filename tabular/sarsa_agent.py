# Author: Marlos C. Machado

from agent import Agent
import numpy as np


class Sarsa(Agent):

    def __init__(self, env, step_size, gamma, epsilon):
        super().__init__(env)
        self.gamma = gamma
        self.alpha = step_size
        self.epsilon = epsilon
        self.curr_s = self.env.get_current_state()

        # self.state_visitation_count = np.array([0, 0, 0, 0, 0, 0])

    def step(self):
        curr_a = self.epsilon_greedy(self.q[self.curr_s], epsilon=self.epsilon)
        r = self.env.act(curr_a)
        next_s = self.env.get_current_state()
        next_a = self.epsilon_greedy(self.q[next_s], epsilon=self.epsilon)
        # self.state_visitation_count[next_s] += 1

        self.update_q_values(self.curr_s, curr_a, r, next_s, next_a)

        self.curr_s = next_s

        self.current_undisc_return += r
        if self.env.is_terminal():
            self.episode_count += 1
            self.total_undisc_return += self.current_undisc_return
            self.current_undisc_return = 0

    def update_q_values(self, s, a, r, next_s, next_a):
        self.q[s][a] = self.q[s][a] + self.alpha * (r + self.gamma - (1.0 - self.env.is_terminal()) *
                                                    self.q[next_s][next_a] - self.q[s][a])
