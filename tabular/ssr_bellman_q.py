# Author: Marlos C. Machado

import sys
import utils
import random
import numpy as np
from agent import Agent

class SSRBellmanQ(Agent):

    def __init__(self, env, gamma, epsilon=0.1, use_learned_model=True, beta=100, theta=0.01):
        super().__init__(env)
        self.beta = beta
        self.theta = theta
        self.gamma = gamma
        self.epsilon = epsilon
        self.use_learned_model = use_learned_model
        self.curr_s = self.env.get_current_state()
        self.num_states = self.env.get_num_states()
        self.total_num_actions = self.env.get_total_num_actions()

        self.visit_count = np.zeros((self.num_states, self.num_states))
        self.transition_count = np.ones((self.num_states, self.total_num_actions, self.num_states))

        self.reward_model = np.zeros((self.num_states, self.total_num_actions, self.num_states))
        self.stochastic_prob = np.ones((self.env.get_num_states(), self.total_num_actions, self.num_states)) * 1./self.num_states
        self.substochastic_prob = np.zeros((self.num_states, self.num_states))

        self.value_function = np.zeros(self.num_states)
        self.pi = np.random.randint(self.total_num_actions, size=self.num_states)

    def _eval_policy(self, p, r):
        # Policy evaluation step.
        delta = 0.0
        for s in self.env.get_state_set():
            v = self.value_function[s]
            v_s = 0.0
            action_set = self.env.get_action_set(s)
            for s_prime in self.env.get_state_set():
                for a in action_set:
                    if a == self.pi[s]:
                        prob_a = 1. - self.epsilon + self.epsilon/len(action_set)
                    else:
                        prob_a = self.epsilon/len(action_set)
                    v_s += prob_a * p[s][self.pi[s]][s_prime] * (r[s][self.pi[s]][s_prime] +
                                                                 self.gamma * self.value_function[s_prime])
            self.value_function[s] = v_s
            delta = max(delta, np.fabs(v - self.value_function[s]))

        return delta

    def _improve_policy(self, p, r):
        # Policy improvement step.
        policy_stable = True
        for s in self.env.get_state_set():
            old_action = self.pi[s]
            action_set = self.env.get_action_set(s)
            temp_v = np.zeros(len(action_set))
            for a in action_set:
                v_s = 0.0
                for s_prime in self.env.get_state_set():
                    v_s += p[s][a][s_prime] * (r[s][a][s_prime] + self.gamma * self.value_function[s_prime])
                temp_v[a] = v_s
            self.pi[s] = utils.argmax(temp_v)

            if old_action != self.pi[s]:
                if np.abs(self.value_function[old_action] - self.value_function[self.pi[s]]) > self.theta:
                    policy_stable = False
        return policy_stable

    def _policy_iteration(self, p, r):
        policy_stable = False
        while not policy_stable:
            # Policy evaluation
            delta = self._eval_policy(p, r)
            while self.theta < delta:
                delta = self._eval_policy(p, r)
            # Policy improvement
            policy_stable = self._improve_policy(p, r)

    def step(self):
        sampled = random.uniform(0, 1)
        if sampled < self.epsilon:
            curr_a = random.choice((0, self.total_num_actions - 1))
        else:
            curr_a = self.pi[self.curr_s]

        r = self.env.act(curr_a)
        next_s = self.env.get_current_state()

        # Update visitation counts, the reward model, the empirical SR and the substochastic SR
        self.visit_count[self.curr_s][next_s] += 1.
        self.transition_count[self.curr_s][curr_a][next_s] += 1
        times_left_s = np.sum(self.visit_count[self.curr_s])
        times_took_sa = np.sum(self.transition_count[self.curr_s][curr_a])

        trans_count = self.transition_count[self.curr_s][curr_a][next_s]
        self.reward_model[self.curr_s][curr_a][next_s] = ((trans_count - 2) * self.reward_model[self.curr_s][curr_a][
            next_s] + r) / (trans_count - 1)

        for s in range(self.num_states):
            self.stochastic_prob[self.curr_s][curr_a][s] = self.transition_count[self.curr_s][curr_a][s] / times_took_sa
            self.substochastic_prob[self.curr_s][s] = self.visit_count[self.curr_s][s] / (times_left_s + 1.)
        substochastic_sr = np.linalg.inv(np.identity(self.env.get_num_states()) - self.gamma * self.substochastic_prob)

        # Compute intrinsic reward
        intr_value = np.dot(substochastic_sr, np.ones((self.env.get_num_states())))
        # This makes the intrinsic reward vector a matrix, where the intrinsic reward is defined only by s', that is,
        # r(s,a,s') is the same for all s and a. I had to do this because all I care about is where I land (for now).
        intr_reward = -1. * np.array([[intr_value, ] * self.total_num_actions,] * self.num_states)

        if self.use_learned_model:
            transition_model = self.stochastic_prob
            general_reward_model = self.reward_model + self.beta * intr_reward
        else:
            transition_model = self.env.transition_matrix
            general_reward_model = self.env.reward_function + self.beta * intr_reward
        self._policy_iteration(transition_model, general_reward_model)

        self.curr_s = next_s

        self.current_undisc_return += r
        if self.env.is_terminal():
            self.episode_count += 1
            self.total_undisc_return += self.current_undisc_return
            self.current_undisc_return = 0
