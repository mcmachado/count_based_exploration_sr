# Author: Marlos C. Machado
import csv
import random
import numpy as np


class MDP:
    def __init__(self, mdp_specification):
        self.time_step = 0
        self.time_limit = np.inf
        self.state_set = set()
        self.current_state = 0
        self.terminal_states = set()
        self._full_action_set = set()
        self._per_state_action_set = []

        self._extract_state_action_sets(mdp_specification)

        for _ in range(self.get_num_states()):
            self._per_state_action_set.append(set())
        self._per_state_action_set = np.array(self._per_state_action_set)
        # TODO: I don't have to enumerate all states to define the start state distribution, just those diff. than zero
        self.distribution_start_state = np.zeros(self.get_num_states())
        self.reward_function = np.zeros((self.get_num_states(), self.get_total_num_actions(), self.get_num_states()))
        self.transition_matrix = np.zeros((self.get_num_states(), self.get_total_num_actions(), self.get_num_states()))
        self._fill_transition_matrix(mdp_specification)

        self._sanity_check()
        self.reset()  # we actually need to make sure the start state is sampled from the defined distribution

    def _extract_state_action_sets(self, file_path):
        with open(file_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            # TODO: Ideally I should check if there is a header first
            next(reader, None)  # skip the headers
            for row in reader:
                if len(row) == 0:  # skip empty lines
                    continue
                elif row[0] == 'start':
                    continue
                elif row[0] == 'terminal':
                    continue
                elif row[0] == 'time_limit':
                    self.time_limit = int(row[1].replace(" ", ""))
                else:
                    # TODO: Allow strings to be used in the specification file, properly mapping them to integers here
                    self.state_set.add(int(row[0].replace(" ", "")))
                    self.state_set.add(int(row[2].replace(" ", "")))
                    self._full_action_set.add(int(row[1].replace(" ", "")))

    def _fill_transition_matrix(self, mdp_specification):
        with open(mdp_specification) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            # TODO: Ideally I should check if there is a header first
            next(reader, None)  # skip the headers
            for row in reader:
                if len(row) == 0:  # skip empty lines
                    continue
                elif row[0] == 'start':
                    self.distribution_start_state[int(row[1].replace(" ", ""))] = float(row[2].replace(" ", ""))
                elif row[0] == 'terminal':
                    self.terminal_states.add(int(row[1].replace(" ", "")))
                elif row[0] == 'time_limit':
                    continue
                else:
                    # TODO: Allow strings to be used in the specification file, properly mapping them to integers here
                    curr_s = int(row[0].replace(" ", ""))
                    action = int(row[1].replace(" ", ""))
                    next_s = int(row[2].replace(" ", ""))
                    self.transition_matrix[curr_s][action][next_s] = float(row[4].replace(" ", ""))
                    self._per_state_action_set[curr_s].add(action)
                    self.reward_function[curr_s][action][next_s] = float(row[3].replace(" ", ""))

    def _sanity_check(self):
        # We check if trans. prob. (s,a) sum to 1, if the action is available on that state, or to 0 otherwise
        for curr_s in self.get_state_set():
            for action in self.get_action_set(curr_s):
                out_trans_prob = np.sum(self.transition_matrix[curr_s][action])
                assert out_trans_prob == 1.0 or out_trans_prob == 0.0,\
                    "The transition probabilities out of state %d do not sum up to 1.0" % curr_s

        # We also make sure that no state has an empty action set
        for state in self.get_state_set():
            assert len(self.get_action_set(state)) > 0, "The action set of state %s is empty!" % state

        # We also have to make sure we actually have a probability distribution in the start state (sum up to 1.0):
            assert np.abs(np.sum(self.distribution_start_state) - 1.0) < 10e-5,\
                "The probabilities of the start state distribution don't sum up to 1.0, but to %f!" \
                % np.sum(self.distribution_start_state)

    def _get_next_state(self, action):
        sampled = random.uniform(0.0, 1.0)
        acum_prob = 0.0
        # TODO: I should keep a list of adj. states to the current state to avoid iterating over all states all the time
        landing_state = -1
        for next_state in self.get_state_set():
            if self.transition_matrix[self.current_state][action][next_state] > 0.0:
                if sampled < self.transition_matrix[self.current_state][action][next_state] + acum_prob:
                    landing_state = next_state
                    break
                else:
                    acum_prob += self.transition_matrix[self.current_state][action][next_state]
        return landing_state

    def get_next_state_and_reward(self, state, action):
        # One step forward model: return the next state and reward given an observation.
        if state in self.terminal_states:
            return state, 0.0  # in case it is the absorbing state encoding end of an episode

        # First I'll save the original state the agent is on
        original_current_state = self.current_state
        # Now I can reset the agent to the state I was told to
        self.current_state = state

        # Now I can ask what will happen next in this new state
        landing_state = self._get_next_state(action)
        return landing_state, self.reward_function[self.current_state][action][landing_state]

        # We need to restore the original state in the environment:
        self.current_state = original_current_state

    def get_state_set(self):
        return self.state_set

    def get_num_states(self):
        return len(self.state_set)

    def get_total_num_actions(self):
        return len(self._full_action_set)

    def get_action_set(self, state):
        assert state < self.get_num_states()
        return self._per_state_action_set[state]

    def act(self, action):
        if self.is_terminal():
            return 0.0

        assert action in self.get_action_set(self.current_state),\
            "The action %d is not available in state %d!" % (action, self.current_state)

        landing_state = self._get_next_state(action)

        assert landing_state != -1, "Something went wrong, I couldn't find a next state for the agent!"
        assert self.transition_matrix[self.current_state][action][landing_state] > 0.0,\
            "Something went wrong, you landed in a state you were not supposed to!"

        observed_reward = self.reward_function[self.current_state][action][landing_state]
        self.current_state = landing_state
        self.time_step += 1
        return observed_reward

    def get_current_state(self):
        return self.current_state

    def is_terminal(self):
        time_is_over = self.time_step >= self.time_limit
        absorbing_state = self.current_state in self.terminal_states
        return time_is_over or absorbing_state

    def reset(self):
        sampled = random.uniform(0.0, 1.0)
        acum_prob = 0.0
        for next_state in range(self.get_num_states()):
            if self.distribution_start_state[next_state] > 0.0:
                if sampled < self.distribution_start_state[next_state] + acum_prob:
                    self.current_state = next_state
                    break
                else:
                    acum_prob += self.distribution_start_state[next_state]

        self.time_step = 0

    def get_time_step(self):
        return self.time_step
