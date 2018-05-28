import numpy as np

from . import CircularBuffer, SumTree


class ExperienceReplay:
    def __init__(self, size, input_height, input_width, frame_history_size, gamma, alpha):
        """
        Constructor of the experience replay buffer.
        :param size: total size of the experience replay buffer.
        :param gamma: gamma value to be used when computing the monte-carlo return in this buffer.
        :param alpha: Alpha used in Schaul et al. (2016) for prioritized experience replay.
        """
        self._size = size
        self._frame_history_size = frame_history_size
        self._gamma = gamma
        self._alpha = alpha

        self._idx_last_updated_return = -1
        self._max_priority = 1.0

        self.tree = SumTree(size)
        self.observations = CircularBuffer(
            size, (input_height, input_width, 1), np.uint8, fetch_wrap=False)
        self.actions = CircularBuffer(size, (), np.uint8, fetch_wrap=False)
        self.rewards = CircularBuffer(size, (), np.float, fetch_wrap=False)
        self.terminals = CircularBuffer(size, (), np.bool, fetch_wrap=False)
        self.mc_return = CircularBuffer(size, (), np.float, fetch_wrap=False)

    def __len__(self):
        """
        :return: current length of experience replay buffer.
        """
        return len(self.observations)

    def _get_full_observations(self, samples, batch_size):
        full_observation = np.empty(
            (batch_size, 84, 84, self._frame_history_size), dtype=np.uint8)
        batch_index = 0
        for start_idx in samples:
            assert np.abs(start_idx) <= self._size - 1
            start_range_idx = start_idx - (self._frame_history_size - 1)
            end_range_idx = start_range_idx + self._frame_history_size

            frame_index_range = np.arange(
                start_range_idx, end_range_idx, dtype=np.int)
            terminal = np.argwhere(self.terminals[frame_index_range])
            assert len(frame_index_range) == self._frame_history_size
            assert frame_index_range[self._frame_history_size - 1] == start_idx
            # assert len(terminal) <= 1

            full_observation[batch_index] = np.concatenate(
                self.observations[frame_index_range], axis=2)
            if len(terminal) > 0:
                full_observation[batch_index, :, :, :np.squeeze(terminal[-1])] = 0
            batch_index += 1

        return full_observation

    def _get_importance_weights(self, samples, batch_size, beta):
        assert len(samples) == batch_size
        importance_weights = np.ones(batch_size, dtype=np.float)
        num_samples = len(self.observations)
        batch_index = 0
        for sample in samples:
            importance_weights[batch_index] = (num_samples * (self.tree[sample] / self.tree.root)) ** (-beta)
            batch_index += 1
        importance_weights /= np.amax(importance_weights)
        return importance_weights

    def _compute_mc_return(self):
        """
        Once an episode is over, this method is called to compute what was the monte-carlo return of each state in
        that episode. This is done through dynamic programming where G(s) = r(s) + gamma * G(s') and G(terminal) = 0.
        """
        assert self.mc_return.start == self.terminals.start
        assert self.mc_return.length == self.terminals.length
        assert self.mc_return.start == self.rewards.start
        assert self.mc_return.length == self.rewards.length

        current_idx = (self.mc_return.start +
                       self.mc_return.length - 1) % self.mc_return.maxlen
        self.mc_return.data[current_idx] = self.rewards.data[current_idx]

        if current_idx > self._idx_last_updated_return:
            num_updates = current_idx - self._idx_last_updated_return
        else:
            num_updates = current_idx + (self.mc_return.maxlen - self._idx_last_updated_return)

        for i in range(1, num_updates):
            idx = (current_idx - i + self.mc_return.maxlen) % self.mc_return.maxlen
            next_idx = (idx + 1) % self.mc_return.maxlen
            self.mc_return.data[idx] = self.rewards.data[idx] + self._gamma * self.mc_return.data[next_idx]

        # Sanity check we didn't update the return of a state from a different episode
        if self._idx_last_updated_return > 0:
            assert self.terminals.data[self._idx_last_updated_return]
        # Sanity check that the first updated value is from a terminal state
        assert self.terminals.data[current_idx]

        self._idx_last_updated_return = current_idx

    def append(self, obs, action, reward, terminal):
        """
        Add a new entry into the experience replay buffer. If the state being added is a terminal state, the monte carlo
        return of each state seen throughout the current episode is calculated and properly set.
        :param obs: observed frame.
        :param action: action taken.
        :param reward: observed reward signal.
        :param terminal: flag indicating whether this sample corresponds to the last state in the episode.
        """
        self.tree.append(self._max_priority ** self._alpha)
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        self.mc_return.append(None)
        if terminal:
            self._compute_mc_return()

    def sample(self, batch_size, beta=0., segment_samples=False):
        """
        Returns x samples randomly drawn from the experience replay buffer, where x is the batch_size.
        If the sample drawn is a terminal state, or if it is part of the current episode (with its
        monte-carlo return having not being computed yet), it is re-drawn.
        :param batch_size: the number of samples to be returned.
        :param beta: amount of importance sampling correction. Set to 0 for none.
        :param segment_samples: setting to true will sample from `batch_size` equal segments

        :return: batch_size samples of
            (1) game screens,
            (2) actions taken,
            (3) rewards seen,
            (4) resulting next game screen,
            (5) whether the state is terminal or not, and
            (6) the monte-carlo return of each one of the states.
        """
        assert len(self.observations) >= batch_size
        samples = np.empty(batch_size, dtype=np.int)
        samples_drawn = 0

        segment = self.tree.root / batch_size
        while samples_drawn < batch_size:
            if segment_samples:
                sample = np.random.uniform(segment * samples_drawn, segment * (samples_drawn + 1))
            else:
                sample = np.random.uniform() * self.tree.root
            sample_idx = self.tree.find(sample)

            # Check if the obtained sample is from the current episode, if so re-sample
            if np.isnan(self.mc_return[sample_idx]):
                continue
            samples[samples_drawn] = sample_idx
            samples_drawn += 1
        assert len(samples) == batch_size

        batch_observations = self._get_full_observations(samples - 1, batch_size)
        batch_rewards = self.rewards[samples]
        batch_actions = self.actions[samples]
        batch_next_observation = self._get_full_observations(samples, batch_size)
        batch_terminals = self.terminals[samples]
        batch_mc_return = self.mc_return[samples]
        importance_weights = self._get_importance_weights(samples, batch_size, beta)

        return samples, batch_observations, batch_actions, batch_rewards, \
            batch_next_observation, batch_terminals, batch_mc_return, importance_weights

    def update_priorities(self, samples, priorities):
        assert len(samples) == len(priorities)
        for idx, priority in zip(samples, priorities):
            assert priority > 0
            self.tree.update(idx, priority ** self._alpha)
            self._max_priority = max(self._max_priority, priority)
