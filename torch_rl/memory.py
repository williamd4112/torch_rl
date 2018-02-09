from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random
from typing import overload

import numpy as np

# [reference] https://github.com/matthiasplappert/keras-rl/blob/master/rl/memory.py

# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, goal, action, reward, state1, terminal1')

def sample_batch_indexes(low, high, size):
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        try:
            r = xrange(low, high)
        except NameError:
            r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    @property
    def last_idx(self):
        return (self.length-1 + self.start)%self.maxlen

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


    def pop(self, size):
        if self.length-size <= 0:
            raise Exception("This should not happen")
        if self.start >= size:
            self.start -= size
        else:
            self.length += self.start - size
            self.start = 0

    def __iter__(self):
        for i in range(self.start, self.length):
            yield self[i]

        if self.start > 0:
            for i in range(0, self.start):
                yield self[i]



def zeroed_observation(observation):
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.


class Memory(object):
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, training=True):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_state(self, current_observation):
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        return state

    def get_config(self):
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config

class SequentialMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(SequentialMemory, self).__init__(**kwargs)
        
        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)
        self.goals = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            # Draw random indexes such that we have at least a single entry before each
            # index.
            batch_idxs = sample_batch_indexes(0, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            assert 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                current_terminal = self.terminals[current_idx - 1] if current_idx - 1 > 0 else False
                if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]
            goal = self.goals[idx-1] if self.goals.length > 0 else None

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, goal=goal, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size
        return experiences

    def sample_and_split(self, batch_size, batch_idxs=None):
        experiences = self.sample(batch_size, batch_idxs)

        state0_batch = []
        goal_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            goal_batch.append(e.goal)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        state0_batch = np.array(state0_batch).reshape(batch_size,-1)
        state1_batch = np.array(state1_batch).reshape(batch_size,-1)
        terminal1_batch = np.array(terminal1_batch).reshape(batch_size,-1)
        reward_batch = np.array(reward_batch).reshape(batch_size,-1)
        action_batch = np.array(action_batch).reshape(batch_size,-1)

        if self.goals.length > 0:
            return state0_batch, goal_batch, action_batch, reward_batch, state1_batch, terminal1_batch
        else:
            return state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch

    def _append(self, observation, action, reward, terminal, training=True):
        super(SequentialMemory, self).append(observation, action, reward, terminal, training=training)
        
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    def _appendg(self, observation, goal, action, reward, terminal, training=True):
        self._append(observation, action, reward, terminal, training=training)
        if training:
            self.goals.append(goal)

    def append(self, *args, **kwargs):
        try:
            self._appendg(*args, **kwargs)
        except Exception as e:
            self._append(*args, **kwargs)

    @property
    def nb_entries(self):
        return len(self.observations)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config



class HindsightMemory(Memory):
    """
        Implementation of replay memory for hindsight experience replay with future
        transition sampling.
    """

    def __init__(self, limit, hindsight_size=8, reward_function=lambda observation,goal: 1, **kwargs):
        super(HindsightMemory, self).__init__(**kwargs)
        self.hindsight_size = hindsight_size
        self.reward_function = reward_function
        self.hindsight_buffer = RingBuffer(limit)
        self.goals = RingBuffer(limit)
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

        self.limit = limit
        self.last_terminal_idx = 0

    def append(self, observation, goal, action, reward, terminal, training=True):
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)
            self.goals.append(goal)
            if terminal:
                """
                Sample hindsight_size of states added from recent terminal state to this one.
                """
                self.add_hindsight()
                self.last_terminal_idx = self.goals.last_idx

            super(HindsightMemory, self).append(observation, action, reward, terminal, training=True)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.nb_entries:
            raise KeyError()
        return self.observations[idx], self.goals[idx], self.actions[idx], self.rewards[idx], self.terminals[idx]


    def pop(self):
        """
        Remove last hindsight_size of elements because they were not
        used to generate hindsight.
        """
        self.observations.pop(self.hindsight_size)
        self.actions.pop(self.hindsight_size)
        self.goals.pop(self.hindsight_size)
        self.terminals.pop(self.hindsight_size)
        self.rewards.pop(self.hindsight_size)

    def add_hindsight(self):
        for i in range(self.last_terminal_idx+1, self.observations.last_idx-self.hindsight_size):
            # For every state in episode sample hindsight_size from future states
            # hindsight_idx = sample_batch_indexes(i+1, self.observations.last_idx, self.hindsight_size)
            hindsight_experience = (self.observations.last_idx - i - 1)*[None]
            for j,idx in enumerate(range(i+1, self.observations.last_idx)):
                hindsight_experience[j] = [i,idx]
            self.hindsight_buffer.append(np.asarray(hindsight_experience))

    def sample_and_split(self, num_transitions, batch_idxs=None):
        batch_size = num_transitions*self.hindsight_size + num_transitions
        batch_idxs = sample_batch_indexes(0, self.hindsight_buffer.length, size=num_transitions)

        state0_batch = []
        reward_batch = []
        goal_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for idx in batch_idxs:
            # Add hindsight experience to batch
            hindsight_idxs = sample_batch_indexes(0, len(self.hindsight_buffer[idx]), self.hindsight_size)
            for root_idx, hindsight_idx in self.hindsight_buffer[idx][hindsight_idxs]:
                state0_batch.append(self.observations[hindsight_idx])
                state1_batch.append(self.observations[hindsight_idx+1])
                reward_batch.append(1)
                action_batch.append(self.actions[hindsight_idx])
                goal_batch.append(self.observations[hindsight_idx+1])
            state0_batch.append(self.observations[root_idx])
            state1_batch.append(self.observations[root_idx + 1])
            reward_batch.append(self.rewards[root_idx])
            action_batch.append(self.actions[root_idx])
            goal_batch.append(self.goals[root_idx])

        # Prepare and validate parameters.
        state0_batch = np.array(state0_batch).reshape(batch_size,-1)
        state1_batch = np.array(state1_batch).reshape(batch_size,-1)
        terminal1_batch = np.array(terminal1_batch).reshape(batch_size,-1)
        reward_batch = np.array(reward_batch).reshape(batch_size,-1)
        action_batch = np.array(action_batch).reshape(batch_size,-1)

        return state0_batch, goal_batch, action_batch, reward_batch, state1_batch, terminal1_batch


    @property
    def nb_entries(self):
        return len(self.observations)


class EpisodeParameterMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(EpisodeParameterMemory, self).__init__(**kwargs)
        self.limit = limit

        self.params = RingBuffer(limit)
        self.intermediate_rewards = []
        self.total_rewards = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            batch_idxs = sample_batch_indexes(0, self.nb_entries, size=batch_size)
        assert len(batch_idxs) == batch_size

        batch_params = []
        batch_total_rewards = []
        for idx in batch_idxs:
            batch_params.append(self.params[idx])
            batch_total_rewards.append(self.total_rewards[idx])
        return batch_params, batch_total_rewards

    def append(self, observation, action, reward, terminal, training=True):
        super(EpisodeParameterMemory, self).append(observation, action, reward, terminal, training=training)
        if training:
            self.intermediate_rewards.append(reward)

    def finalize_episode(self, params):
        total_reward = sum(self.intermediate_rewards)
        self.total_rewards.append(total_reward)
        self.params.append(params)
        self.intermediate_rewards = []

    @property
    def nb_entries(self):
        return len(self.total_rewards)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config


