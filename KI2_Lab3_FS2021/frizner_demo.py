#!/usr/bin/env python
# Adapted from https://github.com/frizner/LunarLander-v2
import numpy as np
import gym
import os
import itertools
import sys

from keras.models import load_model

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class LunarLander(object):
    def __init__(self, model, env, min_memory=2**6):
        self._env_ = env
        self._min_memory_ = min_memory

        # Parameters of the environment
        self._n_inputs_ = self._env_.observation_space.shape
        self._n_output_ = self._env_.action_space.n
        self._n_actions_ = self._env_.action_space.n
        self._model_ = model

    # make a prediction
    def predict(self, state):
        return self._model_.predict(state)

    # run agent
    def run(
            self,
            num_episodes=2000,
    ):
        # arrays to save rewards and last rewards
        rewards = np.empty((0), dtype=np.float)
        last_rewards = np.empty((0), dtype=np.float)
        for i in range(num_episodes):
            s = self._env_.reset()
            done = False
            episode_rewards = 0
            while not done:
                reshaped_s = s.reshape(1, -1)

                s_pred = self.predict(reshaped_s)[0]

                # choose the action
                a = np.argmax(s_pred)

                # do it
                next_s, r, done, _ = self._env_.step(a)
                self._env_.render()
                episode_rewards += r

                s = next_s
            print(f'reward: {int(episode_rewards)}')
        return rewards, last_rewards


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    # Grid parameters
    num_learns = [1]
    epochs = [1]
    lookbacks = [30000]
    batch_sizes = [32]

    env.render()

    for num_learn, epoch, lookback, batch_size in itertools.product(num_learns, epochs, lookbacks, batch_sizes):
        model = load_model(sys.argv[1])

        # create agent
        ll = LunarLander(
            model=model,
            env=env
        )

        # let's learn
        rewards = ll.run(
            num_episodes=2000,
        )