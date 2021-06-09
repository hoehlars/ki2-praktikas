import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == '__main__':
    """
    history = dict()
    history['state'] = np.empty((0, 8), dtype=np.float)
    history['next_state'] = np.empty((0, 8), dtype=np.float)
    history['step'] = np.empty((0), dtype=np.uint16)
    history['action'] = np.empty((0), dtype=np.uint8)
    history['reward'] = np.empty((0), dtype=np.float)
    history['done'] = np.empty((0), dtype=np.bool)
    history['episode'] = np.empty((0), dtype=np.uint16)
    history['len'] = 0
    """

    with open(sys.argv[1], 'rb') as f:
        history = pickle.load(f)
    if len(sys.argv) == 3:
        with open(sys.argv[2], 'rb') as f:
            history2 = pickle.load(f)
        history['reward'] = np.concatenate((history2['reward'],history['reward']))
        history['done'] = np.concatenate((history2['done'],history['done']))

    curr_episode = 0
    reward_per_episode = np.array([], dtype=np.float)
    for i in range(0, len(history['reward'])):
        if reward_per_episode.size > curr_episode:
            reward_per_episode[curr_episode] += history['reward'][i]
        else:
            reward_per_episode = np.append(reward_per_episode, history['reward'][i])

        if (history['done'][i]):
            curr_episode += 1

    print(reward_per_episode)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.plot(range(1, len(reward_per_episode) + 1), reward_per_episode, 'b', label='Episode Reward')
    ax.plot(range(1, len(reward_per_episode) - 99 + 1), moving_average(reward_per_episode, 100), 'r', label='Moving Average (100)')
    # ax[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    ax.set_title('Episode Reward over time', fontsize=16)
    ax.set_xlabel('Episode', fontsize=16)
    ax.set_ylabel('Reward', fontsize=16)
    ax.legend()
    plt.show()