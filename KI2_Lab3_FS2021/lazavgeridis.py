# From: https://github.com/lazavgeridis/LunarLander-v2, dqn version
# Adapted for better ploting

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.set_num_threads(8)

class CNN(nn.Module):

    def __init__(self, env_actions):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc    = nn.Linear(3136, 512) # 64 x 7 x 7
        self.out   = nn.Linear(512, env_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv_to_fc(x)
        x = F.relu(self.fc(x))

        return self.out(x)

    def conv_to_fc(self, x):
        size = x.size()[1:] # all dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return x.view(-1, num_features)


class LinearMapNet(nn.Module):
    def __init__(self, input_shape, env_actions):
        super(LinearMapNet, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, env_actions)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        return self.out(x)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.transitions = []
        self.max_capacity = capacity
        self.next_transition_index = 0


    def length(self):
        return len(self.transitions)


    def store(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        if self.next_transition_index >= self.length():
            self.transitions.append(transition)
        else:
            self.transitions[self.next_transition_index] = transition   # overwrite old experiences

        self.next_transition_index = (self.next_transition_index + 1) % self.max_capacity


    def sample_minibatch(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for _ in range(batch_size):
            transition_index = random.randint(0, self.length() - 1)
            transition = self.transitions[transition_index]
            state, action, reward, next_state, done = transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return states, actions, rewards, next_states, dones

def build_qnetwork(env_actions, learning_rate, input_shape, network, device):
    if network == 'cnn':
        qnet = CNN(env_actions)
    else:
        # model = 'linear'
        qnet = LinearMapNet(input_shape, env_actions)
    return qnet.to(device), torch.optim.RMSprop(qnet.parameters(), lr=learning_rate)

def lmn_input(obs):
    net_input = np.expand_dims(obs, 0)
    net_input = torch.from_numpy(net_input)

    return net_input

def epsilon_greedy(q_func, state, eps, env_actions):
    prob = np.random.random()

    if prob < eps:
        return random.choice(range(env_actions))
    elif isinstance(q_func, CNN) or isinstance(q_func, LinearMapNet):
        with torch.no_grad():
            return q_func(state).max(1)[1].item()
    else:
        qvals = [q_func[state + (action, )] for action in range(env_actions)]
        return np.argmax(qvals)

def decay_epsilon(curr_eps, exploration_final_eps):
    if curr_eps < exploration_final_eps:
        return curr_eps
    
    return curr_eps * 0.996

def fit(qnet, qnet_optim, qtarget_net, loss_func, \
        frames, actions, rewards, next_frames, dones, \
        gamma, env_actions, device):

    # compute action-value for frames at timestep t using q-network
    frames_t = torch.cat(frames).to(device)
    actions = torch.tensor(actions, device=device)
    q_t = qnet(frames_t) # q_t tensor has shape (batch, env_actions)
    q_t_selected = torch.sum(q_t * torch.nn.functional.one_hot(actions, env_actions), 1) 

    # compute td targets for frames at timestep t + 1 using q-target network
    dones = torch.tensor(dones, device=device)
    rewards = torch.tensor(rewards, device=device)
    frames_tp1 = torch.cat(next_frames).to(device)
    q_tp1_best = qtarget_net(frames_tp1).max(1)[0].detach() 
    ones = torch.ones(dones.size(-1), device=device)
    q_tp1_best = (ones - dones) * q_tp1_best
    q_targets = rewards + gamma * q_tp1_best

    # td error
    loss = loss_func(q_t_selected, q_targets)
    qnet_optim.zero_grad()
    loss.backward()
    qnet_optim.step()
    #return loss.item()

def update_target_network(qnet, qtarget_net):
    qtarget_net.load_state_dict(qnet.state_dict())

def save_model(qnet, episode, path):
    torch.save(qnet.state_dict(), os.path.join(path, 'qnetwork_{}.pt'.format(episode)))

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_rewards(chosen_agents, agents_returns, num_episodes, window):
    """
    num_intervals = int(num_episodes / window)
    for agent, agent_total_returns in zip(chosen_agents, agents_returns):
        print(len(agent_total_returns))
        print("\n{} lander average reward = {}".format(agent, sum(agent_total_returns) / num_episodes))
        l = []
        for j in range(num_intervals):
            l.append(round(np.mean(agent_total_returns[j * 100 : (j + 1) * 100]), 1))
        plt.plot(range(0, num_episodes, window), l)
    plt.xlabel("Episodes")
    plt.ylabel("Reward per {} episodes".format(window))
    plt.title("RL Lander(s)")
    plt.legend(chosen_agents, loc="lower right")
    plt.show()
    """
    for agent, agent_total_returns in zip(chosen_agents, agents_returns):
        reward_per_episode = agent_total_returns

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.plot(range(1, len(reward_per_episode) + 1), reward_per_episode, 'b', label='Episode Reward')
    ax.plot(range(1, len(reward_per_episode) - 99 + 1), moving_average(reward_per_episode, 100), 'r', label='Moving Average (100)')
    ax.set_title('Episode Reward over time', fontsize=16)
    ax.set_xlabel('Episode', fontsize=16)
    ax.set_ylabel('Reward', fontsize=16)
    ax.legend()
    plt.show()

def dqn_lander(env, n_episodes, gamma, lr, min_eps, \
                batch_size=32, memory_capacity=50000, \
                network='linear', learning_starts=1000, \
                train_freq=1, target_network_update_freq=1000, \
                print_freq=500, render_freq=500, save_freq=1000):

    # set device to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_function = torch.nn.MSELoss()

    # path to save checkpoints
    PATH = "./data"
    if not os.path.isdir(PATH):
        os.mkdir(PATH)

    num_actions = env.action_space.n
    input_shape = env.observation_space.shape[-1]
    qnet, qnet_optim = build_qnetwork(num_actions, lr, input_shape, network, device)
    qtarget_net, _ = build_qnetwork(num_actions, lr, input_shape, network, device)
    qtarget_net.load_state_dict(qnet.state_dict())
    qnet.train()
    qtarget_net.eval()
    replay_memory = ReplayMemory(memory_capacity)

    epsilon = 1.0 
    return_per_ep = [0.0] 
    saved_mean_reward = None
    t = 0

    for i in range(n_episodes):
        curr_state = lmn_input(env.reset())
        if (i + 1) % render_freq == 0:
            render = True
        else:
            render = False

        while True:
            if render:
                env.render()

            # choose action A using behaviour policy -> ε-greedy; use q-network
            action = epsilon_greedy(qnet, curr_state.to(device), epsilon, num_actions)
            # take action A, earn immediate reward R and land into next state S'
            next_state, reward, done, _ = env.step(action)
            #next_frame = get_frame(env)
            next_state = lmn_input(next_state)

            # store transition (S, A, R, S', Done) in replay memory
            replay_memory.store(curr_state, action, float(reward), next_state, float(done))

            # if replay memory currently stores > 'learning_starts' transitions,
            # sample a random mini-batch and update q_network's parameters
            if t > learning_starts and t % train_freq == 0:
                states, actions, rewards, next_states, dones = replay_memory.sample_minibatch(batch_size)
                #loss = 
                fit(qnet, \
                    qnet_optim, \
                    qtarget_net, \
                    loss_function, \
                    states, \
                    actions, \
                    rewards, \
                    next_states, \
                    dones, \
                    gamma, \
                    num_actions, 
                    device)

            # periodically update q-target network's parameters
            if t > learning_starts and t % target_network_update_freq == 0:
                update_target_network(qnet, qtarget_net)

            t += 1
            return_per_ep[-1] += reward

            if done:
                if (i + 1) % print_freq == 0:
                    print("\nEpisode: {}".format(i + 1))
                    print("Episode return : {}".format(return_per_ep[-1]))
                    print("Total time-steps: {}".format(t))

                if (i + 1) % 100 == 0:
                    mean_100ep_reward = round(np.mean(return_per_ep[-101:-1]), 1)
                    print("\nLast 100 episodes mean reward: {}".format(mean_100ep_reward))

                if t > learning_starts and (i + 1) % save_freq == 0:
                    if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                        print("\nSaving model due to mean reward increase: {} -> {}".format(saved_mean_reward, mean_100ep_reward))
                        save_model(qnet, i + 1, PATH)
                        saved_mean_reward = mean_100ep_reward

                return_per_ep.append(0.0)
                epsilon = decay_epsilon(epsilon, min_eps)

                break

            curr_state = next_state

    return return_per_ep

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, help='number of training episodes', default=10000, required=False)                            # default number of episodes is 10000
    parser.add_argument('--lr', type=float, help='step-size (or learning rate) used in sarsa, q-learning, dqn', default=1e-3, required=False)   # default step-size is 0.001
    parser.add_argument('--gamma', type=float, help='discount rate, should be 0 < gamma < 1', default=0.99, required=False)                     # default gamma is 0.99
    parser.add_argument('--final_eps', type=float, help='decay epsilon unti it reaches its \'final_eps\' value', default=1e-2, required=False)  # default final eploration epsilon is 0.01
    args = parser.parse_args()

    environment = gym.make("LunarLander-v2")
    print("\nTraining DQN lander with arguments num_episodes={}, learning rate={}, gamma={}, final_epsilon={} ..."\
                    .format(args.n_episodes, args.lr, args.gamma, args.final_eps))
    total_rewards = dqn_lander(environment, args.n_episodes, args.gamma, args.lr, args.final_eps)
    print("Done!")

    environment.close()

    # plot rewards per 'win' episodes for each agent
    win = 100
    plot_rewards(['dqn'], [total_rewards], args.n_episodes, win)


if __name__ == '__main__':
    main()