# From: https://github.com/lazavgeridis/LunarLander-v2, dqn version

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import sys
import numpy as np
import random

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

def main():
    _, episodes, model_path = sys.argv
    env = gym.make('LunarLander-v2')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    qnet = LinearMapNet(8, 4).to(device)
    qnet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    qnet.eval()

    for episode in range(int(episodes)):
        episode_reward = 0
        curr_state, done = env.reset(), False
        curr_state = np.expand_dims(curr_state, 0)
        curr_state = torch.from_numpy(curr_state)

        while not done:
            env.render()
            action = epsilon_greedy(qnet, curr_state.to(device), 0.0001, 4)
            next_state, reward, done, _ = env.step(action)
            next_state = np.expand_dims(next_state, 0)
            next_state = torch.from_numpy(next_state)
            episode_reward += reward
            curr_state = next_state

        print(f"Episode reward: {episode_reward}")


if __name__ == '__main__':
    main()