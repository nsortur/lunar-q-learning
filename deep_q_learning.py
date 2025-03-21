import random
import torch
from torch.optim import SGD
import gymnasium as gym
import numpy as np
from models import DeepQ
import matplotlib.pyplot as plt
from tqdm import tqdm

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)

# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'cpu'

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, alpha=0.01, decay_rate=0.999):

    model = DeepQ(in_dim=8, num_actions=4).to(device)
    optim = SGD(model.parameters(), lr=alpha)

    done = False
    rewards = []
    for _ in tqdm(range(num_episodes)):
        done = False
        obs, info = env.reset()
        episode_reward = []
        while not done:
            
            q_vals = model(torch.tensor(obs, device=device))

            if random.random() < epsilon:
                # random legal action
                action = env.action_space.sample()
            else:
                # optimal action
                action = torch.argmax(q_vals).cpu().item()
            
            obs, reward, done, truncated, info = env.step(action)
            # print(reward)
            episode_reward.append(reward)
            
            # update q-table
            q_val = q_vals[action]
            
            # action = env.action_space.sample()
            # env.render()
            # time.sleep(0.1)
            
            q_val_next = model(torch.tensor(obs))
            q_target = (reward + (gamma * torch.max(q_val_next)) - q_val)
            
            loss = q_target - q_val
            loss.backward()
            optim.step()
            optim.zero_grad()

        epsilon *= decay_rate
        rewards.append(np.sum(episode_reward))

    print(epsilon)
    return rewards


decay_rate = 0.99

# default: Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=decay_rate)
rewards = Q_table = Q_learning(num_episodes=2000, gamma=0.9, epsilon=1, alpha=0.01, decay_rate=decay_rate)  # Run Q-learning
plt.plot(rewards)
plt.show()

