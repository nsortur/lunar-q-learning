import random
import torch
from torch.optim import SGD
import gymnasium as gym
import numpy as np
from models import DeepQ
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchrl.data import ReplayBuffer
from torch.nn import MSELoss


env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)

# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'cpu'


def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, alpha=0.01, decay_rate=0.999, minibatch_size=8):

    model = DeepQ(in_dim=8, hidden_neurons=256, num_actions=4).to(device)
    optim = SGD(model.parameters(), lr=alpha)
    buffer = ReplayBuffer()
    loss_fn = MSELoss()
    
    def add_to_buffer(state, action, reward, next_state):
        buffer.add(
            torch.concatenate([
                torch.tensor(state),
                torch.tensor([action]),
                torch.tensor([reward]),
                torch.tensor(next_state)
        ]))

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
            
            obs_next, reward, done, truncated, info = env.step(action)
            # print(reward)
            episode_reward.append(reward)
            
            add_to_buffer(obs, action, reward, obs_next)
            
            samp = buffer.sample(minibatch_size)  # (8 x 18)
            
            obs_buffer = samp[:, :8]
            reward_buffer = samp[:, 9] # (8 x 1)
            obs_next_buffer = samp[:, -8:]
            action_buffer = samp[:, 8]
            
            # env.render()
            # time.sleep(0.1)
            
            q_val_next = model(torch.tensor(obs_next_buffer, device=device).float())
            q_target = (reward_buffer + (gamma * torch.max(q_val_next)))
            
            q_val_obs = model(obs_buffer.to(device).float())
            
            q_val_action = torch.gather(q_val_obs, 1, action_buffer.type(torch.int64).unsqueeze(1)) # now (8 x 1)
            
            loss = loss_fn(q_target.float(), q_val_action.float())
            loss.backward()
            optim.step()
            optim.zero_grad()

        epsilon *= decay_rate
        rewards.append(np.sum(episode_reward))

    print(epsilon)
    return rewards


decay_rate = 0.999

# default: Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=decay_rate)
rewards = Q_table = Q_learning(num_episodes=2000, gamma=0.9, epsilon=1, alpha=0.001, decay_rate=decay_rate)  # Run Q-learning
plt.plot(rewards)
plt.savefig("rewards.png")