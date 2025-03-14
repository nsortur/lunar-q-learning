import random
import torch
import gymnasium as gym
import numpy as np
from collections import defaultdict

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="human")


def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, alpha=0.9, decay_rate=0.999):

    Q_table = defaultdict()

    done = False
    for _ in range(num_episodes):
        obs, info = env.reset()

        while not done:
            state = hash(str(obs))

            if random.random() < epsilon:
                # random legal action
                action = env.action_space.sample()
            else:
                # optimal action
                action = np.argmax(Q_table[state])
            

            obs, reward, done, truncated, info = env.step(action)

            # update q-table
            new_state = hash(str(obs))
            q_val = Q_table[state][action]
            
            # action = env.action_space.sample()
            # env.render()
            # time.sleep(0.1)
            
            # replace with q learning update rule
            # https://huggingface.co/learn/deep-rl-course/en/unit3/deep-q-algorithm
            Q_table[state][action] = q_val + (alpha * (reward + (gamma * np.max(Q_table[new_state])) - q_val))

        epsilon *= decay_rate

    return Q_table


decay_rate = 0.99999

# default: Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=decay_rate)
Q_table = Q_learning(num_episodes=10, gamma=0.9, epsilon=1, alpha=0.9, decay_rate=decay_rate)  # Run Q-learning
