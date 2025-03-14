import random
import torch
import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)


def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
	"""
	Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon is decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """

	num_actions = 6
	num_states = 375

	num_updates = np.zeros((num_states, num_actions))
	Q_table = {}
	for i in range(num_states):
		Q_table[i] = np.zeros(num_actions)


	for i in range(num_episodes):
		obs, reward, done, info = env.reset()

		while not done:
			state = hash(obs)

			if (obs['guard_in_cell']):
				legal_actions_mask = np.array([0, 0, 0, 0, 1, 1], dtype=np.int8)
			else:
				legal_actions_mask = np.array([1, 1, 1, 1, 0, 0], dtype=np.int8)
			
			if (random.random() < epsilon):
				#random legal action
				action = env.action_space.sample(mask=legal_actions_mask)
			else:
				#optimal action
				action = np.argmax(Q_table[state])

			obs, reward, done, info = env.step(action)

			#update q-table
			new_state = hash(obs)

			alpha = 1 / (1 + num_updates[state, action])
			num_updates[state, action] += 1

			q_val = Q_table[state][action]
			Q_table[state][action] = q_val + (alpha * (reward + (gamma * np.max(Q_table[new_state])) - q_val))


		epsilon *= decay_rate

	# print(len(Q_table))
	# print(epsilon)
	return Q_table 

decay_rate = 0.99999

# default: Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=decay_rate)
Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning

