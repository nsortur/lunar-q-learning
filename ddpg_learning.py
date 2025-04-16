import gymnasium as gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
# model.learn(total_timesteps=80000, log_interval=10)
# model.save("ddpg_pendulum2")
vec_env = model.get_env()

del model # remove to demonstrate saving and loading

env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)
model = DDPG.load("ddpg_pendulum", env)
vec_env = model.get_env()

num_episodes = 20
history = {"rewards": [], "episode_lengths": []}

for ep in range(num_episodes):

    obs = vec_env.reset()
    done = False
    truncated = False
    total_reward = 0
    episode_length = 0

    while not (done or truncated):
        action, _states = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)
        total_reward += float(reward)
        episode_length += 1
        env.render()
    
    history["rewards"].append(total_reward)
    history["episode_lengths"].append(episode_length)
    print(f"Episode {ep+1}: Reward = {total_reward}, Length = {episode_length}")

rewards = history["rewards"]
lengths = history["episode_lengths"]
best_reward = max(rewards)
avg_reward = sum(rewards) / len(rewards)
avg_length = sum(lengths) / len(lengths)
std_reward = (sum((r - avg_reward)**2 for r in rewards) / len(rewards))**0.5
std_length = (sum((l - avg_length)**2 for l in lengths) / len(lengths))**0.5

print(f"Best Reward: {best_reward:.2f}")
print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
print(f"Average Episode Length: {avg_length:.2f} ± {std_length:.2f}")