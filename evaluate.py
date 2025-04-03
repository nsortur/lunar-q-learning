import torch
from models import DeepQ
import gymnasium as gym
import imageio
import matplotlib.pyplot as plt

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode='rgb_array')
device = 'cpu'

def visualize_policy(model, episode, max_steps=200):
        frames = []
        state, info = env.reset()
        done = False
        truncated = False
        steps = 0
        rewards = []
        while (not (done or truncated)) or (steps < max_steps):
            frame = env.render()
            frames.append(frame)
            state_tensor = torch.tensor(state, device=device).float()
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).cpu().item()
            state, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            steps += 1
        imageio.mimwrite(f"policy_episode_{episode}.gif", frames, fps=30)
        plt.figure(figsize=(8, 4))
        plt.plot(rewards, marker='o', linestyle='-', color='b', label='Reward per Step')
        plt.title('Episode Reward Progress')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
model = DeepQ(in_dim=8, hidden_neurons=256, num_actions=4).to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

visualize_policy(model, episode=1)

