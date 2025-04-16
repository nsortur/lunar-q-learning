import torch
from models import DeepQ
import gymnasium as gym
import imageio
import pickle
import matplotlib.pyplot as plt
from deep_q_learning import build_action_mapping


device = 'cpu'

def save_new_history(env, num_episodes=20, history_file="evaluation_history.pkl", continuous=False, action_mapping=None):
    history = {"rewards": [], "episode_lengths": []}
    for ep in range(num_episodes):
        state, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        episode_length = 0
        while not (done or truncated):
            state_tensor = torch.tensor(state, device=device).float()
            with torch.no_grad():
                action_index = torch.argmax(model(state_tensor)).cpu().item()
                if continuous:
                    action = action_mapping[action_index]
                else:
                    action = action_index
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            episode_length += 1
        history["rewards"].append(total_reward)
        history["episode_lengths"].append(episode_length)
        print(f"Episode {ep+1}: Reward = {total_reward}, Length = {episode_length}")
    with open(history_file, "wb") as f:
        pickle.dump(history, f)


def print_statistics(history_path="evaluation_history.pkl"):
    with open(history_path, "rb") as f:
        data = pickle.load(f)
    rewards = data["rewards"]
    lengths = data["episode_lengths"]
    best_reward = max(rewards)
    avg_reward = sum(rewards) / len(rewards)
    avg_length = sum(lengths) / len(lengths)
    std_reward = (sum((r - avg_reward)**2 for r in rewards) / len(rewards))**0.5
    std_length = (sum((l - avg_length)**2 for l in lengths) / len(lengths))**0.5
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.2f} ± {std_length:.2f}")

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
        

if __name__ == "__main__":
    
    # uncomment block if using continuous
    # continuous = True
    # fidelity = 10
    # action_mapping = build_action_mapping(fidelity)
    # num_actions = len(action_mapping)
    
    
    # comment block if using continuous
    continuous=False
    action_mapping=None
    num_actions = 4
    
    model = DeepQ(in_dim=8, hidden_neurons=256, num_actions=num_actions).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()
    
    env = gym.make("LunarLander-v3", continuous=continuous, gravity=-10.0,
        enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode='rgb_array')
    
    # visualize_policy(model, episode=1)
    save_new_history(env, num_episodes=20, continuous=continuous, action_mapping=action_mapping)
    print_statistics()

