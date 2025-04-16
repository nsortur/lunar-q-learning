import random
import torch
from torch.optim import Adam
import gymnasium as gym
import numpy as np
from models import DeepQ
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchrl.data import ReplayBuffer
from torch.nn import MSELoss
import pickle
import argparse
from torchrl.data.replay_buffers import ListStorage

device = 'cpu'

def build_action_mapping(fidelity):
    grid_vals = np.linspace(-1, 1, fidelity)
    actions = []
    for main in grid_vals:
        for lateral in grid_vals:
            actions.append(np.array([main, lateral], dtype=np.float32))
    return actions


def Q_learning(env, num_episodes=10000, gamma=0.9, epsilon=1, alpha=0.01, decay_rate=0.999, minibatch_size=8, target_update_freq=50, buffer_storage=None, num_actions=4):
    

    model = DeepQ(in_dim=8, hidden_neurons=256, num_actions=num_actions).to(device)
    target_model = DeepQ(in_dim=8, hidden_neurons=256, num_actions=num_actions).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optim = Adam(model.parameters(), lr=alpha)
    buffer = ReplayBuffer(storage=ListStorage(max_size=buffer_storage))
    loss_fn = MSELoss()
    
    def add_to_buffer(state, action, reward, next_state, done):
        done_val = 1.0 if done else 0.0
        buffer.add(
            torch.concatenate([
                torch.tensor(state),
                torch.tensor([action]),
                torch.tensor([reward]),
                torch.tensor([done_val]),
                torch.tensor(next_state)
        ]))
    
    rewards = []
    lengths = []
    best_reward = -np.inf 
    step_counter = 0
    pbar = tqdm(range(num_episodes))
    for episode in pbar:
        done = False
        truncated = False
        obs, info = env.reset()
        episode_reward = []
        while not (done or truncated):
            
            q_vals = model(torch.tensor(obs, device=device).float())

            if random.random() < epsilon:
                if args.continuous:
                    # randomly choose an index over the discretized action list.
                    random_index = random.randint(0, num_actions - 1)
                    action = action_mapping[random_index]
                    action_index = random_index
                else:
                    action = env.action_space.sample()
                    action_index = action
            else:
                index = torch.argmax(q_vals).cpu().item()
                if args.continuous:
                    action = action_mapping[index]
                    action_index = index
                else:
                    action = index
                    action_index = index
            
            obs_next, reward, done, truncated, info = env.step(action)
            episode_reward.append(reward)
            
            add_to_buffer(obs, action_index, reward, obs_next, done)
            obs = obs_next
            
            samp = buffer.sample(minibatch_size)  # (8 x 18)
            
            obs_buffer = samp[:, :8].to(device)
            reward_buffer = samp[:, 9].to(device) # (8 x 1)
            done_buffer = samp[:, 10].to(device)
            obs_next_buffer = samp[:, 11:].to(device)
            action_buffer = samp[:, 8].to(device)
            
            # q target network
            q_val_next = target_model(obs_next_buffer.to(device).float())
            # q_val_next = model(obs_next_buffer.to(device).float())
            max_q_val_next, _ = torch.max(q_val_next, dim=1)
            q_target = (reward_buffer + (gamma * max_q_val_next * (1 - done_buffer.float())))
            q_target = q_target.unsqueeze(1)
            
            q_val_obs = model(obs_buffer.to(device).float())
            q_val_action = torch.gather(q_val_obs, 1, action_buffer.type(torch.int64).unsqueeze(1)) # now (8 x 1)
            
            loss = loss_fn(q_target.float(), q_val_action.float())
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            step_counter += 1
            # update target network
            if step_counter % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
            

        lengths.append(len(episode_reward))
        epsilon = max(0.1, epsilon*decay_rate)
        rewards.append(np.sum(episode_reward))
        pbar.set_description(f"Reward: {rewards[-1]:.3f}, epsilon: {epsilon:.3f}")
        
        # Save best model
        if rewards[-1] > best_reward:
            best_reward = rewards[-1]
            torch.save(model.state_dict(), "best_model.pt")

    # Save rewards and episode lengths for later plotting
    pickle.dump({"rewards": rewards, "episode_lengths": lengths}, open("training_history.pkl", "wb"))
    torch.save(model.state_dict(), "last_model.pt")
    
    print(epsilon)
    return rewards, lengths


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="Train either plain DQN or DQN with experience replay")
        parser.add_argument('--algorithm', type=str, choices=['dqn', 'dqn_replay'], default='dqn_replay',
                            help='Select algorithm type: "dqn" for plain DQN (no replay) or "dqn_replay" for experience replay')
        parser.add_argument('--episodes', type=int, default=1200, help='Number of training episodes')
        parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
        parser.add_argument('--epsilon', type=float, default=1.0, help='Initial epsilon for exploration')
        parser.add_argument('--alpha', type=float, default=0.0001, help='Learning rate')
        parser.add_argument('--decay_rate', type=float, default=0.995, help='Epsilon decay rate')
        parser.add_argument('--minibatch_size', type=int, default=64, help='Minibatch size (only used for dqn_replay)')
        parser.add_argument('--target_update_freq', type=int, default=50, help='Frequency of target network updates')
        parser.add_argument('--buffer_storage', type=int, default=10000,
                    help='Number of experiences to store in the replay buffer')
        parser.add_argument('--continuous', action='store_true',
                            help='If set, use continuous action space (discretized) instead of discrete')
        parser.add_argument('--fidelity', type=int, default=10,
                            help='Number of discrete slices per dimension when using continuous actions')
        return parser.parse_args()

    args = parse_args()

    # for plain DQN, use minibatch size 1 to update per step (essentially no replay)
    minibatch_size = 1 if args.algorithm == 'dqn' else args.minibatch_size
    buffer_storage = 1 if args.algorithm == 'dqn' else args.buffer_storage
    
    env = gym.make("LunarLander-v3", continuous=args.continuous, gravity=-10.0,
        enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    
    if args.continuous:
        action_mapping = build_action_mapping(args.fidelity)
        num_actions = len(action_mapping)
    else:
        num_actions = env.action_space.n

    rewards, lengths = Q_learning(
        env,
        num_episodes=args.episodes,
        gamma=args.gamma,
        epsilon=args.epsilon,
        alpha=args.alpha,
        decay_rate=args.decay_rate,
        minibatch_size=minibatch_size,
        target_update_freq=args.target_update_freq,
        buffer_storage=buffer_storage,
        num_actions=num_actions
    )

    with open("training_history.pkl", "rb") as f:
        data = pickle.load(f)
    rewards = data["rewards"]
    lengths = data["episode_lengths"]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(rewards, label="Reward per Episode", color='blue', lw=2)
    axs[0].set_title("Training Rewards Over Episodes")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")
    axs[0].legend(loc="best")
    axs[0].grid(True)

    axs[1].plot(lengths, label="Episode Length", color='green', lw=2)
    axs[1].set_title("Episode Lengths Over Training")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Episode Length")
    axs[1].legend(loc="best")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("training_plots.png", dpi=300)

    print(f"Best Reward: {max(rewards):.2f}")

    avg_reward = sum(rewards) / len(rewards)
    avg_length = sum(lengths) / len(lengths)
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.2f}")
