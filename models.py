import torch
from torch import nn

# state -> q values for each action
class DeepQ(nn.Module):
    def __init__(self, in_dim=8, hidden_neurons=128, num_actions=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, num_actions),
        )
        
    def forward(self, state):
        x = self.network(state) # 8 -> hidden_neurons -> 4
        return x # [0.2, 0.4, ...]
