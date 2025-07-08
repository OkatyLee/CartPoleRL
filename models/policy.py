import torch.nn as nn
import torch.nn.functional as F
import torch

class ActorCriticDiscrete(nn.Module):
    def __init__(self):
        super(ActorCriticDiscrete, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, 2)  
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.actor(x)
        state_value = self.critic(x)
        return action_logits, state_value


class ActorCriticContinuous(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_mean = self.actor(x)
        action_std = torch.exp(self.actor_log_std)
        state_value = self.critic(x)
        return (action_mean, action_std), state_value