import torch.nn as nn
import torch.nn.functional as F
import torch

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, ):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, output_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value
    

class RolloutBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0

    def add(self, transition):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.index] = transition
            self.index = (self.index + 1) % self.buffer_size

    def sample(self, batch_size):
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        return [self.buffer[i] for i in indices]
    
    def clear(self):
        self.buffer = []
        self.index = 0
        
