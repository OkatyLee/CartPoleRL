import numpy as np

class RolloutBuffer:
    def __init__(self, buffer_size, n_envs=1, obs_dim=4, action_type=np.float32):
        self.buffer_size = buffer_size
        self.obs = np.zeros((buffer_size, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_envs), dtype=action_type)  
        self.log_probs = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=bool)
        self.index = 0

    def add(self, obs, action, log_prob, value, reward, done):
        self.obs[self.index] = obs
        self.actions[self.index] = action
        self.log_probs[self.index] = log_prob
        self.values[self.index] = value
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.index = (self.index + 1) % self.buffer_size

    def sample(self, idx):
        return self.obs[idx], self.actions[idx], self.log_probs[idx], self.values[idx], self.rewards[idx], self.dones[idx]
    
    def get_buffer(self):
        return self.obs, self.actions, self.log_probs, self.values, self.rewards, self.dones
    
    def clear(self):
        self.index = 0