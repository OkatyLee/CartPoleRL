
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.distributions import Categorical
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
        action_logits = self.actor(x)
        state_value = self.critic(x)
        return action_logits, state_value
    

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

    def sample(self, idx):
        return [self.buffer[i] for i in idx]
    
    def clear(self):
        self.buffer = []
        self.index = 0
        


class PPO:
    def __init__(self, input_dim, output_dim, env, buffer_size=8192, gamma=0.99, clip_epsilon=0.2, lr=3e-4, lam=0.95, n_epochs=10, mb_size=64):
        # MODEL
        self.policy = ActorCritic(input_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer(buffer_size)
        # ENVS
        self.env = env
        self.n_envs = env.num_envs if hasattr(env, 'num_envs') else 1
        # HYPERPARAMS
        self.buffer_size = buffer_size
        self.gamma, self.lam = gamma, lam
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.mb_size = mb_size

    def act(self, obs):
        with torch.no_grad():
            action_logits, value = self.policy(obs)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        return action.numpy(), dist.log_prob(action).numpy(), value.squeeze(-1).numpy()
        
        
        
    def update_policy(self, advantages, returns):
        b_obs, b_actions, b_log_probs, b_values, b_rewards, b_dones = zip(*self.buffer.buffer)
        b_obs = np.array(b_obs, dtype=np.float32).reshape(-1, 4)
        b_actions = np.array(b_actions, dtype=np.int64).reshape(-1)
        b_log_probs = np.array(b_log_probs, dtype=np.float32).reshape(-1)
        b_values = np.array(b_values, dtype=np.float32).reshape(-1)
        b_rewards = np.array(b_rewards, dtype=np.float32).reshape(-1)
        b_dones = np.array(b_dones, dtype=np.float32).reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        b_inds = np.arange(len(b_obs))
        
        for _ in range(self.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), 64):
                end = start + 64
                mb_inds = b_inds[start:end]

                # Minibatches
                mb_obs = torch.tensor(b_obs[mb_inds], dtype=torch.float32)
                mb_actions = torch.tensor(b_actions[mb_inds], dtype=torch.int64)
                mb_log_probs = torch.tensor(b_log_probs[mb_inds], dtype=torch.float32)
                mb_advantages = torch.tensor(b_advantages[mb_inds], dtype=torch.float32)
                mb_returns = torch.tensor(b_returns[mb_inds], dtype=torch.float32)

                # Forward pass
                new_action_logits, new_values = self.policy(mb_obs)
                new_values = new_values.squeeze(-1)
                new_dist = Categorical(logits=new_action_logits)
                new_log_probs = new_dist.log_prob(mb_actions)
                entropy_loss = new_dist.entropy().mean()
                
                # PPO Loss
                ratio = (new_log_probs - mb_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, mb_returns)
                
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                

    def compute_gae(self, next_obs, next_done):
        _, _, _, values, rewards, dones = zip(*self.buffer.buffer)
        values = np.array(values, dtype=np.float32)
        values = torch.tensor(values, dtype=torch.float32)
        rewards = np.array(rewards, dtype=np.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = np.array(dones, dtype=np.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        with torch.no_grad():
            next_value = self.policy(next_obs)[1].squeeze(-1)

        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(self.buffer.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
        returns = advantages + values
        return advantages, returns

    def learn(self, total_timesteps):
        obs, info = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        episode_rewards = np.zeros(self.n_envs)
        reward_history = []
        num_updates = total_timesteps // (self.buffer.buffer_size * self.n_envs) + 1
        for update in range(num_updates):
            for step in range(self.buffer.buffer_size):
                action, log_prob, value = self.act(obs)
                next_obs, reward, done, info = self.env.step(action)
                self.buffer.add((obs, action, log_prob, value, reward, done))

                obs = torch.tensor(next_obs, dtype=torch.float32)
                done = torch.tensor(done, dtype=torch.float32)

                for i in range(self.n_envs):
                    if 'episode' in info[i]:
                        episode_rewards[i] += info[i]['episode']['r']
                        reward_history.append(info[i]['episode']['r'])
                        print(f"Update {update}/{num_updates}, Timestep {step*self.n_envs}: Episode Reward: {info[i]['episode']['r']:.2f}, Mean Reward (last 100): {np.mean(reward_history[-100:]):.2f}")
            
            advantages, returns = self.compute_gae(obs, done)
            
            self.update_policy(advantages, returns)

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()

    def predict(self, obs):
        if not isinstance(obs, np.ndarray) and not isinstance(obs, torch.Tensor):
            obs = np.array(obs, dtype=np.float32)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action_logits, value = self.policy(obs_tensor)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        return action, value