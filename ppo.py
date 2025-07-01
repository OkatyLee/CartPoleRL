
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.distributions import Categorical, Normal
import gymnasium as gym
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
        


class PPO:
    def __init__(self, env, buffer_size=8192, gamma=0.99, clip_epsilon=0.2, lr=3e-4, lam=0.95, n_epochs=10, mb_size=64):
        self.action_space_type = 'discrete' if isinstance(env.action_space, gym.spaces.Discrete) else 'continuous'
        self.act = self._act_discrete if self.action_space_type == 'discrete' else self._act_continuous
        self.predict = self._predict_discrete if self.action_space_type == 'discrete' else self._predict_continuous
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ENV
        self.env = env
        self.n_envs = env.num_envs if hasattr(env, 'num_envs') else 1
        
        # MODEL
        self.policy = ActorCriticDiscrete() if self.action_space_type == 'discrete' else ActorCriticContinuous()
        self.policy = self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer(buffer_size, n_envs=self.n_envs, obs_dim=4, action_type=np.int64 if self.action_space_type == 'discrete' else np.float32)
        self.policy.train()
        
        # HYPERPARAMS
        self.buffer_size = buffer_size
        self.gamma, self.lam = gamma, lam
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.mb_size = mb_size

    def _act_discrete(self, obs):
        with torch.no_grad():
            action_logits, value = self.policy(obs)
        dist = Categorical(logits=action_logits) 
        action = dist.sample()
        return action.numpy(), dist.log_prob(action).numpy(), value.squeeze(-1).numpy()
        
    def _act_continuous(self, obs):
        with torch.no_grad():
            (action_mean, action_std), value = self.policy(obs)
        dist = Normal(loc=action_mean, scale=action_std)
        action = dist.sample()
        return action.squeeze(-1).numpy(), dist.log_prob(action).squeeze(-1).numpy(), value.squeeze(-1).numpy()
        
        
        
    def update_policy(self, advantages, returns):
        b_obs, b_actions, b_log_probs, b_values, b_rewards, b_dones = self.buffer.get_buffer()
        b_obs = torch.from_numpy(b_obs).float().to(self.device).reshape(-1, 4)
        b_actions = torch.from_numpy(b_actions).float().to(self.device).reshape(-1) if self.action_space_type == 'continuous' else torch.from_numpy(b_actions).long().to(self.device).reshape(-1)
        b_log_probs = torch.from_numpy(b_log_probs).float().to(self.device).reshape(-1)
        b_values = torch.from_numpy(b_values).float().to(self.device).reshape(-1)
        b_rewards = torch.from_numpy(b_rewards).float().to(self.device).reshape(-1)
        b_dones = torch.from_numpy(b_dones).float().to(self.device).reshape(-1)
        b_advantages = advantages.reshape(-1).to(self.device)
        b_returns = returns.reshape(-1).to(self.device)

        b_inds = np.arange(len(b_obs))
        
        for _ in range(self.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), 64):
                end = start + 64
                mb_inds = b_inds[start:end]

                # Minibatches
                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_log_probs = b_log_probs[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)  
                mb_returns = b_returns[mb_inds]

                # Forward pass
                new_action_logits, new_values = self.policy(mb_obs)
                new_values = new_values.squeeze(-1)

                new_dist = Categorical(logits=new_action_logits) if self.action_space_type == 'discrete' else Normal(loc=new_action_logits[0], scale=new_action_logits[1])
                new_log_probs = new_dist.log_prob(mb_actions)
                entropy_loss = new_dist.entropy().mean()
                
                # PPO Loss
                ratio = (new_log_probs - mb_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, mb_returns)
                
                loss = policy_loss + 0.1 * value_loss - 0.01 * entropy_loss
                

                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                

    def compute_gae(self, next_obs, next_done):
        _, _, _, values, rewards, dones = self.buffer.get_buffer()
        values = torch.tensor(values, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        with torch.no_grad():
            next_value = self.policy(next_obs)[1].squeeze(-1)

        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
        returns = advantages + values
        return advantages, returns

    def learn(self, total_timesteps):
        obs, info = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        reward_history = []
        length_history = []
        num_updates = total_timesteps // (self.buffer_size) + 1
        for update in range(num_updates):
            for step in range(self.buffer_size//self.n_envs):         
                action, log_prob, value = self.act(obs)
                next_obs, reward, done, info = self.env.step(action)
                self.buffer.add(next_obs, action, log_prob, value, reward, done)
                obs = torch.tensor(next_obs, dtype=torch.float32)
                done = torch.tensor(done, dtype=torch.float32)

                for i in range(self.n_envs):
                    if 'episode' in info[i]:
                        reward_history.append(info[i]['episode']['r'])
                        reward_history = reward_history[-100:]  
                        length_history.append(info[i]['episode']['l'])
                        length_history = length_history[-100:]  
                        print(f"Update {update}/{num_updates}, Timestep {step*self.n_envs}: Episode Reward: {info[i]['episode']['r']:.2f}, Mean Reward (last 100): {np.mean(reward_history[-100:]):.2f}, Mean Length (last 100): {np.mean(length_history[-100:]):.2f}")
            
            advantages, returns = self.compute_gae(obs, done)
            print("Advantage mean:", advantages.mean().item(), "std:", advantages.std().item())
            print("Return mean:", returns.mean().item(), "std:", returns.std().item())
            self.update_policy(advantages, returns)
        print(f"Training completed. Total timesteps: {total_timesteps}, Mean Reward (last 100): {np.mean(reward_history[-100:]):.2f}, Mean Length (last 100): {np.mean(length_history[-100:]):.2f}")
        self.policy.eval()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()

    def _predict_discrete(self, obs):
        if not isinstance(obs, np.ndarray) and not isinstance(obs, torch.Tensor):
            obs = np.array(obs, dtype=np.float32)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action_logits, value = self.policy(obs_tensor)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        return action, value

    def _predict_continuous(self, obs):
        if not isinstance(obs, np.ndarray) and not isinstance(obs, torch.Tensor):
            obs = np.array(obs, dtype=np.float32)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        (action_mean, action_std), value = self.policy(obs_tensor)
        dist = Normal(loc=action_mean, scale=action_std)
        action = dist.sample()
        return action, value