import torch
import numpy as np
from torch.distributions import Categorical, Normal
import torch.nn.functional as F
import gymnasium as gym
from utils.buffer import RolloutBuffer
from models import ActorCriticDiscrete, ActorCriticContinuous
from typing import Union, Tuple, Dict
from pathlib import Path
from utils.logger import setup_logger
import logging
import time

class PPO:
    def __init__(self, env: gym.Env, buffer_size: int=8192, gamma: float=0.99,
                 clip_epsilon: float=0.2, lr: float=3e-4, lam: float=0.95,
                 n_epochs: int=10, mb_size: int=64, log_dir: str="logs",
                 experiment_name: str="ppo_experiment", log_level: int=logging.INFO):
        
        self.experiment_name = experiment_name
        self.episode_count = 0
        self.global_step = 0
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Setup main logger
        self.logger = setup_logger(
            name=f"ppo_{experiment_name}",
            log_file=self.log_dir / f"{experiment_name}.log",
            level=log_level
        )
        # Setup metrics logger
        self.metrics_logger = setup_logger(
            name=f"ppo_metrics_{experiment_name}",
            log_file=self.log_dir / f"{experiment_name}_metrics.log",
            level=log_level,
            format_str='%(asctime)s - %(message)s'
        )
        
        self.logger.info("=" * 60)
        self.logger.info(f"Initializing PPO Agent: {experiment_name}")
        self.logger.info("=" * 60)
        
        
        # Define action space type
        self.action_space_type = 'discrete' if isinstance(env.action_space, gym.spaces.Discrete) else 'continuous'
        self.logger.info(f"Action space type: {self.action_space_type}")
        self.act = self._act_discrete if self.action_space_type == 'discrete' else self._act_continuous
        self.predict = self._predict_discrete if self.action_space_type == 'discrete' else self._predict_continuous
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device:: {self.device}")
        
        # ENV
        self.env = env
        self.n_envs = env.num_envs if hasattr(env, 'num_envs') else 1
        self.logger.info(f"Number of environments: {self.n_envs}")
        
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
        
        self.logger.info("PPO initialization completed successfully")
        

    def _act_discrete(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            action_logits, value = self.policy(obs)
        dist = Categorical(logits=action_logits) 
        action = dist.sample()
        return action.numpy(), dist.log_prob(action).numpy(), value.squeeze(-1).numpy()
        
    def _act_continuous(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            (action_mean, action_std), value = self.policy(obs)
        dist = Normal(loc=action_mean, scale=action_std)
        action = dist.sample()
        return action.squeeze(-1).numpy(), dist.log_prob(action).squeeze(-1).numpy(), value.squeeze(-1).numpy()
        


    def update_policy(self, advantages: np.ndarray, returns: np.ndarray) -> Dict:
        
        self.logger.debug("Starting policy update...")
        update_start_time = time.time()
        
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
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_clipfrac = 0
        n_minibatches = 0
        
        for epoch in range(self.n_epochs):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy_loss = 0
            epoch_minibatches = 0
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), self.mb_size):
                end = start + self.mb_size
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
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy_loss += entropy_loss.item()
                epoch_minibatches += 1
                n_minibatches += 1
            self.logger.debug(f"Epoch {epoch+1}/{self.n_epochs}: "
                            f"Policy Loss: {epoch_policy_loss/epoch_minibatches:.6f}, "
                            f"Value Loss: {epoch_value_loss/epoch_minibatches:.6f}, "
                            f"Entropy: {epoch_entropy_loss/epoch_minibatches:.6f}, "
            )
            
            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss
            total_entropy_loss += epoch_entropy_loss
        update_time = time.time() - update_start_time
        
        metrics = {
            'policy_loss': total_policy_loss / n_minibatches,
            'value_loss': total_value_loss / n_minibatches,
            'entropy': total_entropy_loss / n_minibatches,
            'update_time': update_time
        }
        
        self.logger.info(
            f"Policy update completed in {update_time:.2f}s. "
            f"Policy Loss: {metrics['policy_loss']:.6f}, "
            f"Value Loss: {metrics['value_loss']:.6f}, "
            f"Entropy: {metrics['entropy']:.6f}, "
        )
        
        return metrics

    def compute_gae(self, next_obs: torch.Tensor, next_done: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        self.logger.debug("Computing GAE...")
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
        
        self.logger.debug(f"GAE computed. Advantages - Mean: {advantages.mean():.4f}, Std: {advantages.std():.4f}")
        self.logger.debug(f"Returns - Mean: {returns.mean():.4f}, Std: {returns.std():.4f}")
        
        return advantages, returns

    def learn(self, total_timesteps: int) -> None:
        self.logger.info(f"Starting training for {total_timesteps:,} timesteps")
        self.logger.info(f"Updates planned: {total_timesteps // self.buffer_size}")
        obs, info = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        reward_history = []
        length_history = []
        training_start_time = time.time()
        num_updates = total_timesteps // (self.buffer_size) + 1
        for update in range(num_updates):
            update_start_time = time.time()
            self.logger.info(f"Update {update+1}/{num_updates} started")
            collection_start_time = time.time()
            for step in range(self.buffer_size//self.n_envs):         
                action, log_prob, value = self.act(obs)
                next_obs, reward, done, info = self.env.step(action)
                self.buffer.add(next_obs, action, log_prob, value, reward, done)
                obs = torch.tensor(next_obs, dtype=torch.float32)
                done = torch.tensor(done, dtype=torch.float32)

                for i in range(self.n_envs):
                    if i < len(info) and 'episode' in info[i]:
                        episode_reward = info[i]['episode']['r']
                        episode_length = info[i]['episode']['l']
                        
                        reward_history.append(episode_reward)
                        length_history.append(episode_length)
                        self.episode_count += 1                        
                        reward_history = reward_history[-100:]
                        length_history = length_history[-100:]
                        
                        self.logger.info(f"Episode {self.episode_count}: "
                                       f"Reward: {episode_reward:.2f}, "
                                       f"Length: {episode_length}")
                        
                        self.metrics_logger.info(f"EPISODE,{self.episode_count},{episode_reward:.2f},{episode_length},{self.global_step}")
                        
                        # Средние метрики за последние 100 эпизодов
                        if len(reward_history) >= 10:
                            mean_reward = np.mean(reward_history[-100:])
                            mean_length = np.mean(length_history[-100:])
                            self.logger.info(f"Mean reward (last 100): {mean_reward:.2f}, "
                                           f"Mean length (last 100): {mean_length:.2f}")
            collection_time = time.time() - collection_start_time
            advantages, returns = self.compute_gae(obs, done)
            
            training_metrics = self.update_policy(advantages, returns)
            update_time = time.time() - update_start_time
            elapsed_time = time.time() - training_start_time
            
            # Логируем метрики обновления
            self.logger.info(f"Update {update+1} completed in {update_time:.2f}s "
                           f"(collection: {collection_time:.2f}s, training: {training_metrics['update_time']:.2f}s)")
            
            # Прогресс обучения
            progress = (update + 1) / num_updates * 100
            fps = self.global_step / elapsed_time
            
            self.logger.info(f"Progress: {progress:.1f}% | "
                           f"Steps: {self.global_step:,}/{total_timesteps:,} | "
                           f"FPS: {fps:.0f} | "
                           f"Elapsed: {elapsed_time/60:.1f}m")
            
            # Метрики обучения в отдельный файл
            self.metrics_logger.info(f"UPDATE,{update+1},{training_metrics['policy_loss']:.6f},"
                                   f"{training_metrics['value_loss']:.6f},{training_metrics['entropy']:.6f},"
                                   f"{self.global_step}")
            
            if (update + 1) % 10 == 0:
                checkpoint_path = f"./saved_models/checkpoints/{self.experiment_name}/checkpoint_{update+1}.pth"
                self.save(checkpoint_path)
                self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        
        training_time = time.time() - training_start_time
        final_mean_reward = np.mean(reward_history[-100:]) if reward_history else 0
        final_mean_length = np.mean(length_history[-100:]) if length_history else 0
        
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total training time: {training_time/60:.1f} minutes")
        self.logger.info(f"Total episodes: {self.episode_count}")
        self.logger.info(f"Total timesteps: {self.global_step:,}")
        self.logger.info(f"Final mean reward (last 100): {final_mean_reward:.2f}")
        self.logger.info(f"Final mean length (last 100): {final_mean_length:.2f}")
        self.logger.info(f"Average FPS: {self.global_step/training_time:.0f}")

        final_model_path = f"./saved_models/{self.experiment_name}.pth"
        self.save(str(final_model_path))
        self.logger.info(f"Final model saved: {final_model_path}")
        self.policy.eval()

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()

    def _predict_discrete(self, obs: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(obs, np.ndarray) and not isinstance(obs, torch.Tensor):
            obs = np.array(obs, dtype=np.float32)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action_logits, value = self.policy(obs_tensor)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        return action, value

    def _predict_continuous(self, obs: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(obs, np.ndarray) and not isinstance(obs, torch.Tensor):
            obs = np.array(obs, dtype=np.float32)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        (action_mean, action_std), value = self.policy(obs_tensor)
        dist = Normal(loc=action_mean, scale=action_std)
        action = dist.sample()
        return action, value