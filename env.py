import gymnasium as gym
import numpy as np

class CartPoleEnv(gym.Env):
    
    def __init__(self, render=False):
        super(CartPoleEnv, self).__init__()
        self.render_mode = "human" if render else None
        self.state = None
        low = np.array([-5, -np.inf, -30*np.pi/180, -np.inf], dtype=np.float32)
        high = -low
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.episode_reward = 0.0

    def reset(self):
        super().reset()
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        episode_reward = self.episode_reward
        self.episode_reward = 0.0
        return self.state, {"episode": {"r": episode_reward, "l": episode_reward}}

    def step(self, action):
        dt = 0.02
        g = 9.8
        m = 0.5
        M = 1.0
        l = 0.5
        x, x_dot, theta, theta_dot = self.state
        F = 10 if  action == 1 else -10
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        temp = (F + m*l*sin_theta*np.square(theta_dot)) / (M + m)
        theta_ddot = (g * sin_theta - cos_theta * temp) / (l * (4.0 / 3.0 - m * np.square(cos_theta) / (M + m)))
        x_ddot = temp - l * theta_ddot * cos_theta / (M + m)
        x = x + x_dot * dt
        x_dot = x_dot + x_ddot * dt
        theta = theta + theta_dot * dt
        theta_dot = theta_dot + theta_ddot * dt
        terminated = bool(x < -2.4 or x > 2.4 or theta < -12 * np.pi / 180 or theta > 12 * np.pi / 180)
        reward = 1.0 if not terminated else 0.0
        self.episode_reward += reward
        if self.render_mode == "human":
            self.render()
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        
        return self.state, reward, terminated, False, {}

    def close(self):
        return super().close()

    def render(self):
        pass