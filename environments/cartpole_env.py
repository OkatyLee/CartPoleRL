import gymnasium as gym
import numpy as np

class CartPoleEnv(gym.Env):
    
    def __init__(self, force_mag=100, render=False, action_space_type='discrete'):
        self.action_space_type = action_space_type
        if action_space_type not in ['discrete', 'continuous']:
            raise ValueError("action_space_type must be either 'discrete' or 'continuous'")
        super(CartPoleEnv, self).__init__()
        self.render_mode = "human" if render else None
        self.state = None
        self.force_mag = force_mag
        
        low = np.array([-5, -np.inf, -30*np.pi/180, -np.inf], dtype=np.float32)
        high = -low
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2) if action_space_type=='discrete' else gym.spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        self.episode_reward = 0.0
        self.episode_length = 0

    def reset(self):
        super().reset()
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        episode_reward = self.episode_reward
        episode_length = self.episode_length
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.state, {"episode": {"r": episode_reward, "l": episode_length}}

    def step(self, action):
        dt = 0.02
        g = 9.8
        m = 0.1
        M = 1.0
        l = 0.5
        x, x_dot, theta, theta_dot = self.state
        if self.action_space_type == 'discrete':
            F = self.force_mag if action else -self.force_mag
        else:
            F = np.clip(action * self.force_mag, -self.force_mag, self.force_mag)
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
        gamma, alpha, beta, lam = 0.99, 0.05, 0.1, 0.01
        shaping_term = gamma * (-alpha * x_dot**2 - beta * theta_dot**2 - lam * action**2) + alpha * x**2 + beta * theta**2
        #reward = 1.0 + shaping_term if not terminated else shaping_term
        reward = 1 if not terminated else 0
        self.episode_reward += reward
        self.episode_length += 1
        if self.render_mode == "human":
            self.render()
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        
        return self.state, reward, terminated, False, {}

    def close(self):
        return super().close()

    def render(self):
        pass