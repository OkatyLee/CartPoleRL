import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data

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
        reward = 1.0 + shaping_term if not terminated else shaping_term
        #reward = 1 if not terminated else 0
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
    
class CartPoleBulletEnv(gym.Env):
    def __init__(self, force_mag=100,  render=False, action_space_type='discrete'):
        self.action_space_type = action_space_type
        if action_space_type not in ['discrete', 'continuous']:
            raise ValueError("action_space_type must be either 'discrete' or 'continuous'")
        super(CartPoleBulletEnv, self).__init__()
        self.render_mode = "human" if render else None
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(0.02, physicsClientId=self.client)
        self.action_space = gym.spaces.Discrete(2) if action_space_type == 'discrete' else gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        low = np.array([-5, -np.inf, -30*np.pi/180, -np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, -low, dtype=np.float32)
        self.episode_reward = 0.0
        self.episode_length = 0
        self.g = -9.8
        self.force_mag = force_mag
        self.time_step = 0.02
        self.cartpole = None
    
    def _get_observation(self):
        if self.cartpole is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        state = p.getJointState(self.cartpole, 0, physicsClientId=self.client)
        x = state[0]
        x_dot = state[1]
        state = p.getJointState(self.cartpole, 1, physicsClientId=self.client)
        theta = state[0]
        theta_dot = state[1]
        return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset()
        

        p.resetSimulation(physicsClientId=self.client)
        
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        
        self.cartpole = p.loadURDF("cartpole.urdf", [0,0,0.3], physicsClientId=self.client)
        
        p.resetJointState(self.cartpole, 0, np.random.uniform(low=-0.05, high=0.05), physicsClientId=self.client)
        p.resetJointState(self.cartpole, 1, np.random.uniform(low=-0.05, high=0.05), physicsClientId=self.client)
        p.setJointMotorControl2(
            bodyUniqueId=self.cartpole,
            jointIndex=1,  
            controlMode=p.VELOCITY_CONTROL,
            force=1e-6,
            physicsClientId=self.client
        )
        observation = self._get_observation()
        info = {"episode": {"r": self.episode_reward, "l": self.episode_length}}
        self.episode_reward = 0.0
        self.episode_length = 0
        return observation, info
    
    def step(self, action):
        if self.action_space_type == 'discrete':
            force = self.force_mag if action == 1 else -self.force_mag
        else:
            force = action * self.force_mag
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=force, physicsClientId=self.client)
        
        p.stepSimulation(physicsClientId=self.client)
        observation = self._get_observation()
        
        x, x_dot, theta, theta_dot = observation
        terminated = bool(x < -2.4 or x > 2.4 or theta < -12 * np.pi / 180 or theta > 12 * np.pi / 180)
        reward = 1.0 - theta**2 - 0.1 * x**2 - 0.1 * action**2 if not terminated else -10.0

        self.episode_reward += reward
        self.episode_length += 1
        
        return observation, reward, terminated, False, {}

    def close(self):
        if self.client >= 0:
            p.disconnect(self.client)
            self.client = -1

    def render(self):
        pass
    
