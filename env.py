import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data

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
        m = 0.1
        M = 1.0
        l = 0.5
        x, x_dot, theta, theta_dot = self.state
        F = 100 if  action == 1 else -100
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
    
class CartPoleBulletEnv(gym.Env):
    def __init__(self, render=False):
        super(CartPoleBulletEnv, self).__init__()
        self.render_mode = "human" if render else None
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(0.02, physicsClientId=self.client)
        self.action_space = gym.spaces.Discrete(2)
        low = np.array([-5, -np.inf, -30*np.pi/180, -np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, -low, dtype=np.float32)
        self.episode_reward = 0.0
        self.g = -9.8
        self.force_mag = 100
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
        info = {"episode": {"r": self.episode_reward, "l": self.episode_reward}}
        self.episode_reward = 0.0
        return observation, info
    
    def step(self, action):
        force = self.force_mag if action == 1 else -self.force_mag
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=force, physicsClientId=self.client)
        
        p.stepSimulation(physicsClientId=self.client)
        observation = self._get_observation()
        
        x, x_dot, theta, theta_dot = observation
        terminated = bool(x < -2.4 or x > 2.4 or theta < -12 * np.pi / 180 or theta > 12 * np.pi / 180)
        reward = 1.0 if not terminated else 0.0
        self.episode_reward += reward
        
        
        return observation, reward, terminated, False, {}

    def close(self):
        if self.client >= 0:
            p.disconnect(self.client)
            self.client = -1

    def render(self):
        pass
    
