import gymnasium as gym
import os
from ppo import PPO
from stable_baselines3.common.env_util import make_vec_env


vec_env = make_vec_env("CartPole-v1", n_envs=4)
model = PPO(4, 2, vec_env, buffer_size=4096)
if os.path.exists('ppo_cartpole'):
    print('Find exsting weights. Start loading')
    model.load("ppo_cartpole")
else:
    print('Creating new model') 
    model.learn(total_timesteps=100000)
    model.save("ppo_cartpole")


obs = vec_env.reset()
while True:
    actions, _states = model.predict(obs)
    actions = actions.cpu().numpy() 
    actions = actions.astype(int)
    obs, rewards, dones, info = vec_env.step(actions)

    vec_env.render("human")