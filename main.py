import gymnasium as gym
import os
from ppo import PPO
from stable_baselines3.common.env_util import make_vec_env


vec_env = make_vec_env("CartPole-v1", n_envs=4)

#if os.path.exists('ppo_cartpole.zip'):
#    print('Find exsting weights. Start loading')
#    model = PPO.load("ppo_cartpole")
#else:
#    print('Creating new model') 
#    model = PPO("MlpPolicy", vec_env, verbose=1)
#    model.learn(total_timesteps=25000)
#    model.save("ppo_cartpole")

model = PPO(4, 2, vec_env, buffer_size=2048)
model.learn(total_timesteps=25000)

obs = vec_env.reset()
while True:
    actions, _states = model.predict(obs)
    actions = actions.cpu().numpy() 
    actions = actions.astype(int)
    obs, rewards, dones, info = vec_env.step(actions)

    vec_env.render("human")