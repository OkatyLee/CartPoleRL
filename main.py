import gymnasium as gym
import os
from ppo import PPO
from env import CartPoleEnv, CartPoleBulletEnv
from vec_env import make_vec_env
import time

if __name__ == "__main__":
    vec_env = make_vec_env(CartPoleEnv, n_envs=4)
    model = PPO(4, 2, vec_env, buffer_size=2**14)
    if os.path.exists('ppo_cartpole'):
        print('Find exsting weights. Start loading')
        model.load("ppo_cartpole")
    else:
        print('Creating new model') 
        model.learn(total_timesteps=1_000_000)
        model.save("ppo_cartpole")
    vec_env.close()
    vec_env = make_vec_env(CartPoleBulletEnv, n_envs=4, render=False)
    model = PPO(4, 2, vec_env, buffer_size=4096)
    model.load('ppo_cartpole')
    model.learn(total_timesteps=10_000)
    model.save('ppo_cartpole_bullet')
    vec_env.close()
    print('Training finished. Start testing')
    vec_env = make_vec_env(CartPoleBulletEnv, n_envs=1, render=True)
    model = PPO(4, 2, vec_env, buffer_size=4096)
    model.load('ppo_cartpole_bullet')
    obs, info = vec_env.reset()
    while True:
        
        actions, _states = model.predict(obs)
        actions = actions.cpu().numpy() 
        actions = actions.astype(int)
        obs, rewards, done, info = vec_env.step(actions)

        vec_env.render()
        
        