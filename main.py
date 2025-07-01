import gymnasium as gym
import os
from ppo import PPO
from env import CartPoleEnv, CartPoleBulletEnv
from vec_env import make_vec_env
import time

if __name__ == "__main__":
    ACTION_SPACE_TYPE = 'discrete'  # 'discrete' or 'continuous'
    FORCE_MAG = 10  # Force magnitude for the environment
    vec_env = make_vec_env(
        CartPoleEnv,
        n_envs=4, 
        render=False,
        action_space_type=ACTION_SPACE_TYPE, 
        force_mag=FORCE_MAG
    )
    model = PPO(vec_env, buffer_size=2**12)
    if os.path.exists('ppo_cartpole'):
        print('Found existing weights. Start loading')
        model.load("ppo_cartpole")
    else:
        print('Creating new model') 
        model.learn(total_timesteps=1_000_000)
        model.save("ppo_cartpole")
    vec_env.close()
    vec_env = make_vec_env(
        CartPoleBulletEnv,
        n_envs=4,
        render=False,
        action_space_type=ACTION_SPACE_TYPE,
        force_mag=FORCE_MAG
    )
    model = PPO(vec_env, buffer_size=2**10)
    model.load('ppo_cartpole')
    model.learn(total_timesteps=100_000)
    model.save('ppo_cartpole_bullet')
    vec_env.close()
    vec_env = make_vec_env(
        CartPoleBulletEnv,
        n_envs=1,
        render=True,
        action_space_type=ACTION_SPACE_TYPE, 
        force_mag=FORCE_MAG
    )
    model = PPO(vec_env)
    model.load('ppo_cartpole_bullet')
    obs, info = vec_env.reset()
    while True:
        
        actions, _states = model.predict(obs)
        actions = actions.cpu().numpy() 
        actions = actions.astype(int)
        obs, rewards, dones, info = vec_env.step(actions)

        vec_env.render()
        
        