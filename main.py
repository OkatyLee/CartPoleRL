import gymnasium as gym
import os
from models import PPO
from environments import CartPoleEnv, CartPoleBulletEnv
from utils import make_vec_env
import multiprocessing

if __name__ == "__main__":
    
    ACTION_SPACE_TYPE = 'continuous'  # 'discrete' or 'continuous'
    FORCE_MAG = 10  # Force magnitude for the environment
    multiprocessing.set_start_method('spawn', force=True)  # Use 'spawn' to avoid issues with PyBullet
    vec_env = make_vec_env(
        CartPoleEnv,
        n_envs=4, 
        render=False,
        action_space_type=ACTION_SPACE_TYPE,
        force_mag=FORCE_MAG
    )
    model = PPO(vec_env, buffer_size=2**14)
    pth = 'saved_models/ppo_cartpole.pth'
    if os.path.exists(pth):
        print('Found existing weights. Start loading')
        model.load(pth)
    else:
        print('Creating new model') 
        model.learn(total_timesteps=1_000_000)
    vec_env.close()
    vec_env = make_vec_env(
        CartPoleBulletEnv,
        n_envs=4,
        render=False,
        action_space_type=ACTION_SPACE_TYPE,
        force_mag=FORCE_MAG
    )
    model = PPO(vec_env, buffer_size=2**10, experiment_name='ppo_cartpole_bullet')
    model.load(pth)
    model.learn(total_timesteps=100_000)
    print('Training complete. Saving model weights.')
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
        
        