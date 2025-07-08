import multiprocessing as mp
import numpy as np
import functools

def env_worker(conn, env_fn):
    env = env_fn()
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == 'reset':
                obs = env.reset()
                conn.send(obs)
            elif cmd == 'step':
                obs, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    obs, info = env.reset()
                conn.send((obs, reward, terminated, truncated, info))
            elif cmd == 'close':
                env.close()
                conn.close()
                break
            elif cmd == 'render':
                env.render()
                conn.send(None)
            elif cmd == 'get_spaces':
                conn.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError(f"Command '{cmd}' not recognized.")
    except KeyboardInterrupt:
        print("Environment worker interrupted.")
    finally:
        env.close()
        conn.close()
        
class VecEnv:
    
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)

        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in range(self.num_envs)])

        self.processes = [
            mp.Process(target=env_worker, args=(child_conn, env_fn), daemon=True)
            for child_conn, env_fn in zip(self.child_conns, env_fns)
        ]
        for process in self.processes:
            process.start()
            
        for conn in self.child_conns:
            conn.close()
            
        self.parent_conns[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.parent_conns[0].recv()
        
    def step(self, actions):
        for conn, action in zip(self.parent_conns, actions):
            conn.send(('step', action))
            
        results = [conn.recv() for conn in self.parent_conns]
        
        obs, rewards, terminations, truncations, infos = zip(*results)
        dones = [t or tr for t, tr in zip(terminations, truncations)]
        return (
            np.stack(obs),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            list(infos)
        )
        
    def reset(self):
        for conn in self.parent_conns:
            conn.send(('reset', None))
        
        results = [conn.recv() for conn in self.parent_conns]
        obs, infos = zip(*results)
        return np.stack(obs), list(infos)

    def close(self):
        for conn in self.parent_conns:
            conn.send(('close', None))
        
        for process in self.processes:
            process.join()
        
        for conn in self.parent_conns:
            conn.close()
        
        print("VecEnv закрыта.")
        
    def render(self, mode='human'):
        for conn in self.parent_conns:
            conn.send(('render', None))
            conn.recv()

def create_env(env_class, **kwargs):
    return env_class(**kwargs)

def make_vec_env(env_class, n_envs=1, **kwargs):
    env_fns = [functools.partial(create_env, env_class, **kwargs) for _ in range(n_envs)]
    return VecEnv(env_fns)