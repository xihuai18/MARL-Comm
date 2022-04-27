import pettingzoo.butterfly.pistonball_v6 as pistonball_v6
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.env import SubprocVectorEnv

env = PettingZooEnv(pistonball_v6.env(continuous=False, n_pistons=2))

def get_env():
    return PettingZooEnv(pistonball_v6.env(continuous=False, n_pistons=2))

venv = SubprocVectorEnv([get_env for _ in range(2)])