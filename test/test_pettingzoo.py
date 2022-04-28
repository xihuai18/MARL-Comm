import pettingzoo.butterfly.pistonball_v6 as pistonball_v6
import pettingzoo.mpe.simple_push_v2 as simple_push_v2
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.env import SubprocVectorEnv
import supersuit as ss

# env = PettingZooEnv(pistonball_v6.env(continuous=False, n_pistons=2))

# def get_env():
#     return PettingZooEnv(pistonball_v6.env(continuous=False, n_pistons=2))

# venv = SubprocVectorEnv([get_env for _ in range(2)])


env = PettingZooEnv(
    ss.pad_observations_v0(ss.pad_observations_v0(simple_push_v2.env()))
)


# def get_env():
#     return PettingZooEnv(simple_push_v2.env())


# venv = SubprocVectorEnv([get_env for _ in range(2)])
