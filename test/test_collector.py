from pprint import pprint

import gym
import pettingzoo.butterfly.pistonball_v6 as pistonball_v6
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import BaseVectorEnv, SubprocVectorEnv
from tianshou.policy import RandomPolicy

from marl_comm.data import MACollector
from marl_comm.env import MAEnvWrapper, get_MA_VectorEnv_cls
from marl_comm.ma_policy import MAPolicyManager

n_pistons = 5
n_envs = 3


def get_env():
    return MAEnvWrapper(
        pistonball_v6.env(continuous=False, n_pistons=n_pistons))


def get_policy():
    env = get_env()

    observation_space = (env.observation_space["observation"] if isinstance(
        env.observation_space, gym.spaces.Dict) else env.observation_space)

    agents = [
        RandomPolicy(observation_space, env.action_space)
        for _ in range(n_pistons)
    ]

    policy = MAPolicyManager(agents, env)

    return policy


def test_single_env():
    policy = get_policy()
    env = get_env()

    collector = MACollector(
        policy=policy,
        env=env,
        buffer=VectorReplayBuffer(2000, len(env)),
        exploration_noise=True,
    )

    stats = collector.collect(n_step=1000)

    env.close()

    return collector, stats


def test_vector_env():
    ma_venv_cls = get_MA_VectorEnv_cls(SubprocVectorEnv)
    venv = ma_venv_cls([get_env for _ in range(n_envs)])

    policy = get_policy()

    collector = MACollector(
        policy=policy,
        env=venv,
        buffer=VectorReplayBuffer(2000, len(venv)),
        exploration_noise=True,
    )

    stats = collector.collect(n_step=1000 * n_envs)

    venv.close()

    return collector, stats


if __name__ == "__main__":
    print("single env")
    collector, stats = test_single_env()
    print("stats")
    pprint(stats)

    print("vector env")
    collector, stats = test_vector_env()
    print("stats")
    pprint(stats)
