import pettingzoo.butterfly.pistonball_v6 as pistonball_v6
from tianshou.env import BaseVectorEnv, SubprocVectorEnv

from marl_comm.env import MAEnvWrapper, get_MA_VectorEnv_cls

# single env
env = MAEnvWrapper(pistonball_v6.env(continuous=False, n_pistons=2))
print("env len", len(env))  # 2
obs = env.reset()
# print("init obs:", obs)
act = env.action_space.sample()
print("act:", act)
obs, reward, done, info = env.step(act)
# print("next_obs:", obs)
print("reward:", reward)
print("done:", done)
print("info:", info)
print("action space", env.action_space)
print("observation space", env.observation_space)

# multiple envs


def get_env():
    return MAEnvWrapper(pistonball_v6.env(continuous=False, n_pistons=2))


ma_venv_cls = get_MA_VectorEnv_cls(SubprocVectorEnv)

venv = ma_venv_cls([get_env for _ in range(3)])

print("venv len", len(venv))  # 6
obs = venv.reset()
# print("init obs:", obs)
act = [act_sp.sample() for act_sp in venv.get_env_attr("action_space")]
print("act:", act)
obs, reward, done, info = venv.step(act)
# print("next_obs:", obs)
print("reward:", reward)
print("done:", done)
print("info:", info)
print("action space", venv.action_space)
print("observation space", venv.observation_space)
