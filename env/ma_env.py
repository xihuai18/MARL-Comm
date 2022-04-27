from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
from tianshou.env import BaseVectorEnv, PettingZooEnv


class MAEnvWrapper(PettingZooEnv):
    def step(self, action: Any) -> Tuple[Dict, List[int], bool, Dict]:
        obs, rew, done, info = super().step(action)
        info["env_id"] = self.agent_idx[obs["agent_id"]]

        return obs, rew, done, info

    def __len__(self) -> int:
        return self.num_agents


# NOTE
""" 
MAVectorEnv has the layout [(agent0, env0), (agent0, env1), ..., (agent1, env0), (agent1, env1), ...]
"""


def ma_venv_init(
    self: BaseVectorEnv,
    p_cls: Type[BaseVectorEnv],
    env_fns: List[Callable[[], gym.Env]],
    **kwargs: Any
) -> None:
    p_cls.__init__(self, env_fns, **kwargs)

    self.p_cls = p_cls

    agents = self.get_env_attr("agents")[0]
    agent_idx = self.get_env_attr("agent_idx")[0]

    self.agents = agents
    self.agent_idx = agent_idx
    self.agent_num = len(agent_idx)


def ma_venv_len(self: BaseVectorEnv) -> int:
    return sum(self.get_env_attr("num_agents"))


def ma_venv_step(
    self: BaseVectorEnv,
    action: np.ndarray,
    id: Optional[Union[int, List[int], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_stack, rew_stack, done_stack, info_stack = self.p_cls.step(self, action, id)
    for obs, info in zip(obs_stack, info_stack):
        # self.env_num is the number of environments, while the env_num in collector is `the number of agents` * `the number of environments`
        info["env_id"] = self.agent_idx[obs["agent_id"]] * self.env_num + info["env_id"]
    return obs_stack, rew_stack, done_stack, info_stack


def get_MA_VectorEnv_cls(p_cls: Type[BaseVectorEnv]) -> Type[BaseVectorEnv]:
    """
    Get the class of Multi-Agent VectorEnv.
    """

    def init_func(
        self: BaseVectorEnv, env_fns: List[Callable[[], gym.Env]], **kwargs: Any
    ) -> None:
        ma_venv_init(self, p_cls, env_fns, **kwargs)

    name = "MA" + p_cls.__name__

    attr_dict = {"__init__": init_func, "__len__": ma_venv_len, "step": ma_venv_step}

    return type(name, (p_cls,), attr_dict)
