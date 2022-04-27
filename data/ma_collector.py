from typing import Any, Union

import gym
import numpy as np
from tianshou.data import AsyncCollector
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from marl_comm.env import get_MA_VectorEnv


class MultiAgentCollector(AsyncCollector):
    def __init__(
        self,
        env: Union[gym.Env, BaseVectorEnv],
        **kwargs: Any,
    ) -> None:
        if hasattr(env, "num_agents"):
            agents = env.agents
        else:
            agents = env.get_env_attr("agents", [0])[0]
        if hasattr(env, "agent_idx"):
            agent_idx = env.agent_idx
        else:
            agent_idx = env.get_env_attr("agents_idx", [0])[0]

        self.agents = agents
        self.agent_idx = agent_idx
        self.agent_num = len(agent_idx)

        if not isinstance(env, BaseVectorEnv):
            env = get_MA_VectorEnv(DummyVectorEnv, [lambda: env])

        super().__init__(env=env, **kwargs)

    def reset_env(self) -> None:
        """Reset all of the environments.
        obs is inited in the following format:
        [{'agent_id': 'agent_id_for_agent0', 'obs': obs0}, {'agent_id': 'agent_id_for_agent1', 'obs': empty_array}, ...]
        """
        local_obs = self.env.reset()

        self._ready_env_ids = [
            self.agent_idx[local_obs[env_i]["agent_id"]] * len(local_obs) + env_i
            for env_i in range(len(local_obs))
        ]
        global_obs = []
        for agent_i in range(0, self.agent_num):
            for _ in range(self.env.env_num):  # self.env.env_num is the num of MAEnvs
                item = {
                    "agent_id": self.agents[agent_i],
                    "obs": np.empty_like(local_obs[0]["obs"]),
                }
                if "mask" in local_obs[0]:
                    item["mask"] = local_obs[0]["mask"]
                global_obs.append(item)

        for local_env_i, obs in enumerate(local_obs):
            global_env_i = (
                self.agent_idx[obs["agent_id"]] * len(local_obs) + local_env_i
            )
            global_obs[global_env_i] = obs

        obs = global_obs

        if self.preprocess_fn:
            obs = self.preprocess_fn(obs=global_obs, env_id=self._ready_env_ids).get(
                "obs", global_obs
            )
        self.data.obs = obs