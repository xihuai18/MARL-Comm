from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from numba import njit
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy
from torch import nn


class MAPolicyManager(BasePolicy):
    """Base Multi-agent Policy Manager in marl_comm, including the following schemes:
    ```mermaid
    flowchart TD
        subgraph CTDE [CTDE]
            subgraph p [Parameter]
                IP([Individual Parameter])
                PS([Parameter Sharing])
                IPGI([Individual Parameter with Global Information])
            end
            subgraph c [Critic]
                IC([Individual Critic])
                JC([Joint Critic])
            end
        end
        subgraph FD [Fully Decentralized]
        end
    ```
    """

    def __init__(
        self,
        policies: List[BasePolicy],
        env: PettingZooEnv,
        train_scheme: str = "CTDE",
        parameter_mode: str = "Indvd",
        critic_mode: str = "IC",
        comm: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        :param List[BasePolicy] policies: the list of policies
        :param PettingZooEnv env: the environment
        :param str train_scheme: which training scheme to use, choices include {"CTDE", "FD"}, defaults to "CTDE"
        :param str parameter_mode: parameter mode in CTDE, choices include {"IP", "PS", "IPGI"}, defaults to "Indvd"
        :param str critic_mode: choices include {"IC", "JC"}, defaults to "IC"
        :param bool comm: whether to use communication
        """
        assert train_scheme in ["CTDE", "FD"], "train_scheme must be in {'CTDE', 'FD'}"
        assert parameter_mode in [
            "IP",
            "PS",
            "IPGI",
        ], "parameter_mode must be in {'IP', 'PS', 'IPGI'}"
        assert critic_mode in ["IC", "JC"], "critic_mode must be in {'IC', 'JC'}"
        super().__init__(action_space=env.action_space, **kwargs)
        self.train_scheme = train_scheme
        self.parameter_mode = parameter_mode
        self.critic_mode = critic_mode
        self.comm = comm

        self.agent_idx = env.agent_idx

        if not self.parameter_mode == "PS":
            assert len(policies) == len(
                env.agents
            ), "One policy must be assigned for each agent."

            for i, policy in enumerate(policies):
                policy.set_agent_id(env.agents[i])
        else:
            # shallow copy for each agent
            assert (
                len(policies) == 1
            ), "Only one policy can be assigned for parameter sharing."
            # The agent id is 0 for the parameter sharing policy
            policies = policies * len(env.agents)

        self.policies = dict(zip(env.agents, policies))

    def replace_policy(self, policy: BasePolicy, agent_id: int) -> None:
        """Replace the "agent_id"th policy in this manager."""
        policy.set_agent_id(agent_id)
        self.policies[agent_id] = policy

    def _process_critic_input(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        pass

    def _process_actor_input(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        pass

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        """Dispatch batch data from obs.agent_id to every policy's process_fn.
        Save original multi-dimensional rew in "save_rew", set rew to the
        reward of each agent during their "process_fn", and restore the
        original reward afterwards.
        """
        results = {}
        # reward can be empty Batch (after initial reset) or nparray.
        has_rew = isinstance(buffer.rew, np.ndarray)
        if has_rew:  # save the original reward in save_rew
            # Since we do not override buffer.__setattr__, here we use _meta to
            # change buffer.rew, otherwise buffer.rew = Batch() has no effect.
            save_rew, buffer._meta.rew = buffer.rew, Batch()
        for agent, policy in self.policies.items():
            agent_index = np.nonzero(batch.obs.agent_id == agent)[0]
            if len(agent_index) == 0:
                results[agent] = Batch()
                continue
            tmp_batch, tmp_indice = batch[agent_index], indice[agent_index]
            if has_rew:
                tmp_batch.rew = tmp_batch.rew[:, self.agent_idx[agent]]
                # A buffer with only the selected agent is needed for preprocessing
                buffer._meta.rew = save_rew[:, self.agent_idx[agent]]
            if not hasattr(tmp_batch.obs, "mask"):
                if hasattr(tmp_batch.obs, "obs"):
                    tmp_batch.obs = tmp_batch.obs.obs
                if hasattr(tmp_batch.obs_next, "obs"):
                    tmp_batch.obs_next = tmp_batch.obs_next.obs
            results[agent] = policy.process_fn(tmp_batch, buffer, tmp_indice)
        if has_rew:  # restore from save_rew
            buffer._meta.rew = save_rew
        return Batch(results)

    def exploration_noise(
        self, act: Union[np.ndarray, Batch], batch: Batch
    ) -> Union[np.ndarray, Batch]:
        """Add exploration noise from sub-policy onto act."""
        for agent_id, policy in self.policies.items():
            agent_index = np.nonzero(batch.obs.agent_id == agent_id)[0]
            if len(agent_index) == 0:
                continue
            act[agent_index] = policy.exploration_noise(
                act[agent_index], batch[agent_index]
            )
        return act

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Dispatch batch data from obs.agent_id to every policy's forward.

        :param state: if None, it means all agents have no state. If not
            None, it should contain keys of "agent_1", "agent_2", ...

        :return: a Batch with the following contents:

        ::

            {
                "act": actions corresponding to the input
                "state": {
                    "agent_1": output state of agent_1's policy for the state
                    "agent_2": xxx
                    ...
                    "agent_n": xxx}
                "out": {
                    "agent_1": output of agent_1's policy for the input
                    "agent_2": xxx
                    ...
                    "agent_n": xxx}
            }
        """

    def learn(
        self, batch: Batch, **kwargs: Any
    ) -> Dict[str, Union[float, List[float]]]:
        """Dispatch the data to all policies for learning.

        :return: a dict with the following contents:

        ::

            {
                "agent_1/item1": item 1 of agent_1's policy.learn output
                "agent_1/item2": item 2 of agent_1's policy.learn output
                "agent_2/xxx": xxx
                ...
                "agent_n/xxx": xxx
            }
        """
        if not self.train_scheme == "CTDE":
            results = {}
            for agent_id, policy in self.policies.items():
                data = batch[agent_id]
                if not data.is_empty():
                    out = policy.learn(batch=data, **kwargs)
                    for k, v in out.items():
                        results[agent_id + "/" + k] = v
            return results
        else:
            raise NotImplementedError
