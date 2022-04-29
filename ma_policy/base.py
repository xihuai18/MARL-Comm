from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tianshou.data import Batch, ReplayBuffer
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy

from marl_comm.data import MAReplayBuffer


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
        :param str parameter_mode: parameter mode in CTDE, choices include {"Indvd", "shared", "IndvdGI"}, defaults to "Indvd"
        :param str critic_mode: choices include {"IC", "JC"}, defaults to "IC"
        :param bool comm: whether to use communication
        """
        assert train_scheme in ["CTDE",
                                "FD"], "train_scheme must be in {'CTDE', 'FD'}"
        assert parameter_mode in [
            "Indvd",
            "shared",
            "IndvdGI",
        ], 'parameter_mode must be in {"Indvd", "shared", "IndvdGI"}'
        assert critic_mode in ["IC",
                               "JC"], "critic_mode must be in {'IC', 'JC'}"
        super().__init__(action_space=env.action_space, **kwargs)
        self.train_scheme = train_scheme
        self.parameter_mode = parameter_mode
        self.critic_mode = critic_mode
        self.comm = comm

        self.agent_idx = env.agent_idx

        assert self._check_policy(
            policies, env), "policies are not consistent with the paramters"

        if not self.parameter_mode == "shared":
            assert len(policies) == len(
                env.agents), "One policy must be assigned for each agent."

            for i, policy in enumerate(policies):
                policy.set_agent_id(env.agents[i])
        else:
            # shallow copy for each agent
            assert (len(policies) == 1
                    ), "Only one policy can be assigned for parameter sharing."
            # The agent id is 0 for the parameter sharing policy
            policies = policies * len(env.agents)

        self.policies = dict(zip(env.agents, policies))

    def replace_policy(self, policy: BasePolicy, agent_id: int) -> None:
        """Replace the "agent_id"th policy in this manager."""
        policy.set_agent_id(agent_id)
        self.policies[agent_id] = policy

    def _process_critic_input(
        self,
        batch: Batch,
        state: Batch = None,
        buffer: MAReplayBuffer = None,
        indice: np.ndarray = None,
    ) -> Batch:
        """Process the input of the critic when updating policies, such as adding messages, or make joint obs"""
        return batch

    def _process_actor_input(
        self,
        batch: Batch,
        state: Batch = None,
        buffer: ReplayBuffer = None,
        indice: np.ndarray = None,
    ) -> Batch:
        """Process the input of the actor when updating policies, such as adding messages"""
        return batch

    def _check_policy(self, policies: List[BasePolicy],
                      env: PettingZooEnv) -> bool:
        """Check the input dim of the actor and critic, without considering the preprocessing net."""
        # TODO: We can only access the last layer of the actor and critic, while it is hard to check whether the input dim of the preprocessing net is right.
        return True

    def process_fn(self, batch: Batch, buffer: MAReplayBuffer,
                   indice: np.ndarray) -> Batch:
        """The batch from buffer has the structure
        {
            "agent_1/item1": item 1 of agent_1's policy.learn output
            "agent_1/item2": item 2 of agent_1's policy.learn output
            "agent_2/xxx": xxx
            ...
            "agent_n/xxx": xxx
        }
        """
        batch = self._process_critic_input(batch, buffer=buffer, indice=indice)

        results = {}
        for agent_i, (agent, policy) in enumerate(self.policies.items()):
            results[agent] = policy.process_fn(
                batch[agent], buffer.get_agent_buffer(agent_i), indice)
        return Batch(results)

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        """Add exploration noise from sub-policy onto act."""
        for agent_id, policy in self.policies.items():
            agent_index = np.nonzero(batch.obs.agent_id == agent_id)[0]
            if len(agent_index) == 0:
                continue
            act[agent_index] = policy.exploration_noise(
                act[agent_index], batch[agent_index])
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
                "policy": {
                    "agent_1": output of agent_1's policy for the input
                    "agent_2": xxx
                    ...
                    "agent_n": xxx}
            }
        """
        results: List[Tuple[bool, np.ndarray, Batch, Union[np.ndarray, Batch],
                            Batch]] = []
        for agent_id, policy in self.policies.items():
            agent_index = np.nonzero(batch.obs.agent_id == agent_id)[0]
            if len(agent_index) == 0:
                # (has_data, agent_index, out, act, state)
                results.append(
                    (False, np.array([-1]), Batch(), Batch(), Batch()))
                continue
            tmp_batch = batch[agent_index]
            if not hasattr(tmp_batch.obs, "mask"):
                if hasattr(tmp_batch.obs, "obs"):
                    tmp_batch.obs = tmp_batch.obs.obs
                if hasattr(tmp_batch.obs_next, "obs"):
                    tmp_batch.obs_next = tmp_batch.obs_next.obs
            out = policy(
                batch=tmp_batch,
                state=None if state is None else state[agent_id],
                **kwargs,
            )
            act = out.act
            each_state = (out.state if
                          (hasattr(out, "state")
                           and out.state is not None) else Batch())
            out.state = each_state
            results.append((True, agent_index, out, act, each_state))
        holder = Batch.cat([{
            "act": act
        } for (has_data, agent_index, out, act, each_state) in results
                            if has_data])
        state_dict, out_dict = {}, {}
        for (agent_id, _), (has_data, agent_index, out, act,
                            state) in zip(self.policies.items(), results):
            if has_data:
                holder.act[agent_index] = act
            state_dict[agent_id] = state
            out_dict[agent_id] = out
        holder[
            "policy"] = out_dict  # other infos could be added to holder["policy"]
        if not any([b.is_empty() for b in state_dict.values()]):
            holder["state"] = state_dict
        return holder

    def learn(self, batch: Batch,
              **kwargs: Any) -> Dict[str, Union[float, List[float]]]:
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
        if not (self.train_scheme == "CTDE"
                and self.parameter_mode == "shared"):
            results = {}
            for agent_id, policy in self.policies.items():
                data = batch[agent_id]
                if not data.is_empty():
                    out = policy.learn(batch=data, **kwargs)
                    for k, v in out.items():
                        results[agent_id + "/" + k] = v
        else:
            # CTDE with shared parameters
            data = Batch.cat(
                [data[agent_id] for agent_id in self.policies.keys()])
            if not data.is_empty():
                policy = self.policies[0]
                results = policy.learn(batch=data, **kwargs)

        return results
