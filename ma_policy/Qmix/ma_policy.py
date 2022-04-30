import copy
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
from tianshou.data import ReplayBuffer, to_torch_as
import torch
import torch.nn as nn
from marl_comm.data import MAReplayBuffer
from marl_comm.ma_policy import MAPolicyManager
from tianshou.data import Batch
from tianshou.env import PettingZooEnv
from tianshou.policy import DQNPolicy


class QMIXPolicy(MAPolicyManager):

    def __init__(self,
                 policies: List[DQNPolicy],
                 env: PettingZooEnv,
                 discount_factor: float = 0.99,
                 estimation_step: int = 1,
                 target_update_freq: int = 0,
                 reward_normalization: bool = False,
                 mixer: Optional[nn.Module] = None,
                 mixer_lr: Optional[float] = 0.0005,
                 mixer_optim_cls: Type[
                     torch.optim.Optimizer] = torch.optim.Adam,
                 mixer_optim_kwargs: Optional[Dict[str, Any]] = {},
                 **kwargs: Any) -> None:
        train_scheme = "FD" if not mixer else "CTDE"
        parameter_mode = "Indvd"
        critic_mode = "IC"
        comm = False

        assert env.state_space is not None

        super().__init__(policies, env, train_scheme, parameter_mode,
                         critic_mode, comm, **kwargs)

        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self._n_step = estimation_step
        self._freq = target_update_freq
        self._iter = 0
        self._rew_norm = reward_normalization

        self.mixer = mixer
        if self.mixer:
            self.target_mixer = copy.deepcopy(self.mixer)
            self.mixer_optim = mixer_optim_cls(self.parameters(),
                                               lr=mixer_lr,
                                               **mixer_optim_kwargs)

    def sync_weight(self) -> None:
        if self.mixer:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        for policy in self._policies:
            policy.sync_weight()

    def _get_agent_q(self, buffer: MAReplayBuffer, indices: np.ndarray,
                     agent_i: int) -> torch.Tensor:
        batch = buffer.get_agent_buffer(agent_i)[indices]
        q = self._policies[agent_i](batch).logits
        q = q[np.arange(len(q)), batch.act]
        return q

    def _mixed_q(self,
                 buffer: MAReplayBuffer,
                 indices: np.ndarray,
                 target: bool = False) -> torch.Tensor:
        if target:
            target_qs = [
                self._policies[agent_i]._target_q(
                    buffer.get_agent_buffer(agent_i), indices)
                for agent_i in range(self.agent_num)
            ]
            target_qs = torch.stack(target_qs, dim=1)
            target_state = buffer.get_agent_buffer(
                0)[indices]["obs_next"]["state"]
            return self.target_mixer(target_qs, target_state)
        else:
            qs = [
                self._get_agent_q(buffer, indices, agent_i)
                for agent_i in range(self.agent_num)
            ]
            qs = torch.stack(qs, dim=1)
            state = buffer.get_agent_buffer(0)[indices]["obs"]["state"]
            return self.mixer(qs, state)

    def process_fn(self, batch: Batch, buffer: MAReplayBuffer,
                   indices: np.ndarray) -> Batch:
        if not self.mixer:
            return super().process_fn(batch, buffer, indices)
        else:

            def target_q_fn(_buffer: ReplayBuffer, indices: np.ndarray):
                return self._mixed_q(buffer, indices, target=True)

            batch = self.compute_nstep_return(batch[self.agents[0]],
                                              buffer.get_agent_buffer(0),
                                              indices, target_q_fn,
                                              self._gamma, self._n_step,
                                              self._rew_norm)
            batch.mixed_q = self._mixed_q(buffer, indices)

            return batch

    def learn(self, batch: Batch,
              **kwargs: Any) -> Dict[str, Union[float, List[float]]]:
        if not self.mixer:
            return super().learn(batch, **kwargs)
        else:
            if self._iter % self._freq == 0:
                self.sync_weight()
            self.mixer_optim.zero_grad()
            weight = batch.pop("weight", 1.0)
            mixed_q = batch.mixed_q
            returns = to_torch_as(batch.returns.flatten(), mixed_q)
            td_error = returns - mixed_q
            loss = (td_error.pow(2) * weight).mean()
            batch.weight = td_error
            loss.backward()
            self.mixer_optim.step()
            self._iter += 1
            return {"loss": loss.item()}