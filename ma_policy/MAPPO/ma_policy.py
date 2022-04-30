from typing import Any, List

import numpy as np
from marl_comm.data import MAReplayBuffer
from marl_comm.ma_policy import MAPolicyManager
from tianshou.data import Batch
from tianshou.env import PettingZooEnv
from tianshou.policy import PPOPolicy


class MAPPOPolicy(MAPolicyManager):

    def __init__(self,
                 policies: List[PPOPolicy],
                 env: PettingZooEnv,
                 joint_critic: bool = False,
                 **kwargs: Any) -> None:
        train_scheme = "FD" if not joint_critic else "CTDE"
        parameter_mode = "Indvd"
        critic_mode = "IC" if not joint_critic else "JC"
        comm = False
        super().__init__(policies, env, train_scheme, parameter_mode,
                         critic_mode, comm, **kwargs)

    def process_fn(self, batch: Batch, buffer: MAReplayBuffer,
                   indice: np.ndarray) -> Batch:
        for agent in self.agents:
            if self.critic_mode == "JC":
                bsz = batch[agent].obs.obs.shape[0]
                batch[agent].critic_obs = np.concatenate([
                    batch[agent].obs.obs.reshape(bsz, -1),
                    batch[agent].obs.state.reshape(bsz, -1)
                ],
                                                         axis=-1)

                batch[agent].critic_obs_next = np.concatenate([
                    batch[agent].obs_next.obs.reshape(bsz, -1),
                    batch[agent].obs_next.state.reshape(bsz, -1)
                ],
                                                              axis=-1)
            else:
                batch[agent].critic_obs = batch[agent].obs.obs
                batch[agent].critic_obs_next = batch[agent].obs_next.obs
            batch[agent].obs = batch[agent].obs.obs
            batch[agent].obs_next = batch[agent].obs_next.obs
        
        results = super().process_fn(batch, buffer, indice)
        return results