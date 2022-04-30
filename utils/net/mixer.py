from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tianshou.env import PettingZooEnv
from tianshou.utils.net.common import MLP


class QMixer(nn.Module):

    def __init__(
        self,
        agent_num: int,
        state_space: Union[int, Sequence[int]],
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.agent_num = agent_num
        self.state_dim = int(np.prod(state_space))
        self.hidden_sizes = hidden_sizes
        if len(self.hidden_sizes) == 1:
            self.hyper_w_1 = MLP(self.state_dim,
                                 self.hidden_sizes[0] * self.agent_num,
                                 device=device)
            self.hyper_w_final = MLP(self.state_dim,
                                     self.hidden_sizes[0],
                                     device=device)
        elif len(self.hidden_sizes) == 2:
            self.hyper_w_1 = MLP(self.state_dim,
                                 self.hidden_sizes[1] * self.agent_num,
                                 [self.hidden_sizes[0]],
                                 device=device)
            self.hyper_w_final = MLP(self.state_dim,
                                     self.hidden_sizes[1],
                                     [self.hidden_sizes[0]],
                                     device=device)
        else:
            raise NotImplementedError

        self.hyper_b_1 = MLP(self.state_dim,
                             self.hidden_sizes[-1],
                             device=device)

        self.V = MLP(self.state_dim, 1, [self.hidden_sizes[-1]], device=device)

    def forward(self, agent_qs: torch.Tensor,
                state: torch.Tensor) -> torch.Tensor:
        """
        :param agent_qs: [batch_size, agent_num, 1]
        :param state: [batch_size, state_dim]
        :return: [batch_size, 1]
        """
        bsz = agent_qs.shape[0]
        state = state.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.agent_num)

        w1 = torch.abs(self.hyper_w_1(state))
        b1 = self.hyper_b_1(state)
        w1 = w1.view(-1, self.agent_num, self.hidden_sizes[-1])
        b1 = b1.view(-1, 1, self.hidden_sizes[-1])
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        w_final = torch.abs(self.hyper_w_final(state))
        w_final = w_final.view(-1, self.hidden_sizes[-1], 1)

        v = self.V(state).view(-1, 1, 1)

        y = torch.bmm(hidden, w_final) + v

        q_tot = y.view(bsz, -1, 1)

        return q_tot