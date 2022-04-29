from typing import Any, List, Optional, Tuple, Type, Union

import numpy as np
from tianshou.data import Batch, ReplayBuffer, ReplayBufferManager, VectorReplayBuffer


class MAReplayBuffer(ReplayBufferManager):
    def __init__(
        self,
        total_size: int,
        agents: List[str],
        buffer_cls: Type[ReplayBuffer],
        ma_env_num: int = None,
        **kwargs: Any
    ) -> None:
        """MultiAgent ReplayBuffer.

        :param int size: the size for each agent's buffer.
        :param int agent_num:
        """
        agent_num = len(agents)
        assert agent_num > 0
        assert not issubclass(buffer_cls, ReplayBufferManager) or ma_env_num is not None

        buffer_list = [
            buffer_cls(total_size, ma_env_num, **kwargs)
            if issubclass(buffer_cls, ReplayBufferManager)
            else buffer_cls(total_size, **kwargs)
            for _ in range(agent_num)
        ]
        super().__init__(buffer_list)

        self.buffer_num = sum(
            [
                buf.buffer_num if issubclass(buffer_cls, ReplayBufferManager) else 1
                for buf in self.buffers
            ]
        )
        self.ma_env_num = ma_env_num or 1
        self.ma_buffer_num = len(self.buffers)
        self.agents = agents

    def _set_batch_for_children(self) -> None:
        for buf in self.buffers:
            buf._set_batch_for_children()

    def get_agent_buffer(
        self, agent_id: int
    ) -> Union[ReplayBuffer, ReplayBufferManager]:
        return self.buffers[agent_id]

    def add(
        self, batch: Batch, buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        buffer_ids = np.asarray(buffer_ids)
        idxs = np.arange(len(buffer_ids))

        ptrs = np.empty_like(buffer_ids)
        ep_rews = np.empty_like(buffer_ids)
        ep_lens = np.empty_like(buffer_ids)
        ep_idxs = np.empty_like(buffer_ids)

        for i in range(self.ma_buffer_num):
            _buffer_ids = buffer_ids[(buffer_ids // self.ma_env_num) == i]
            _idxs = idxs[(buffer_ids // self.ma_env_num) == i]
            if len(_buffer_ids) == 0:
                continue

            _batch = batch[_idxs]

            _ma_buffer_ids = _buffer_ids % self.ma_env_num

            _ptrs, _ep_rews, _ep_lens, _ep_idxs = self.buffers[i].add(
                _batch, _ma_buffer_ids
            )
            ptrs[_idxs] = _ptrs
            ep_rews[_idxs] = _ep_rews
            ep_lens[_idxs] = _ep_lens
            ep_idxs[_idxs] = _ep_idxs

        return ptrs, ep_rews, ep_lens, ep_idxs

    def sample_indices(self, batch_size: int) -> np.ndarray:
        return self.buffers[0].sample_indices(batch_size)

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        sample = Batch()
        indices = self.sample_indices(batch_size)
        for agent_i, agent in enumerate(self.agents):
            sample[agent] = self.buffers[agent_i][indices]
        return sample, indices
