import numpy as np
from typing import Any, List, Tuple, Type
from tianshou.data import ReplayBufferManager, ReplayBuffer, Batch


def get_extend_replaybuffer_cls(
    name: str, cls: Type[ReplayBuffer], extend_keys: Tuple[str]
) -> Type[ReplayBuffer]:
    """extend the _reserved_keys of ReplayBuffer

    :param str name: the name of the extended class
    :param Type[ReplayBuffer] cls: the base replaybuffer class
    :param Tuple[str] extend_keys: the keys to be extended
    :return Type[ReplayBuffer]: the extended replaybuffer class
    """
    attrs_dict = {"_reserved_keys": extend_keys}
    return type(name, (cls,), attrs_dict)


class MAReplayBuffer(ReplayBufferManager):
    def __init__(
        self, size: int, agent_num: int, global_infos: List[str] = None, **kwargs: Any
    ) -> None:
        assert agent_num > 0
        buffer_list = [ReplayBuffer(size, **kwargs) for _ in range(agent_num)]

        if global_infos is not None:
            replaybuffer_name = "ReplayBuffer" + "_".join(global_infos)
            replaybuffer_cls = get_extend_replaybuffer_cls(
                replaybuffer_name, ReplayBuffer, global_infos
            )
            self.global_buffer = replaybuffer_cls(size, **kwargs)
        else:
            self.global_buffer = None

        super().__init__(buffer_list)

    def sample_indices(self, batch_size: int) -> np.ndarray:
        pass

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        pass
