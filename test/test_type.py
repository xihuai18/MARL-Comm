from typing import List, Tuple, Type
from tianshou.data import ReplayBuffer

# test using type(name, (cls,), attrs_dict) as the template


def get_extend_replaybuffer(
    name: str, replaybuffer_cls: Type[ReplayBuffer], extend_keys: Tuple[str]
) -> Type[ReplayBuffer]:
    return type(
        name,
        (replaybuffer_cls,),
        {"_reserved_keys": replaybuffer_cls._reserved_keys + extend_keys},
    )


extend_a_cls = get_extend_replaybuffer("extend_a", ReplayBuffer, ("a",))
a = extend_a_cls(10)
print(a._reserved_keys)
extend_b_cls = get_extend_replaybuffer("extend_b", ReplayBuffer, ("b",))
b = extend_b_cls(10)
print(b._reserved_keys)
