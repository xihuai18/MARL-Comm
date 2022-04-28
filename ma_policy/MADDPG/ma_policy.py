from typing import List, Any
from tianshou.policy import MultiAgentPolicyManager, BasePolicy
from tianshou.env.pettingzoo_env import PettingZooEnv


class DDPGMAPolicyManager(MultiAgentPolicyManager):
    def __init__(
        self, policies: List[BasePolicy], env: PettingZooEnv, **kwargs: Any
    ) -> None:
        super().__init__(policies, env, **kwargs)
