from tianshou.data import Batch, ReplayBuffer, VectorReplayBuffer
from marl_comm.data import MAReplayBuffer
import numpy as np


def test_replaybuffer():
    obs = np.arange(12)
    next_obs = np.arange(1, 13)
    buffer = MAReplayBuffer(100, ["agent_0", "agent_1"], ReplayBuffer)
    for i in range(100):
        batch = Batch(
            {
                "obs": {"obs": [obs], "agent_id": [f"agent_{i % 2}"]},
                "next_obs": [next_obs],
                "act": [i],
                "rew": [i ** 2],
                "done": [False],
            }
        )
        buffer.add(batch, [i % 2])
    print(buffer.sample(4))

    vbuffer = MAReplayBuffer(100, ["agent_0", "agent_1"], VectorReplayBuffer, 2)
    for i in range(100):
        batch = Batch(
            {
                "obs": {"obs": [obs] * 2, "agent_id": [f"agent_{i % 2}"] * 2},
                "next_obs": [next_obs] * 2,
                "act": [i] * 2,
                "rew": [i ** 2] * 2,
                "done": [i % 2] * 2,
            }
        )
        vbuffer.add(batch, [(i % 2) * 2, (i % 2) * 2 + 1])
    print(vbuffer.sample(2))


if __name__ == "__main__":
    test_replaybuffer()
