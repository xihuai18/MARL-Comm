from tianshou.data import Batch, ReplayBuffer, VectorReplayBuffer
import numpy as np


def test_replaybuffer():
    obs = np.arange(12)
    next_obs = np.arange(1, 13)
    buffer = ReplayBuffer(100)
    for i in range(100):
        batch = Batch({"obs": {"obs": obs, "agent_id": f"agent_{i % 10}"}, "next_obs": next_obs,
                      "act": i, "rew": i**2, "done": i % 2})
        buffer.add(batch)
    print(buffer.sample(2))

    vbuffer = VectorReplayBuffer(100, 5)
    for i in range(100):
        batch = Batch({"obs": {"obs": [obs], "agent_id": [f"agent_{i % 10}"]}, "next_obs": [next_obs],
                      "act": [i], "rew": [i**2], "done": [i % 2]})
        vbuffer.add(batch, [i % 5])
    print(vbuffer.sample(2))


if __name__ == '__main__':
    test_replaybuffer()
