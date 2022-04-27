from tianshou.data import Batch

import numpy as np

b = Batch({"a": {"id": [1, 2, 3], "obs": np.zeros((3, 3))}, "b": {}})

print(len(b))
