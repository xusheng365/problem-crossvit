import mindspore.nn as nn
import numpy as np
import mindspore as ms

x = ms.Tensor(np.ones((3, 3, 3, 3)),ms.float32)
x = np.reshape(x, (3, 3, 3*3))
print(x.shape)
