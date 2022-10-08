import mindspore as ms
import numpy as np
import mindspore.nn as nn

input_net = ms.Tensor(np.array([[0, 0, 0], [0, 0, 0]]), ms.float32)
net = nn.Dense(3, 4)
output = net(input_net)
print(output)