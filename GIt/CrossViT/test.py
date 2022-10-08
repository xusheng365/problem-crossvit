import mindspore.nn as nn
import mindspore
import numpy as np
from mindspore import Tensor

x = Tensor(np.random.rand(5, 10, 10), mindspore.float32)
#print(x)
shape1 = x.shape[2:]
print(type(shape1))
print(shape1)
m = nn.LayerNorm(shape1,  begin_norm_axis=2, begin_params_axis=2)
output = m(x).shape
print(output)

#进行层归一化，pytorch表示可以输入单的，但是必须与最后一维保持相同的维度大小

def zz(x):
    print(type(x))

x=1
print(type(x))
zz(1)


import mindspore.nn as nn

conv = nn.Conv2d(100, 20, 3)
bn = nn.BatchNorm2d(20)
relu = nn.ReLU()
print(bn)
cell_ls = nn.CellList([bn])
cell_ls.insert(0, conv)
cell_ls.append(relu)
cell_ls.extend([relu, relu])
