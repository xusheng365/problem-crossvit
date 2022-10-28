import mindspore
import mindspore.nn as nn
import mindspore as ms
import numpy as np
from mindspore import Tensor, ops
import mindspore.ops.functional as F

# x = Tensor(np.random.rand(5, 10, 10), ms.float32)
# #print(x)
# shape1 = x.shape[2:]
# print(type(shape1))
# print(shape1)
# m = nn.LayerNorm(shape1,  begin_norm_axis=2, begin_params_axis=2)
# output = m(x).shape
# print(output)
#
# #进行层归一化，pytorch表示可以输入单的，但是必须与最后一维保持相同的维度大小
#
# def zz(x):
#     print(type(x))
#
# x=1
# print(type(x))
# zz(1)


# import mindspore.nn as nn
#
# conv = nn.Conv2d(100, 20, 3)
# bn = nn.BatchNorm2d(20)
# relu = nn.ReLU()
# print(bn)
# cell_ls = nn.CellList([bn])
# cell_ls.insert(0, conv)
# cell_ls.append(relu)
# cell_ls.extend([relu, relu])


# a = ms.Tensor(np.ones([2, 1, 3]).astype(np.float32))
# print(type(a))
# b = ms.Tensor(np.ones([2, 1, 3]).astype(np.float32))
# print(type(b))
# concat_op = ms.ops.Concat(1)
# output = concat_op((a, b))
# print(output.shape)

# a = ms.Parameter(Tensor(np.array([1.0, 1.0, 1.0], np.float32)))
# print(a)
# d = []
# d.append(a)
# d = tuple(d)
# b = ms.ParameterTuple(d)
# print(b)
from mindspore.common.initializer import initializer, TruncatedNormal

# d=[]
# c = ms.Parameter(Tensor(np.ones([1, 4, 3], np.float32)))
# for i in range(3):
#     c = ms.Parameter(Tensor(np.ones([i+1, 4, 3], np.float32)), name=str(i))
#     d.append(c)
# d=tuple(d)
# g = ms.ParameterTuple(d)
# print(g)


# tensor1 = initializer(TruncatedNormal(), [1, 2, 3], ms.float32)
# print(type(tensor1))

# input_x = Tensor(np.arange(16).reshape((2, 2, 2, 2)).astype(np.float32))
# grid = Tensor(np.arange(0.2, 1, 0.1).reshape((2, 2, 1, 2)).astype(np.float32))
# output = F.grid_sample(input_x, grid, interpolation_mode='bilinear', padding_mode='zeros',
#                        align_corners=True)
# print(output)



# x = Tensor(np.ones([3, 3, 1, 16]))
# y = mindspore.dataset.vision.Inter(x, (224, 224), interpolation_mode='bicubic', padding_mode='zeros', align_corners=True)
# print(y)

import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
x = Tensor(np.ones((1, 1, 192)), dtype=mstype.float32)
y1 = Tensor(np.ones((3, 1, 96)), dtype=mstype.float32)
y2 = Tensor(np.ones((3, 1, 192)), dtype=mstype.float32)
y = [y1, y2]
# output = x.expand_as(y1)
#print(output)

d = []
#print("i ",i)
c = ms.Parameter(Tensor(np.zeros([1, 1, 96], np.float32)), name='1')
d.append(c)
e = ms.Parameter(Tensor(np.zeros([1, 1, 192], np.float32)), name='2')
d.append(e)
#print(d)
d = tuple(d)
cls_token = ms.ParameterTuple(d)


for i in range(2):
    cls_tokens = cls_token[i]
    p=cls_token[i].T
    print(p)
    z=cls_tokens.expand_as(y[i])
    print(z.shape)






#
#
# x = Tensor(np.ones([3, 3, 1, 16]))
# y = ops.grid_sample(x, (224, 224), interpolation_mode='bicubic', padding_mode='zeros', align_corners=True)
# print(y)

