import mindspore as ms
import mindspore.nn as nn
import numpy as np


class Name(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.a = nn.SequentialCell([Block(dim=dim, num_heads=4)])
        # print((nn.SequentialCell([Block(dim=dim, num_heads=4)]))[0]) 可以使用标签读取，从而读取sequential当中的数据
        self.a.append((nn.SequentialCell([Block(dim=dim, num_heads=4)]))[0])
        self.b = nn.SequentialCell([Block(dim=dim, num_heads=4)])
        self.b.append((nn.SequentialCell([Block(dim=dim, num_heads=4)]))[0])

    def construct(self, x):
        out = self.a(x)
        outs = self.b(out)
        return outs


class Block(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.con1 = nn.Conv2d(3, 3, 1)
        self.con2 = nn.Conv2d(3, 3, 1)

    def construct(self, x):
        x = self.con1(x)
        #print(x.shape)
        out = self.con2(x)
        return out


net = Name(dim=(16,))
# for name, param in net.parameters_and_names():
#     print(name)
#     print(param)

x = ms.Tensor(np.ones([3, 3, 3, 3]), ms.float32)
x = net(x).shape
print(x)
