import mindspore as ms
import mindspore.nn as nn
import numpy as np


class MyModule(nn.Cell):
    def __init__(self):
        super().__init__(auto_prefix=True) #事实证明这个auto_prefix没有用，没有任何作用
        self.linears = nn.SequentialCell()
        self.linears.append(nn.SequentialCell([nn.Conv2d(10, 10, kernel_size=1, has_bias=True)]))
        #self.linears2=nn.CellList()
        #self.linears2.append(nn.SequentialCell([nn.Conv2d(10, 10, kernel_size=1, has_bias=True)]))
        print(self.linears)

    def construct(self, x):
        # ModuleList can act as an iterable, or be indexed         using ints
        x = self.linears(x)
        return x

net = MyModule()
for name, param in net.parameters_and_names():
    print(name, param)
x = ms.Tensor(np.ones((1, 10, 220, 10)), ms.float32)
x = net(x)  # nn.linear仅仅是一个线性变换
print(x)
