import mindspore.nn as nn
import mindspore
import numpy as np


class LeNet5(nn.Cell):
    """
    LeNet-5网络结构
    """
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        # 卷积层，输入的通道数为num_channel，输出的通道数为6，卷积核大小为5*5
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        # 卷积层，输入的通道数为6，输出的通道数为16，卷积核大小为5*5
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        #进行实验加入celllist
        self.a=nn.CellList()
        tmp=[]
        tmp.append([self.conv1, self.conv2])  #应该是这个list的原因，会造成参数名字的丢失
        #print(tmp)
        self.a.append(nn.SequentialCell(*tmp))
        #print(self.a)
        #print(self.conv1)  #区别在于，两者放到了同一个list当中
        # 全连接层，输入个数为16*5*5，输出个数为120
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        # 全连接层，输入个数为120，输出个数为84
        self.fc2 = nn.Dense(120, 84)
        # 全连接层，输入个数为84，分类的个数为num_class
        self.fc3 = nn.Dense(84, num_class)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 池化层
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # 多维数组展平为一维数组
        self.flatten = nn.Flatten()

    def construct(self, x):
        # 使用定义好的运算构建前向网络
        # x = self.conv1(x)
        # x = self.relu(x)
        x=self.a(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

network = LeNet5(num_class=10)


# for name, param in network.parameters_and_names():
#     print(name)
#     print(param.name)

x=mindspore.Tensor(np.ones((1,1,32,32)),mindspore.float32)
# out=network(x)
# print(out)