"""
MindSpore implementation of `DenseNet`.
Refer to: Densely Connected Convolutional Networks
"""

import math
from collections import OrderedDict
from typing import Tuple

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init
from mindspore import Tensor

# from .layers.pooling import GlobalAvgPooling
# from .utils import load_pretrained
# from .registry import register_model

# __all__ = [
#     "DenseNet",
#     "densenet121",
#     "densenet161",
#     "densenet169",
#     "densenet201"
# ]

import mindspore.nn as nn
import mindspore.ops as ops


class GlobalAvgPooling(nn.Cell):

    def __init__(self,
                 keep_dims: bool = False
                 ) -> None:
        super(GlobalAvgPooling, self).__init__()
        self.mean = ops.ReduceMean(keep_dims=keep_dims)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': 'features.conv0', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'densenet121': _cfg(url='https://download.mindspore.cn/toolkits/mindcv/densenet/densenet_121_imagenet2012.ckpt'),
    'densenet169': _cfg(url='https://download.mindspore.cn/toolkits/mindcv/densenet/densenet_169_imagenet2012.ckpt'),
    'densenet201': _cfg(url='https://download.mindspore.cn/toolkits/mindcv/densenet/densenet_201_imagenet2012.ckpt'),
    'densenet161': _cfg(url='https://download.mindspore.cn/toolkits/mindcv/densenet/densenet_161_imagenet2012.ckpt'),
}


class _DenseLayer(nn.Cell):
    """Basic unit of DenseBlock (using bottleneck layer)"""

    def __init__(self,
                 num_input_features: int,
                 growth_rate: int,
                 bn_size: int,
                 drop_rate: float
                 ) -> None:
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, pad_mode='pad', padding=1)

        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(keep_prob=1 - self.drop_rate)

    def construct(self, features: Tensor) -> Tensor:
        bottleneck = self.conv1(self.relu1(self.norm1(features)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck)))
        if self.drop_rate > 0.:
            new_features = self.dropout(new_features)
        return new_features


class _DenseBlock(nn.Cell):
    """DenseBlock. Layers within a block are densely connected."""

    def __init__(self,
                 num_layers: int,
                 num_input_features: int,
                 bn_size: int,
                 growth_rate: int,
                 drop_rate: float) -> None:
        super(_DenseBlock, self).__init__()
        self.cell_list = nn.CellList()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.cell_list.append(layer)

    def construct(self, init_features: Tensor) -> Tensor:
        features = init_features
        for layer in self.cell_list:
            new_features = layer(features)
            features = ops.concat((features, new_features), axis=1)
        return features


class _Transition(nn.Cell):
    """Transition layer between two adjacent DenseBlock"""

    def __init__(self,
                 num_input_features: int,
                 num_output_features: int,
                 ) -> None:
        super(_Transition, self).__init__()
        self.features = nn.SequentialCell(OrderedDict([
            ('norm', nn.BatchNorm2d(num_input_features)),
            ('relu', nn.ReLU()),
            ('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1)),
            ('pool', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

    def construct(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x


class DenseNet(nn.Cell):
    def __init__(self,
                 growth_rate: int = 32,
                 block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
                 num_init_features: int = 64,
                 bn_size: int = 4,
                 drop_rate: float = 0.,
                 in_channels: int = 3,
                 num_classes: int = 1000) -> None:
        super(DenseNet, self).__init__()
        layers = OrderedDict()
        # first Conv2d
        num_features = num_init_features
        layers['conv0'] = nn.Conv2d(in_channels, num_features, kernel_size=7, stride=2, pad_mode='pad', padding=3)
        layers['norm0'] = nn.BatchNorm2d(num_features)
        layers['relu0'] = nn.ReLU()
        layers['pool0'] = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        # DenseBlock
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            layers['denseblock%d' % (i + 1)] = block
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                transition = _Transition(num_features, num_features // 2)
                layers['transition%d' % (i + 1)] = transition
                num_features = num_features // 2

        # final bn+ReLU
        layers['norm5'] = nn.BatchNorm2d(num_features)
        layers['relu5'] = nn.ReLU()

        self.num_features = num_features
        self.features = nn.SequentialCell(layers)
        self.pool = GlobalAvgPooling()
        self.classifier = nn.Dense(self.num_features, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.HeNormal(math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer(init.HeUniform(math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu'),
                                         cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer('ones', cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer('zeros', cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.HeUniform(math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu'),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


net=DenseNet()
for name, param in net.parameters_and_names():
    print(name, param)
#
# @register_model
# def densenet121(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> DenseNet:
#     default_cfg = default_cfgs['densenet121']
#     model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, in_channels=in_channels,
#                      num_classes=num_classes, **kwargs)
#
#     if pretrained:
#         load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
#
#     return model
#
#
# @register_model
# def densenet161(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> DenseNet:
#     default_cfg = default_cfgs['densenet161']
#     model = DenseNet(growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96, in_channels=in_channels,
#                      num_classes=num_classes, **kwargs)
#
#     if pretrained:
#         load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
#
#     return model
#
#
# @register_model
# def densenet169(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> DenseNet:
#     default_cfg = default_cfgs['densenet169']
#     model = DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64, in_channels=in_channels,
#                      num_classes=num_classes, **kwargs)
#
#     if pretrained:
#         load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
#
#     return model
#
#
# @register_model
# def densenet201(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> DenseNet:
#     default_cfg = default_cfgs['densenet201']
#     model = DenseNet(growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64, in_channels=in_channels,
#                      num_classes=num_classes, **kwargs)
#
#     if pretrained:
#         load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
#
#     return model
