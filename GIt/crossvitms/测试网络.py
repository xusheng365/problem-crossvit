from mindspore import nn
import mindspore as ms
import numpy as np
from models.new import VisionTransformer

net4 = VisionTransformer(img_size=[240, 224],
                         patch_size=[12, 16], embed_dim=[96, 192], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                         num_heads=[6, 6], mlp_ratio=[4, 4, 1], qkv_bias=True,
                         norm_layer=nn.LayerNorm)

# for name, param in net4.parameters_and_names():
#     print(name, param)
x = ms.Tensor(np.ones((3, 3, 1, 16)), ms.float32)
out = net4(x)
print(out)
