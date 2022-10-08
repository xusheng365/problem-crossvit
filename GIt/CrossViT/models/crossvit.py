import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

from models.Block import Block
from Identity import Identity
from drop_path import DropPath
from helpers import to_2tuple
from mlp import Mlp

# 动态图的配置
# from mindspore import context
# context.set_context(mode=context.PYNATIVE_MODE)

_model_urls = {
    'crossvit_15_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_224.pth',
    'crossvit_15_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_dagger_224.pth',
    'crossvit_15_dagger_384': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_dagger_384.pth',
    'crossvit_18_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_224.pth',
    'crossvit_18_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_dagger_224.pth',
    'crossvit_18_dagger_384': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_dagger_384.pth',
    'crossvit_9_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_9_224.pth',
    'crossvit_9_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_9_dagger_224.pth',
    'crossvit_base_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_base_224.pth',
    'crossvit_small_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_small_224.pth',
    'crossvit_tiny_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_tiny_224.pth',
}


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=True):
        super().__init__()
        img_size = to_2tuple(img_size)  # 生成224*224
        patch_size = to_2tuple(patch_size)  # 生成16*16
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # 像素面积的缩小倍数
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:  # conv2d修改 sequentialcell修改
                self.proj = nn.SequentialCell(
                    nn.Conv2d(in_chans, embed_dim // 4, pad_mode='pad', kernel_size=7, stride=4, padding=3),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, pad_mode='pad', kernel_size=3, stride=3, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 2, embed_dim, pad_mode='pad', kernel_size=3, stride=1, padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.SequentialCell(
                    nn.Conv2d(in_chans, embed_dim // 4, pad_mode='pad', kernel_size=7, stride=4, padding=3),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, pad_mode='pad', kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 2, embed_dim, pad_mode='pad', kernel_size=3, stride=2, padding=1),
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, pad_mode='valid')

    def construct(self, x):
        B, C, H, W = x.shape  # x的四维
        # FIXME look at relaxing size constraints

        # assert H == self.img_size[0] and W == self.img_size[1], \
        # f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        print(x)
        B, C, H, W = x.shape
        # print(B,C,H,W)
        x = x.reshape(B, C, H * W)
        x = ops.transpose(x, (0, 2, 1))
        return x  # 修改，输出结果的shape大小相同，但是数据不相同


# 这个函数的问题，flatten的问题，无法在特定维度进行展开,已经解决
# net1 = PatchEmbed()
# x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
# out = net1(x)


class CrossAttention(nn.Cell):  # 交叉注意力的算法
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 数组是否可以整除？
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Dense(dim, dim, has_bias=qkv_bias)  # 修改
        self.wk = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.wv = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1.0 - attn_drop)  # 修改，反人类化，pytorch表示丢弃，ms表示保留的概率
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(1.0 - proj_drop)

    def construct(self, x):  # transformer的qkv进行计算，修改construct

        B, N, C = x.shape  # 3,3,16
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads)
        q = ops.transpose(q, (0, 2, 1, 3))  # B1C -> B1H(C/H) -> BH1(C/H) 3 8 1 2

        # torch的transpose只能转换一个维度，而permute可以转换好几个维度，修改
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads)
        k = ops.transpose(k, (0, 2, 1, 3))  # BNC -> BNH(C/H) -> BHN(C/H) 3832

        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads)
        v = ops.transpose(v, (0, 2, 1, 3))  # BNC -> BNH(C/H) -> BHN(C/H)3832

        # 修改@的结果运算以及，transpose的转换
        batchmatual = ops.BatchMatMul(transpose_b=True)
        attn = batchmatual(q, k) * self.scale
        print(attn.shape)

        # attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N

        softmax = nn.Softmax()
        attn = softmax(attn)
        print(attn.shape)
        # attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        print(attn.shape)

        batchmatual2 = ops.BatchMatMul()
        x = batchmatual2(attn, v)
        print(attn.shape)
        print(v.shape)
        print(x.shape)

        x = ops.transpose(x, (0, 2, 1, 3))
        x = x.reshape(B, 1, C)
        print(x.shape)

        # x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        print(x.shape)
        print(2222)

        return x


# net2 = CrossAttention(dim=16)
# x = ms.Tensor(np.ones((3, 3, 16)), ms.float32)
# x = net2(x).shape
# print(x)


class CrossAttentionBlock(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()

        self.norm1 = norm_layer(dim, begin_norm_axis=2, begin_params_axis=2)
        self.attn = CrossAttention(
            dim[0], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim, begin_norm_axis=2, begin_params_axis=2)  # normlayer无法输入单个数字
            mlp_hidden_dim = int(16 * mlp_ratio)
            # 修改mlp
            self.mlp = Mlp(in_features=dim[0], hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x):
        print(x)
        # 这个0:1表示只选择这个维度的两行
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# x = ms.Tensor(np.ones((3, 3, 16)), ms.float32)
# print(type(x))
# x = x[:, 0:1, ...]
# print(x.shape)
# net3 = CrossAttentionBlock(dim=(16,), num_heads=4)
# x = net3(x).shape
# print(x)


class MultiScaleBlock(nn.Cell):

    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        #print(dim)  # 输出正确
        num_branches = len(dim)
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base
        for d in range(num_branches):
            for i in range(depth[d]):
                if d == 0 and i == 0:
                    self.blocks=nn.SequentialCell([Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                               drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer)])
                else:
                    self.blocks.append((nn.SequentialCell(  # 修改严重仔细查看,还没有进行，输入block，但是要求list和字典，加上括号，转换成list
                        [Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                               drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer)]))[0])

        if len(self.blocks) == 0:
            self.blocks = None

        #print(self.blocks)
        for d in range(num_branches):
            if dim[d] == dim[(d + 1) % num_branches] and False:
                tmp = [Identity()]
                self.projs=nn.SequentialCell(*tmp)
            else:  # 修改，因为norm_layer的问题,转换成list形式，因为norm_layer只接受list的形式
                tmp = [norm_layer([dim[d], ]), act_layer(), nn.Dense(dim[d], dim[(d + 1) % num_branches])]
                #print(tmp)  # 设置只有一个数据
                self.projs = nn.SequentialCell(*tmp)
            #self.projs.append(nn.SequentialCell(*tmp))


        for d in range(num_branches):
            d_ = (d + 1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                # 统一规定，输入的dim均为数组形式，应用于attention的操作
                # 是不经过这里的
                self.fusion = nn.SequentialCell([
                    CrossAttentionBlock(dim=[dim[d_], ], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                        has_mlp=False)])

            else:
                self.fusion = nn.SequentialCell([
                    CrossAttentionBlock(dim=[dim[d_], ], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                        has_mlp=False)])
                for _ in range(depth[-1]-1):
                    # 经过这个分支
                    self.fusion.append((nn.SequentialCell([
                        CrossAttentionBlock(dim=[dim[d_], ], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                                             qk_scale=qk_scale,
                                             drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1],
                                             norm_layer=norm_layer,
                                             has_mlp=False)]))[0])


        #self.revert_projs = nn.CellList()
        for d in range(num_branches):
            if dim[(d + 1) % num_branches] == dim[d] and False:
                # print(555555)
                tmp = [Identity()]
                self.revert_projs=nn.SequentialCell(*tmp)
            else:
                # print(66666)  # 经过这个分支
                tmp = [norm_layer([dim[(d + 1) % num_branches]]), act_layer(),
                       nn.Dense(dim[(d + 1) % num_branches], dim[d])]
                self.revert_projs = nn.SequentialCell(*tmp)
            #self.revert_projs.append(nn.SequentialCell(*tmp))
            # print(77777)  # 经过这个分支，也已经添加结束了，在进行construct的过程中进行
            #x = ms.Tensor(np.ones((3, 3, 3, 16)), ms.float32)
            # for  block in zip( self.blocks):
            #
            #     print(block)

    def construct(self, x):
        #print(x)
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]
        # only take the cls token out
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        # cross attention
        outs = []
        for i in range(self.num_branches):
            # 修改torch.cat为ops.concat主要表现在数据的精度的自动转换上面
            tmp = ops.Concat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), 1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = ops.Concat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), 1)
            outs.append(tmp)
        return outs


# 网络内容这部分推导失败，进行下一步的推理，vit的推理推理
#
net3 = MultiScaleBlock(dim=(16,), patches=2, depth=[1], num_heads=[1], mlp_ratio=[1], drop_path=[0.5, ])
# for name, param in net3.parameters_and_names():
#     print(name, param)
#print(net3.parameters_dict())
x = ms.Tensor(np.ones((3, 3, 3, 16)), ms.float32)
x = net3(x).shape
# print(net3)

#
# def _compute_num_patches(img_size, patches):
#     return [i // p * i // p for i, p in zip(img_size, patches)]


# class VisionTransformer(nn.Cell):
#     """ Vision Transformer with support for patch or hybrid CNN input stage
#     """
#
#     def __init__(self, img_size=(224, 224), patch_size=(8, 16), in_chans=3, num_classes=1000, embed_dim=(192, 384),
#                  depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
#                  num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None, drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, multi_conv=False):
#         super().__init__()
#
#         self.num_classes = num_classes
#         if not isinstance(img_size, list):
#             img_size = to_2tuple(img_size)
#         self.img_size = img_size
#
#         num_patches = _compute_num_patches(img_size, patch_size)
#         self.num_branches = len(patch_size)
#
#         self.patch_embed = nn.CellList()
#         if hybrid_backbone is None:  #修改，是有差异的，一个是列表，另一个是元组的形式进行存储
#             self.pos_embed = ms.ParameterTuple(
#                 [mindspore.Parameter(mindspore.ops.Zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
#             for im_s, p, d in zip(img_size, patch_size, embed_dim):
#                 self.patch_embed.append(
#                     PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_chans, embed_dim=d, multi_conv=multi_conv))
#         else:
#             self.pos_embed = ms.ParameterTuple()  #修改，是有差异的，一个是列表，另一个是元组的形式进行存储
#             from .t2t import T2T, get_sinusoid_encoding
#             tokens_type = 'transformer' if hybrid_backbone == 't2t' else 'performer'
#             for idx, (im_s, p, d) in enumerate(zip(img_size, patch_size, embed_dim)):
#                 self.patch_embed.append(T2T(im_s, tokens_type=tokens_type, patch_size=p, embed_dim=d))
#                 self.pos_embed.append(
#                     mindspore.Parameter(data=get_sinusoid_encoding(n_position=1 + num_patches[idx], d_hid=embed_dim[idx]),
#                                  requires_grad=False))
#
#             del self.pos_embed
#             self.pos_embed = nn.ParameterList(
#                 [mindspore.Parameter(mindspore.ops.Zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
#
#         self.cls_token = nn.ParameterList(
#             [nn.Parameter(mindspore.ops.Zeros(1, 1, embed_dim[i])) for i in range(self.num_branches)])
#         self.pos_drop = nn.Dropout(p=drop_rate)
#
#         total_depth = sum([sum(x[-2:]) for x in depth])
#         dpr = [x.item() for x in mindspore.ops.LinSpace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule
#         dpr_ptr = 0
#         self.blocks = nn.ModuleList()
#         for idx, block_cfg in enumerate(depth):
#             curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
#             dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
#             blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio,
#                                   qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
#                                   drop_path=dpr_,
#                                   norm_layer=norm_layer)
#             dpr_ptr += curr_depth
#             self.blocks.append(blk)
#
#         self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])
#         self.head = nn.ModuleList([nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity() for i in
#                                    range(self.num_branches)])
#
#         for i in range(self.num_branches):
#             if self.pos_embed[i].requires_grad:
#                 trunc_normal_(self.pos_embed[i], std=.02)
#             trunc_normal_(self.cls_token[i], std=.02)
#
#         self.apply(self._init_weights)
#
#
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         out = {'cls_token'}
#         if self.pos_embed[0].requires_grad:
#             out.add('pos_embed')
#         return out
#
#     def get_classifier(self):
#         return self.head
#
#     def reset_classifier(self, num_classes, global_pool=''):
#         self.num_classes = num_classes
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
#
#     def forward_features(self, x):
#         B, C, H, W = x.shape
#         xs = []
#         for i in range(self.num_branches):
#             x_ = torch.nn.functional.interpolate(x, size=(self.img_size[i], self.img_size[i]), mode='bicubic') if H != self.img_size[i] else x
#             tmp = self.patch_embed[i](x_)
#             cls_tokens = self.cls_token[i].expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#             tmp = torch.cat((cls_tokens, tmp), dim=1)
#             tmp = tmp + self.pos_embed[i]
#             tmp = self.pos_drop(tmp)
#             xs.append(tmp)
#
#         for blk in self.blocks:
#             xs = blk(xs)
#
#         # NOTE: was before branch token section, move to here to assure all branch token are before layer norm
#         xs = [self.norm[i](x) for i, x in enumerate(xs)]
#         out = [x[:, 0] for x in xs]
#
#         return out
#
#     def forward(self, x):
#         xs = self.forward_features(x)
#         ce_logits = [self.head[i](x) for i, x in enumerate(xs)]
#         ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)
#         return ce_logits
