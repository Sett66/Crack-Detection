from typing import Sequence
import copy
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import ModuleList
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import resize_pos_embed, to_2tuple
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.SAVSS_dev.models.modules.patch_embed import ConvPatchEmbed
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_Layer
from models.GBC import BottConv

@BACKBONES.register_module()
class SAVSS(BaseBackbone):
    arch_zoo = {
        'Crack': {
            'patch_size': 8,
            'embed_dims': 128,
            'num_layers': 4,
            'num_convs_patch_embed': 2,
            'layers_with_dwconv': [],
            'layer_cfgs': {
                'use_rms_norm': False,
                'mamba_cfg': {
                    'd_state': 16,
                    'expand': 2,
                    'conv_size': 7,
                    'dt_init': "random",
                    'conv_bias': True,
                    'bias': True,
                    'default_hw_shape': (512 // 8, 512 // 8)
                }
            }
        }
    }

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 arch=None,
                 patch_size=16,  # 图像分块的大小，默认为 16
                 embed_dims=192,  # 嵌入维度，即每个图像块被映射到的向量维度，默认为 192
                 num_layers=20,  # 模型中 SAVSS_Layer 的数量，默认为 20
                 num_convs_patch_embed=1,  # PatchEmbed 模块中卷积层的数量，默认为 1。
                 with_pos_embed=True,  # 是否使用位置嵌入，默认为 True
                 out_indices=-1,  # 指定输出特征的层索引，默认为 -1 表示最后一层。
                 drop_rate=0.,  # Dropout 层的丢弃率，默认为 0。
                 drop_path_rate=0.,  # DropPath 层的丢弃率，默认为 0。
                 norm_cfg=dict(type='LN', eps=1e-6),  # 归一化层的配置，默认为使用 Layer Normalization（LN）。
                 final_norm=True,  # 是否在最后使用归一化层，默认为 True。
                 interpolate_mode='bicubic',  # 位置嵌入插值的模式，默认为 'bicubic'。
                 layer_cfgs=dict(),  # SAVSS_Layer 的配置，默认为空字典
                 layers_with_dwconv=[],  # layers_with_dwconv：需要使用深度可分离卷积的层的索引列表。
                 init_cfg=None,  # 初始化配置
                 test_cfg=dict(),  # 测试配置
                 convert_syncbn=False,  # 是否将 BN 转换为 SyncBN。
                 freeze_patch_embed=False,  # 是否冻结 PatchEmbed 模块的参数。
                 **kwargs):
        super(SAVSS, self).__init__(init_cfg)

        self.test_cfg = test_cfg
        self.img_size = to_2tuple(img_size)  # 将 img_size 转换为二元组形式，确保输入大小是 (height, width) 的格式
        self.convert_syncbn = convert_syncbn
        self.arch = arch

        if self.arch is None:
            self.embed_dims = embed_dims
            self.num_layers = num_layers
            self.patch_size = patch_size
            self.num_convs_patch_embed = num_convs_patch_embed
            self.layers_with_dwconv = layers_with_dwconv
            _layer_cfgs = layer_cfgs
        else:
            assert self.arch in self.arch_zoo.keys()
            self.embed_dims = self.arch_zoo[self.arch]['embed_dims']
            self.num_layers = self.arch_zoo[self.arch]['num_layers']
            self.patch_size = self.arch_zoo[self.arch]['patch_size']
            self.num_convs_patch_embed = self.arch_zoo[self.arch]['num_convs_patch_embed']
            self.layers_with_dwconv = self.arch_zoo[self.arch]['layers_with_dwconv']
            _layer_cfgs = self.arch_zoo[self.arch]['layer_cfgs']

        self.with_pos_embed = with_pos_embed
        self.interpolate_mode = interpolate_mode
        self.freeze_patch_embed = freeze_patch_embed
        _drop_path_rate = drop_path_rate

        self.patch_embed = ConvPatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            num_convs=self.num_convs_patch_embed,
            patch_size=self.patch_size,
            stride=self.patch_size
        )
        self.patch_resolution = self.patch_embed.init_out_size
        # 获取 PatchEmbed 模块的输出分辨率 patch_resolution，并计算图像块的总数 num_patches。
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        # 如果 with_pos_embed 为 True，则创建一个可学习的位置嵌入参数 pos_embed，并使用截断正态分布进行初始化。
        if with_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))
            trunc_normal_(self.pos_embed, std=0.02)
        # 初始化一个 Dropout 层 drop_after_pos，用于在位置嵌入后进行随机丢弃。
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # 将 out_indices 转换为列表形式。
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        # 确保 out_indices 是一个序列。
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            # 处理负索引，将其转换为正索引。
            # 检查索引的有效性，确保在合法范围内。
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # 使用 np.linspace 生成一个从 0 到 _drop_path_rate 的等差数列，作为每层的 DropPath 丢弃率。
        dpr = np.linspace(0, _drop_path_rate, self.num_layers)
        self.drop_path_rate = _drop_path_rate

        self.layer_cfgs = _layer_cfgs
        self.layers = ModuleList()  # 初始化一个 ModuleList 用于存储 SAVSS_Layer。
        # 如果 layer_cfgs 是字典，则复制 num_layers 份。
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [copy.deepcopy(_layer_cfgs) for _ in range(self.num_layers)]

        # 为每层的配置更新嵌入维度和 DropPath 丢弃率，并根据 layers_with_dwconv 决定是否使用深度可分离卷积。
        for i in range(self.num_layers):
            _layer_cfg_i = layer_cfgs[i]
            _layer_cfg_i.update({
                "embed_dims": self.embed_dims,
                "drop_path_rate": dpr[i]
            })
            if i in self.layers_with_dwconv:
                _layer_cfg_i.update({"with_dwconv": True})
            else:
                _layer_cfg_i.update({"with_dwconv": False})
            # 实例化 SAVSS_Layer 并添加到 layers 中。
            self.layers.append(
                SAVSS_Layer(**_layer_cfg_i)
            )

        self.final_norm = final_norm
        # 如果 final_norm 为 True，则初始化最后一层的归一化层。
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        # 为每个需要输出特征的层（除最后一层）初始化归一化层，如果 norm_cfg 为 None，则使用 nn.Identity 作为占位符。
        for i in out_indices:
            if i != self.num_layers - 1:
                if norm_cfg is not None:
                    norm_layer = build_norm_layer(norm_cfg, self.embed_dims)[1]
                else:
                    norm_layer = nn.Identity()
                self.add_module(f'norm_layer{i}', norm_layer)
        self.conv128to128 = BottConv(in_channels=128, out_channels=128, mid_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv128to64 = BottConv(in_channels=128, out_channels=64, mid_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv128to32 = BottConv(in_channels=128, out_channels=32, mid_channels=8, kernel_size=1, stride=1, padding=0)
        self.conv128to16 = BottConv(in_channels=128, out_channels=16, mid_channels=4, kernel_size=1, stride=1, padding=0)
        self.gn128 = nn.GroupNorm(num_channels=128, num_groups=8)
        self.gn64 = nn.GroupNorm(num_channels=64, num_groups=4)
        self.gn32 = nn.GroupNorm(num_channels=32, num_groups=2)
        self.gn16 = nn.GroupNorm(num_channels=16, num_groups=2)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)
    # 获取对象 self 中名为 self.norm1_name 的属性或方法。通过这个属性方法，可以方便地获取最后一层归一化层。

    def init_weights(self):
        super(SAVSS, self).init_weights()
        # 检查 init_cfg 是否为字典类型，并且其 type 是否为 'Pretrained'。如果不是使用预训练模型，那么进入下一步。
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # 检查是否使用了位置嵌入。如果使用了，就使用 trunc_normal_ 函数对位置嵌入参数 self.pos_embed 进行截断正态分布初始化，标准差为 0.02。
            if self.with_pos_embed:
                trunc_normal_(self.pos_embed, std=0.02)
        # 根据配置决定是否冻结 patch_embed 模块的参数。
        self.set_freeze_patch_embed()

    def set_freeze_patch_embed(self):
        if self.freeze_patch_embed:
            self.patch_embed.eval()
            # 将 patch_embed 模块设置为评估模式，这通常会影响一些在训练和评估阶段行为不同的层，如 Dropout 和 BatchNorm。
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            # 将每个参数的 requires_grad 属性设置为 False，表示这些参数不需要计算梯度，从而在训练过程中不会被更新。

    def forward(self, x):
        x, patch_resolution = self.patch_embed(x)
        if self.with_pos_embed:
            pos_embed = resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=0
            )
            x = x + pos_embed
        x = self.drop_after_pos(x)

        # 初始化两个列表，分别用于存储中间特征和最终输出特征。
        outs_before = []
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, hw_shape=patch_resolution)
            # 如果是最后一层且配置了最终归一化层，将特征 x 传入 norm1 进行归一化。
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            # 如果当前层的索引在 out_indices 列表中，说明该层的输出需要被记录。
            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x.reshape(B, *patch_resolution, C) # 将特征 x 重新调整形状为 (B, H, W, C) 的形式。
                # 如果不是最后一层，获取对应的归一化层并对 patch_token 进行归一化。
                if i != self.num_layers - 1:
                    norm_layer = getattr(self, f'norm_layer{i}')
                    patch_token = norm_layer(patch_token)
                patch_token = patch_token.permute(0, 3, 1, 2)
                # patch_token 的维度顺序调整为 (B, C, H, W)。
                outs_before.append(patch_token)
                # 将处理后的 patch_token 加入 outs_before 列表。

                if i == self.out_indices[0]:
                    patch_token_mid = self.gn32(self.conv128to32(patch_token))
                    patch_token_mid = nn.Upsample(size=(128, 128), mode="bilinear")(patch_token_mid)
                    outs.append(patch_token_mid)
                elif i == self.out_indices[1]:
                    patch_token_mid = self.gn64(self.conv128to64(patch_token))
                    patch_token_mid = nn.Upsample(size=(64, 64), mode="bilinear")(patch_token_mid)
                    outs.append(patch_token_mid)
                elif i == self.out_indices[2]:
                    patch_token_mid = self.gn128(self.conv128to128(patch_token))
                    patch_token_mid = nn.Upsample(size=(32, 32), mode="bilinear")(patch_token_mid)
                    outs.append(patch_token_mid)
                elif i == self.out_indices[3]:
                    patch_token_mid = self.gn128(self.conv128to128(patch_token))
                    patch_token_mid = nn.Upsample(size=(32, 32), mode="bilinear")(patch_token_mid)
                    outs.append(patch_token_mid)
                else:
                    continue

        return outs