'''
Author: Chenhongyi Yang
Reference: https://github.com/OliverRensu/Shunted-Transformer
'''
import torch
from mmcv.cnn.bricks.transformer import AdaptivePadding
from mmcv.cnn import (build_conv_layer, build_norm_layer)
from mmcv.runner.base_module import BaseModule
from mmcv.utils import to_2tuple
from torch import nn

class ConvPatchEmbed(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,  # 嵌入向量的维度，即将每个图像小块映射到的向量维度，默认值为 768。
                 num_convs=0,  # 在投影层之前额外添加的卷积层数量，默认值为 0。
                 conv_type='Conv2d',  # 用于生成 patch 嵌入的卷积层类型，默认值为 'Conv2d'。
                 patch_size=16,
                 stride=16,  # 卷积层的步长，默认值为 16。
                 padding='corner',
                 dilation=1,  # 卷积层的扩张率，默认值为 1。
                 bias=True,
                 norm_cfg=None,  # 归一化层的配置字典，默认值为 None。
                 input_size=None,
                 init_cfg=None):
        super(ConvPatchEmbed, self).__init__(init_cfg=init_cfg)

        assert patch_size % 2 == 0  # 确保 patch_size 是偶数。

        self.embed_dims = embed_dims
        # 处理 stride 参数，如果 stride 为 None，则将其设置为 patch_size // 2。
        if stride is None:
            stride = patch_size // 2
        else:
            stride = stride // 2

        # 使用一个 7x7 的卷积层、组归一化层和 ReLU 激活函数构建 Stem 模块，将输入图像的通道数转换为 64。
        self.stem = torch.nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7,7), stride=(2,2), padding=3, bias=False),
            # nn.BatchNorm2d(64),
            nn.GroupNorm(num_channels=64, num_groups=4),
            nn.ReLU(True))

        # 如果 num_convs 大于 0，则添加指定数量的 3x3 卷积层、组归一化层和 ReLU 激活函数。
        if num_convs > 0:
            convs = []
            for _ in range(num_convs):
                convs.append(torch.nn.Conv2d(64, 64, (3,3), (1,1), padding=1, bias=False))
                convs.append(torch.nn.GroupNorm(num_channels=64, num_groups=4))
                convs.append(torch.nn.ReLU(True))
            self.convs = torch.nn.Sequential(*convs)
        else:
            self.convs = None

        # 将 kernel_size、stride 和 dilation 转换为元组。
        kernel_size = to_2tuple(patch_size//2)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        # 如果 padding 是字符串，则使用 AdaptivePadding 进行自适应填充，并将卷积层的填充设置为 0。
        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adaptive_padding = None
        padding = to_2tuple(padding)

        # self.projection = build_conv_layer(
        #     cfg=dict(
        #     type=conv_type,
        #     in_channels=64,
        #     out_channels=embed_dims,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding,
        #     dilation=dilation,
        #     bias=bias)

        # 构建投影层 使用一个卷积层将通道数从 64 转换为 embed_dims。
        self.projection = nn.Conv2d(
            in_channels=64,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )

        # 如果 norm_cfg 不为 None，则构建归一化层。
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        # 如果 input_size 不为 None，则计算输出的空间大小。
        # input_size 是整数，会将其转换为 (input_size, input_size) 的元组形式
        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            _input_size = (input_size[0] // 2, input_size[1] // 2)
            # 在 stem 模块中使用了步长为 2 的卷积层，所以这里将输入大小的高和宽都除以 2，得到经过 stem 模块后特征图的大致大小
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(_input_size)
                input_h, input_w = _input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                _input_size = (input_h, input_w)
                # 如果使用了自适应填充（self.adaptive_padding 不为 None），
                # 则调用 self.adaptive_padding.get_pad_shape(_input_size) 方法计算需要填充的高度和宽度 pad_h 和 pad_w。
                # 将填充后的高度和宽度加到 _input_size 上，更新 _input_size 为填充后的特征图大小。

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            # 卷积层输出大小的计算公式
            h_out = (_input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (_input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

            - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (out_h, out_w).
        """
        x = self.stem(x)
        if self.convs is not None:
            x = self.convs(x)

        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])  # 张量(B, C, H, W) 获取经过投影层卷积后特征图x的空间尺寸，即高度和宽度
        x = x.flatten(2).transpose(1, 2)  # 从索引为 2 的维度（即高度维度）开始展平。将高度和宽度维度合并为一个维度，交换后的张量形状变为(B, H * W, C)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size

