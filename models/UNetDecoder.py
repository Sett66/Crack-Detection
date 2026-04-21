import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GBC import BottConv

class ConvGN_SiLU(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, pad=1, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, pad, bias=False)
        g = min(groups, out_ch)
        self.norm = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DoubleConv(nn.Module):
    """(ConvGN_SiLU -> ConvGN_SiLU) 组成的块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvGN_SiLU(in_channels, out_channels, ks=3, pad=1),
            ConvGN_SiLU(out_channels, out_channels, ks=3, pad=1)
        )

    def forward(self, x):
        return self.double_conv(x)


# 假设您的 ConvGN_SiLU, DoubleConv 和 UNetUpBlock 模块定义正确，
# 并且 UNetUpBlock 能够处理 skip_channels=0 的情况。

# 重新定义 UNetUpBlock 以确保兼容性
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 1. 上采样部分：使用转置卷积（ConvTranspose2d）
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # 2. 卷积部分
        fused_channels = (in_channels // 2) + skip_channels
        self.conv = DoubleConv(fused_channels, out_channels)

    def forward(self, x_deep, x_skip=None):
        x_deep = self.up(x_deep)

        if x_skip is not None:
            # 填充，确保尺寸匹配
            diffY = x_skip.size()[2] - x_deep.size()[2]
            diffX = x_skip.size()[3] - x_deep.size()[3]

            x_deep = F.pad(x_deep, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
            # 拼接
            x = torch.cat([x_skip, x_deep], dim=1)
        else:
            # 无跳跃连接
            x = x_deep

        return self.conv(x)


class SAVSS_UNet_Decoder(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        # 假设 Backbone 输出通道为 [16, 32, 64, 128, 256]
        C0, C1, C2, C3, C4 = 16, 32, 64, 128, 256

        # UP1: 16x16 -> 32x32
        self.up1 = UNetUpBlock(in_channels=C4, skip_channels=C3, out_channels=C3)
        # Output: 256 @ 32x32

        # UP2: 32x32 -> 64x64
        self.up2 = UNetUpBlock(in_channels=C3, skip_channels=C2, out_channels=C2)
        # Output: 128 @ 64x64

        # UP3: 64x64 -> 128x128
        self.up3 = UNetUpBlock(in_channels=C2, skip_channels=C1, out_channels=C1)
        # Output: 64 @ 128x128

        # UP4: 128x128 -> 256x256 (使用最高分辨率的跳跃连接 C0)
        self.up4 = UNetUpBlock(in_channels=C1, skip_channels=C0, out_channels=C0)
        # Output: 32 @ 256x256

        # self.CrackFFM = CrackAwareFFM(C0, C0)
        # self.pre = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     BottConv(C0, 1, 1, kernel_size=1),
        #     nn.Conv2d(1, 1, kernel_size=1)
        # )


        self.out_conv = nn.Conv2d(C0, n_classes, kernel_size=1)

    def forward(self, x_outs):
        # x_outs = [x_s0@256, x_s1@128, x_s2@64, x_s3@32, x_s4@16]
        x_s0, x_s1, x_s2, x_s3, x_s4 = x_outs

        # Bottleneck
        x = x_s4  # 512 @ 16x16

        # D1: 16x16 -> 32x32
        x = self.up1(x, x_s3)  # 256 @ 32x32

        # D2: 32x32 -> 64x64
        x = self.up2(x, x_s2)  # 128 @ 64x64

        # D3: 64x64 -> 128x128
        x = self.up3(x, x_s1)  # 64 @ 128x128

        # D4: 128x128 -> 256x256
        x = self.up4(x, x_s0)  # 32 @ 256x256

        logits = self.out_conv(x)
        # logits = self.pre(self.CrackFFM(x))  # 1 @ 256x256

        return logits