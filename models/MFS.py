import torch.nn as nn
import torch
from models.GBC import BottConv
from models.DySample import DySample
import torch.nn.functional as F

class CrackAwareFFM(nn.Module):
    """针对裂缝检测优化的特征融合模块"""

    def __init__(self, in_channels, out_channels, reduction=8):
        super().__init__()
        self.reduction = reduction
        mid_channels = in_channels // 2

        # 裂缝感知的通道注意力
        self.crack_channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 裂缝感知的空间注意力
        self.crack_spatial_att = nn.Sequential(
            BottConv(in_channels, in_channels // 8, max(1, in_channels // 8), 3, padding=1),
            nn.ReLU(),
            BottConv(in_channels // 8, 1, max(1, (in_channels // 8) // 8), 3, padding=1),
            nn.Sigmoid()
        )

        # 边缘保持分支
        self.edge_preserve = nn.Sequential(
            BottConv(in_channels, in_channels // 8, max(1, in_channels // 8), 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

        # 信息整合 - 使用GroupNorm替代BatchNorm
        self.fusion = nn.Sequential(
            BottConv(in_channels, mid_channels, max(1, in_channels // 8), kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.GroupNorm(8, mid_channels),  # 使用GroupNorm，组数为8
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # 裂缝感知通道注意力
        ca = self.crack_channel_att(x)
        x = x * ca
        # 裂缝感知空间注意力
        sa = self.crack_spatial_att(x)
        x = x * sa
        # 边缘保持
        edge_weight = self.edge_preserve(x)
        x = x * edge_weight
        # 特征融合
        x = self.fusion(x)
        return x


class MFS(nn.Module):
    def __init__(self, embedding_dim):
        super(MFS, self).__init__()

        self.embedding_dim = embedding_dim
        self.fuse_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embedding_dim, 1),
                nn.Dropout2d(0.1)
            ) for c in [32, 64, 128, 128]
        ])


        # 上采样到原图尺寸
        self.upsample_layers = nn.ModuleList([
            DySample(embedding_dim, scale=2),
            DySample(embedding_dim, scale=4),
            DySample(embedding_dim, scale=8),
            DySample(embedding_dim, scale=8),
        ])

        # self.fuse = nn.Sequential(
        #     nn.Conv2d(embedding_dim * 4, embedding_dim, 1, bias=False),
        #     nn.GroupNorm(8, embedding_dim),
        #     # nn.ReLU(inplace=True)
        #     nn.SiLU()
        # )

        self.fuse = CrackAwareFFM(embedding_dim * 4, embedding_dim)


        # 预测头
        self.pre = nn.Sequential(
            nn.Dropout(p=0.1),
            BottConv(embedding_dim, 1, 1, kernel_size=1),
            nn.Conv2d(1, 1, kernel_size=1)
        )

    def forward(self, inputs):
        fused_feats = []
        for i in range(4):
            fused = self.fuse_convs[i](inputs[i])
            fused = self.upsample_layers[i](fused)
            fused_feats.append(fused)

        out_c = torch.cat(fused_feats, dim=1)

        out = self.fuse(out_c)

        out = self.pre(out)

        return out
