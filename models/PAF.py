'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GBC import BottConv

class PAF(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 after_relu: bool = False,
                 mid_norm: nn.Module = nn.BatchNorm2d,
                 in_norm: nn.Module = nn.BatchNorm2d):
        super().__init__()
        self.after_relu = after_relu

        self.feature_transform = nn.Sequential(
            BottConv(in_channels, mid_channels, mid_channels=16, kernel_size=1),
            mid_norm(mid_channels)
        )

        self.channel_adapter = nn.Sequential(
            BottConv(mid_channels, in_channels, mid_channels=16, kernel_size=1),
            in_norm(in_channels)
        )

        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, base_feat: torch.Tensor, guidance_feat: torch.Tensor) -> torch.Tensor:
        base_shape = base_feat.size()  # 记录 base_feat 的形状，后续用于调整其他特征的尺寸。

        # 如果 after_relu 为 True，则对 base_feat 和 guidance_feat 应用 ReLU 激活函数，引入非线性。
        if self.after_relu:
            base_feat = self.relu(base_feat)
            guidance_feat = self.relu(guidance_feat)

        # 将 guidance_feat 输入到 feature_transform 模块中，通道数从 in_channels 转换为 mid_channels。
        guidance_query = self.feature_transform(guidance_feat)
        # 将 base_feat 输入到 feature_transform 模块中，通道数同样从 in_channels 转换为 mid_channels。
        base_key = self.feature_transform(base_feat)
        # 使用 F.interpolate 函数将 guidance_query 的尺寸调整为与 base_feat 相同，确保后续操作的特征尺寸一致。
        guidance_query = F.interpolate(guidance_query, size=[base_shape[2], base_shape[3]], mode='bilinear', align_corners=False)
        similarity_map = torch.sigmoid(self.channel_adapter(base_key * guidance_query))
        # base_key * guidance_query：对 base_key 和调整尺寸后的 guidance_query 进行逐元素相乘，得到一个中间特征。
        # self.channel_adapter：将中间特征输入到 channel_adapter 模块中，通道数从 mid_channels 转换回 in_channels。
        # torch.sigmoid：对 channel_adapter 的输出应用 Sigmoid 函数，将其值映射到 [0, 1] 区间，得到相似度图 similarity_map。
        # 这个相似度图表示了 base_feat 和 guidance_feat 在每个通道和空间位置上的相似程度。
        resized_guidance = F.interpolate(guidance_feat, size=[base_shape[2], base_shape[3]], mode='bilinear', align_corners=False)
        # 使用 F.interpolate 函数将 guidance_feat 的尺寸调整为与 base_feat 相同，以便进行特征融合。
        fused_feature = (1 - similarity_map) * base_feat + similarity_map * resized_guidance
        # 根据相似度图 similarity_map 对 base_feat 和 resized_guidance 进行加权融合。
        # 当 similarity_map 接近 1 时，融合后的特征更倾向于 resized_guidance；
        # 当 similarity_map 接近 0 时，融合后的特征更倾向于 base_feat。

        return fused_feature