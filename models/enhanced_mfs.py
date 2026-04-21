"""
增强版MFS模块 - 针对裂缝分割任务优化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalFeatureFusion(nn.Module):
    """层次化特征融合模块，针对裂缝检测优化"""
    
    def __init__(self, channels_list=[16, 32, 64, 128], embedding_dim=8):
        super().__init__()
        self.channels_list = channels_list
        self.embedding_dim = embedding_dim
        
        # 自适应通道调整
        self.channel_adjusters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embedding_dim, 1),
                nn.GroupNorm(8, embedding_dim),
                nn.SiLU()
            ) for c in channels_list
        ])
        
        # 多尺度特征交互
        self.scale_interactions = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1, groups=embedding_dim)
                for _ in range(len(channels_list))
            ]) for _ in range(len(channels_list))
        ])
        
        # 裂缝感知的尺度权重
        self.scale_weights = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embedding_dim, embedding_dim // 4, 1),
                nn.ReLU(),
                nn.Conv2d(embedding_dim // 4, 1, 1),
                nn.Sigmoid()
            ) for _ in range(len(channels_list))
        ])
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Conv2d(embedding_dim * len(channels_list), embedding_dim, 1),
            nn.GroupNorm(8, embedding_dim),
            nn.SiLU()
        )
    
    def forward(self, features):
        # 调整通道数
        adjusted_features = []
        for i, feat in enumerate(features):
            adjusted = self.channel_adjusters[i](feat)
            adjusted_features.append(adjusted)
        
        # 多尺度交互
        enhanced_features = []
        for i, feat in enumerate(adjusted_features):
            # 与其他尺度特征交互
            interactions = []
            for j, other_feat in enumerate(adjusted_features):
                if i != j:
                    # 调整到相同尺寸
                    if feat.shape[-2:] != other_feat.shape[-2:]:
                        other_feat = F.interpolate(other_feat, size=feat.shape[-2:], mode='bilinear', align_corners=False)
                    interaction = self.scale_interactions[i][j](other_feat)
                    interactions.append(interaction)
            
            # 融合交互特征
            if interactions:
                interaction_feat = torch.stack(interactions, dim=0).mean(dim=0)
                enhanced_feat = feat + interaction_feat
            else:
                enhanced_feat = feat
            
            # 应用尺度权重
            weight = self.scale_weights[i](enhanced_feat)
            enhanced_feat = enhanced_feat * weight
            enhanced_features.append(enhanced_feat)
        
        # 上采样到统一尺寸
        target_size = enhanced_features[0].shape[-2:]  # 使用最大尺寸
        upsampled_features = []
        for feat in enhanced_features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(feat)
        
        # 最终融合
        fused = torch.cat(upsampled_features, dim=1)
        output = self.final_fusion(fused)
        
        return output


class CrackAwareAttention(nn.Module):
    """裂缝感知的注意力机制"""
    
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        
        # 方向感知的空间注意力 - 针对裂缝的线性特征
        self.directional_conv = nn.ModuleList([
            # 水平方向
            nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels),
            # 垂直方向  
            nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels),
            # 对角线方向
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels),
        ])
        
        # 裂缝连续性建模
        self.continuity_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels),
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels),
        )
        
        # 注意力权重生成
        self.attention_gen = nn.Sequential(
            nn.Conv2d(channels * 4, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 方向感知特征提取
        directional_feats = []
        for conv in self.directional_conv:
            feat = conv(x)
            directional_feats.append(feat)
        
        # 连续性建模
        continuity_feat = self.continuity_conv(x)
        directional_feats.append(continuity_feat)
        
        # 融合方向特征
        directional_concat = torch.cat(directional_feats, dim=1)
        spatial_weight = self.attention_gen(directional_concat)
        
        # 应用空间注意力
        enhanced_x = x * spatial_weight
        
        # 应用通道注意力
        channel_weight = self.channel_att(enhanced_x)
        output = enhanced_x * channel_weight
        
        return output


class MultiScaleCrackAttention(nn.Module):
    """多尺度裂缝注意力"""
    
    def __init__(self, channels, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        
        # 多尺度卷积
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, dilation=s),
                nn.GroupNorm(8, channels),
                nn.ReLU()
            ) for s in scales
        ])
        
        # 尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(channels * len(scales), channels, 1),
            nn.GroupNorm(8, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        scale_feats = []
        for conv in self.scale_convs:
            feat = conv(x)
            scale_feats.append(feat)
        
        # 融合多尺度特征
        scale_concat = torch.cat(scale_feats, dim=1)
        scale_weight = self.scale_fusion(scale_concat)
        
        return x * scale_weight


class EdgeEnhancementModule(nn.Module):
    """边缘增强模块 - 专门针对裂缝边缘优化"""
    
    def __init__(self, channels):
        super().__init__()
        
        # Sobel算子边缘检测
        self.sobel_x = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        
        # 初始化Sobel算子
        self._init_sobel_kernels()
        
        # 边缘增强卷积
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
        # 多方向边缘检测
        self.directional_edges = nn.ModuleList([
            # 水平边缘
            nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), groups=channels),
            # 垂直边缘
            nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), groups=channels),
            # 对角线边缘
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
        ])
        
        # 边缘融合
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            nn.GroupNorm(8, channels),
            nn.Sigmoid()
        )
    
    def _init_sobel_kernels(self):
        """初始化Sobel算子"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # 为每个通道设置相同的Sobel核
        for i in range(self.sobel_x.out_channels):
            self.sobel_x.weight[i, i] = sobel_x
            self.sobel_y.weight[i, i] = sobel_y
    
    def forward(self, x):
        # Sobel边缘检测
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        sobel_edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        
        # 多方向边缘检测
        directional_edges = [sobel_edge]
        for conv in self.directional_edges:
            edge = conv(x)
            directional_edges.append(edge)
        
        # 融合所有边缘信息
        edge_concat = torch.cat(directional_edges, dim=1)
        edge_weight = self.edge_fusion(edge_concat)
        
        # 边缘增强
        enhanced_x = x + x * edge_weight
        
        return enhanced_x


class CrackEdgeRefinement(nn.Module):
    """裂缝边缘精细化模块"""
    
    def __init__(self, channels):
        super().__init__()
        
        # 边缘细化网络
        self.edge_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
        # 裂缝连续性保持
        self.continuity_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        # 边缘细化
        edge_refined = self.edge_refine(x)
        
        # 连续性保持
        continuity_feat = self.continuity_conv(x)
        
        # 融合
        output = x * edge_refined + continuity_feat
        
        return output


class AdaptiveUpsampling(nn.Module):
    """自适应上采样模块 - 针对裂缝特征优化"""
    
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        
        # 可学习的上采样核
        self.learnable_upsample = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=2*scale_factor, 
            stride=scale_factor, 
            padding=scale_factor//2,
            bias=False
        )
        
        # 边缘保持上采样
        self.edge_preserve_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1)
        )
        
        # 裂缝连续性增强
        self.continuity_enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 5, padding=2),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 可学习上采样
        upsampled = self.learnable_upsample(x)
        
        # 边缘保持
        edge_preserved = self.edge_preserve_conv(upsampled)
        
        # 连续性增强
        continuity_weight = self.continuity_enhance(edge_preserved)
        enhanced = edge_preserved * continuity_weight
        
        return enhanced


class ProgressiveUpsampling(nn.Module):
    """渐进式上采样 - 逐步恢复细节"""
    
    def __init__(self, channels, scales=[2, 4, 8]):
        super().__init__()
        self.scales = scales
        
        # 每个尺度的上采样模块
        self.upsample_modules = nn.ModuleList([
            AdaptiveUpsampling(channels, channels, scale)
            for scale in scales
        ])
        
        # 尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(channels * len(scales), channels, 1),
            nn.GroupNorm(8, channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        upsampled_feats = []
        current_feat = x
        
        for i, upsample_module in enumerate(self.upsample_modules):
            upsampled = upsample_module(current_feat)
            upsampled_feats.append(upsampled)
            
            # 为下一个尺度准备输入
            if i < len(self.upsample_modules) - 1:
                current_feat = upsampled
        
        # 融合所有尺度的特征
        if len(upsampled_feats) > 1:
            # 调整到相同尺寸
            target_size = upsampled_feats[-1].shape[-2:]
            aligned_feats = []
            for feat in upsampled_feats:
                if feat.shape[-2:] != target_size:
                    feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                aligned_feats.append(feat)
            
            fused = torch.cat(aligned_feats, dim=1)
            output = self.scale_fusion(fused)
        else:
            output = upsampled_feats[0]
        
        return output


class EnhancedMFS(nn.Module):
    """增强版MFS模块 - 集成所有优化"""
    
    def __init__(self, embedding_dim=8, channels_list=[16, 32, 64, 128]):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.channels_list = channels_list
        
        # 1. 层次化特征融合
        self.hierarchical_fusion = HierarchicalFeatureFusion(channels_list, embedding_dim)
        
        # 2. 裂缝感知注意力
        self.crack_attention = CrackAwareAttention(embedding_dim)
        
        # 3. 多尺度裂缝注意力
        self.multiscale_attention = MultiScaleCrackAttention(embedding_dim)
        
        # 4. 边缘增强
        self.edge_enhancement = EdgeEnhancementModule(embedding_dim)
        
        # 5. 边缘精细化
        self.edge_refinement = CrackEdgeRefinement(embedding_dim)
        
        # 6. 渐进式上采样
        self.progressive_upsampling = ProgressiveUpsampling(embedding_dim, scales=[2, 4, 8])
        
        # 7. 最终预测头
        self.prediction_head = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
            nn.GroupNorm(8, embedding_dim),
            nn.ReLU(),
            nn.Conv2d(embedding_dim, 1, 1),
            nn.Sigmoid()
        )
        
        # 8. 辅助损失分支（用于训练）
        self.auxiliary_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embedding_dim, 1, 1),
                nn.Sigmoid()
            ) for _ in range(len(channels_list))
        ])
    
    def forward(self, inputs, return_aux=False):
        """
        Args:
            inputs: List of feature maps from backbone
            return_aux: Whether to return auxiliary outputs for training
        """
        # 1. 层次化特征融合
        fused_feat = self.hierarchical_fusion(inputs)
        
        # 2. 裂缝感知注意力
        attended_feat = self.crack_attention(fused_feat)
        
        # 3. 多尺度注意力
        multiscale_feat = self.multiscale_attention(attended_feat)
        
        # 4. 边缘增强
        edge_enhanced = self.edge_enhancement(multiscale_feat)
        
        # 5. 边缘精细化
        refined_feat = self.edge_refinement(edge_enhanced)
        
        # 6. 渐进式上采样
        upsampled_feat = self.progressive_upsampling(refined_feat)
        
        # 7. 最终预测
        main_output = self.prediction_head(upsampled_feat)
        
        if return_aux:
            # 辅助输出（用于多尺度监督）
            aux_outputs = []
            for i, aux_head in enumerate(self.auxiliary_heads):
                aux_out = aux_head(refined_feat)
                aux_outputs.append(aux_out)
            return main_output, aux_outputs
        
        return main_output
