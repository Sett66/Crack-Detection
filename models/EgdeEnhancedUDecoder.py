import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# ---------------------------
# 基础模块
# ---------------------------

# --- 在 PatchMerging2D 之后添加以下类 ---
# class PatchExpanding2D(nn.Module):
#     def __init__(self, dim, out_dim):
#         super().__init__()
#         self.dim = dim  # 输入通道数 (C_in)
#         self.out_dim = out_dim  # 输出通道数 (C_out)
#
#         # 1. 线性层：C_in -> 4 * C_out。
#         # 实现通道扩展和准备 Pixel Unshuffling，保证最终输出通道为 out_dim
#         self.linear = nn.Linear(dim, 4 * out_dim, bias=False)
#         # 2. LayerNorm
#         self.norm = nn.LayerNorm(4 * out_dim)
#
#     def forward(self, x):
#         # x: [B, H, W, C_in] (来自 Bottleneck 或上一个 UpBlock)
#
#         # 1. 线性投影和归一化
#         x = self.linear(x)  # [B, H, W, 4*C_out]
#         x = self.norm(x)
#
#         # 2. Pixel Unshuffling (重排): [B, H, W, 4*C_out] -> [B, 2H, 2W, C_out]
#         # p1=2, p2=2 表示空间分辨率放大 2 倍
#         x = rearrange(x, 'b h w (c p1 p2) -> b (h p1) (w p2) c', p1=2, p2=2, c=self.out_dim)
#
#         return x  # [B, 2H, 2W, C_out]

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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvGN_SiLU(in_channels, out_channels, ks=3, pad=1),
            ConvGN_SiLU(out_channels, out_channels, ks=3, pad=1)
        )
    def forward(self, x):
        return self.double_conv(x)

# ---------------------------
# 边界注意（EdgeAttention） - 只对 grad magnitude 做 attention
# ---------------------------
class EdgeAttention(nn.Module):
    def __init__(self, in_ch, mid_ch=None):
        """
        输入: feature map x [B, C, H, W]
        输出: edge-enhanced feature [B, C, H, W]
        Implementation:
          - depthwise sobel_x / sobel_y -> per-channel gradients
          - grad magnitude -> 1-channel map -> small conv -> sigmoid attention
          - produce enhanced feature and return
        """
        super().__init__()
        if mid_ch is None:
            mid_ch = max(8, min(64, in_ch // 4))

        # depthwise sobel_x and sobel_y
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32)
        sobel_y = sobel_x.t()

        # convs for gx and gy: groups=in_ch for depthwise
        self.grad_x = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.grad_y = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        with torch.no_grad():
            kx = sobel_x.unsqueeze(0).unsqueeze(0).repeat(in_ch, 1, 1, 1)
            ky = sobel_y.unsqueeze(0).unsqueeze(0).repeat(in_ch, 1, 1, 1)
            self.grad_x.weight.copy_(kx)
            self.grad_y.weight.copy_(ky)
        self.grad_x.weight.requires_grad = False
        self.grad_y.weight.requires_grad = False

        # refine: input is 1-channel grad magnitude
        self.refine = nn.Sequential(
            nn.Conv2d(1, mid_ch, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # control how much edge modulation to add (residual fraction)
        self.edge_alpha = nn.Parameter(torch.tensor(-1.2))  # sigmoid ~ 0.119 initially -> small effect

    def forward(self, x):
        # x: [B, C, H, W]
        gx = self.grad_x(x)  # [B, C, H, W]
        gy = self.grad_y(x)  # [B, C, H, W]
        # magnitude per channel then sum -> single channel
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6).sum(dim=1, keepdim=True)  # [B,1,H,W]
        attn = self.refine(grad_mag)  # [B,1,H,W] in (0,1)
        rho = torch.sigmoid(self.edge_alpha)  # in (0,1)
        # apply per-pixel multiplicative modulation as lightweight enhancement
        enhanced = x * (1.0 + attn)
        # residual-style weak blending to avoid destroying global features
        out = (1.0 - rho) * x + rho * enhanced
        return out

# ---------------------------
# Channel Align (1x1) - 使 skip 特征与 decoder 通道对齐并做微调
# ---------------------------
class ChannelAlign(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        if in_ch == out_ch:
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True)
            )
    def forward(self, x):
        return self.net(x)

# ---------------------------
# 上采样模块：上采样 -> concat(skip, up) -> DoubleConv -> optional EdgeAttention (residual)
# ---------------------------
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_edge_attn=False):
        super().__init__()
        # up reduces channels to half
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        mid_ch = in_channels // 2
        # F_int = skip_channels // 2  # 设定 F_int = F_l / 2
        #
        # self.attn_gate = AttentionGate(F_g=in_channels, F_l=skip_channels, F_int=F_int)
        self.conv = DoubleConv(mid_ch + skip_channels, out_channels)
        self.use_edge_attn = use_edge_attn
        if use_edge_attn:
            self.edge_attn = EdgeAttention(out_channels)

    def forward(self, x_deep, x_skip):
        # x_deep: [B, in_ch, h, w] -> up to [B, in_ch//2, 2h, 2w]

        x = self.up(x_deep)
        # ensure spatial match
        if x_skip is not None and x.shape[2:] != x_skip.shape[2:]:
            x = F.interpolate(x, size=x_skip.shape[2:], mode='bilinear', align_corners=False)
        # concat
        if x_skip is None:
            x_cat = x
        else:
            x_cat = torch.cat([x_skip, x], dim=1)
        x_out = self.conv(x_cat)
        if self.use_edge_attn:
            # residual weak fusion inside EdgeAttention
            x_out = self.edge_attn(x_out)
        return x_out


# ---------------------------
# 上采样模块：上采样 -> concat(skip, up) -> DoubleConv -> optional EdgeAttention (residual)
# ---------------------------
# class UNetUpBlock(nn.Module):
#     def __init__(self, in_channels, skip_channels, out_channels, use_edge_attn=False):
#         super().__init__()
#
#         # ⚠️ 修改上采样层：使用 PatchExpanding2D
#         # PatchExpanding(C_in -> C_in/2)
#         # PatchExpanding 期望输入 [B, H, W, C_in]，输出 [B, 2H, 2W, C_out]
#         self.up = PatchExpanding2D(in_channels, in_channels // 2)
#         mid_ch = in_channels // 2  # PatchExpanding 后的通道数
#
#         # DoubleConv 的输入通道：PatchExpanding 输出通道 (mid_ch) + Skip 连接通道 (skip_channels)
#         self.conv = DoubleConv(mid_ch + skip_channels, out_channels)
#         self.use_edge_attn = use_edge_attn
#         if use_edge_attn:
#             self.edge_attn = EdgeAttention(out_channels)
#
#     def forward(self, x_deep, x_skip):
#         # x_deep: [B, C_in, h, w] -> PatchExpanding 需要 [B, h, w, C_in]
#         x_perm = x_deep.permute(0, 2, 3, 1).contiguous()
#
#         # 1. Patch Expanding: [B, h, w, C_in] -> [B, 2h, 2w, C_out]
#         x_up_perm = self.up(x_perm)
#
#         # 2. 转换回 [B, C_out, 2h, 2w]
#         x = x_up_perm.permute(0, 3, 1, 2).contiguous()
#
#         # ensure spatial match (PatchExpanding 保证了 2x 上采样，通常不需要 interpolate)
#         if x_skip is not None and x.shape[2:] != x_skip.shape[2:]:
#             x = F.interpolate(x, size=x_skip.shape[2:], mode='bilinear', align_corners=False)
#
#         # 3. concat
#         if x_skip is None:
#             x_cat = x
#         else:
#             x_cat = torch.cat([x_skip, x], dim=1)
#
#         # 4. DoubleConv and EdgeAttention
#         x_out = self.conv(x_cat)
#         if self.use_edge_attn:
#             x_out = self.edge_attn(x_out)
#         return x_out

class MultiScaleRefine(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv3 = nn.Conv2d(ch, ch, 3, padding=3, dilation=3)
        self.conv5 = nn.Conv2d(ch, ch, 3, padding=5, dilation=5)
        self.fuse = nn.Conv2d(ch * 3, ch, 1)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x_cat = torch.cat([x1, x3, x5], dim=1)
        return self.act(self.fuse(x_cat)) + x


# ---------------------------
# 最终解码器（静态通道）
# ---------------------------
class EnhancedUNetDecoder(nn.Module):
    """
    静态通道版，channels = [16,32,64,128,256]
    - Only last two up blocks use EdgeAttention
    - ChannelAlign applied to each skip feature before fusion
    """
    def __init__(self, n_classes=1):
        super().__init__()
        # C0, C1, C2, C3, C4 = 8, 16, 32, 64, 128
        C0, C1, C2, C3, C4 = 16, 32, 64, 128, 256
        # C0, C1, C2, C3, C4 = 32, 64, 128, 256, 512

        # channel align layers (skip features -> matching decoder expectation)
        self.align0 = ChannelAlign(C0, C0)
        self.align1 = ChannelAlign(C1, C1)
        self.align2 = ChannelAlign(C2, C2)
        self.align3 = ChannelAlign(C3, C3)
        self.align4 = ChannelAlign(C4, C4)

        # up blocks (from deepest to shallow)
        # up1: C4 -> C3 (use_edge=False)
        self.up1 = UNetUpBlock(in_channels=C4, skip_channels=C3, out_channels=C3, use_edge_attn=False)
        self.up2 = UNetUpBlock(in_channels=C3, skip_channels=C2, out_channels=C2, use_edge_attn=True)
        # last two with EdgeAttention enabled
        self.up3 = UNetUpBlock(in_channels=C2, skip_channels=C1, out_channels=C1, use_edge_attn=True)
        self.up4 = UNetUpBlock(in_channels=C1, skip_channels=C0, out_channels=C0, use_edge_attn=True)

        # final refine and classifier
        self.final_refine = EdgeAttention(C0)
        self.ms_refine = MultiScaleRefine(C0)
        self.out_conv = nn.Conv2d(C0, n_classes, 1)

    def forward(self, x_outs):
        """
        x_outs = [s0@H, s1@H/2, s2@H/4, s3@H/8, s4@H/16]
        channels = [16,32,64,128,256]
        """
        # defensive unpack
        if not isinstance(x_outs, (list, tuple)) or len(x_outs) != 5:
            raise ValueError("x_outs must be list of 5 tensors: [s0,s1,s2,s3,s4]")

        x_s0, x_s1, x_s2, x_s3, x_s4 = x_outs

        # align skip channels (safe no-op if channels match)
        x_s0 = self.align0(x_s0)
        x_s1 = self.align1(x_s1)
        x_s2 = self.align2(x_s2)
        x_s3 = self.align3(x_s3)
        x_s4 = self.align4(x_s4)

        # decode path
        x = x_s4                     # bottleneck
        x = self.up1(x, x_s3)        # -> size of s3
        x = self.up2(x, x_s2)        # -> size of s2
        x = self.up3(x, x_s1)        # -> size of s1 (edge attn)
        x = self.up4(x, x_s0)        # -> size of s0 (edge attn)

        x = self.final_refine(x)
        x = self.ms_refine(x)
        logits = self.out_conv(x)

        return logits

# ---------------------------
# quick test (auto device)
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 1
    s0 = torch.randn(B, 16, 256, 256, device=device)
    s1 = torch.randn(B, 32, 128, 128, device=device)
    s2 = torch.randn(B, 64, 64, 64, device=device)
    s3 = torch.randn(B, 128, 32, 32, device=device)
    s4 = torch.randn(B, 256, 16, 16, device=device)
    outs = [s0, s1, s2, s3, s4]

    model = EnhancedUNetDecoder(n_classes=1).to(device)
    model.eval()
    with torch.no_grad():
        y = model(outs)
    print("Output shape:", y.shape)  # expect [B, 1, 256, 256]
