import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from timm.models.layers import DropPath
from models.GBC import GBC

class ConvGN_SiLU(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, pad=1, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, pad, bias=False)
        g = min(groups, out_ch)
        self.norm = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = (2 * dim) if out_dim < 0 else out_dim
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        # x: [..., H, W, C] - expects last three dims H,W,C
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return x

    def forward(self, x):
        # x: [B, H, W, C]
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x  # [B, H/2, W/2, out_dim]

def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    return torch.matmul(x1, x2.transpose(-2, -1))

class LocalCluster(nn.Module):
    def __init__(self, dim, w_size=8, clusters=5):
        super().__init__()
        self.dim = dim
        self.w_size = w_size
        self.clusters = clusters
        self.centers_proposal = nn.AdaptiveAvgPool2d((clusters, clusters))
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        # internal small proj
        small = max(1, dim // 16)
        self.f = nn.Conv2d(dim // 2, small, kernel_size=1)
        self.v = nn.Conv2d(dim // 2, small, kernel_size=1)
        self.p = nn.Conv2d(small, dim, kernel_size=1)

    def cluster(self, f, v, Wg, Hg):
        # f, v: [(b*Wg*Hg), c_small, w_tile, h_tile]
        bb, cc, ww, hh = f.shape
        centers = self.centers_proposal(f)  # [b*... , c_small, clusters, clusters]
        val_centers = rearrange(self.centers_proposal(v), 'b c w h -> b (w h) c')  # [Btiles, clusters, c_small]
        # similarity between centers and patches
        sim = torch.sigmoid(self.sim_beta + self.sim_alpha * pairwise_cos_sim(
            centers.reshape(bb, cc, -1).permute(0, 2, 1),
            f.reshape(bb, cc, -1).permute(0, 2, 1)
        ))  # [Btiles, clusters, (w*h)]
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim, device=sim.device)
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        v2 = rearrange(v, 'b c w h -> b (w h) c')  # [Btiles, (w*h), c_small]
        out = ((v2.unsqueeze(1) * sim.unsqueeze(-1)).sum(dim=2) + val_centers) / (sim.sum(dim=-1, keepdim=True) + 1.0)
        out = (out.unsqueeze(2) * sim.unsqueeze(-1)).sum(dim=1)  # [Btiles, (w*h), c_small]
        out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        out = rearrange(out, "(b Wg Hg) e w h-> b e (Wg w) (Hg h)", Wg=Wg, Hg=Hg)
        return out

    def forward(self, x_in):
        # x_in: [B, C, H, W]
        B, C, H, W = x_in.shape
        pad_h = (self.w_size - (H % self.w_size)) % self.w_size
        pad_w = (self.w_size - (W % self.w_size)) % self.w_size
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x_in, (0, pad_w, 0, pad_h))
        else:
            x = x_in
        # tile reshape
        x_tiles = rearrange(x, "b c (Wg w) (Hg h)-> (b Wg Hg) c w h", Wg=self.w_size, Hg=self.w_size)
        # split channels
        x1, x2 = x_tiles.chunk(2, dim=1)
        f = self.f(x1)
        v = self.v(x2)
        out = self.cluster(f, v, self.w_size, self.w_size)  # [B, small_c, H_pad, W_pad]
        out = self.p(out)  # [B, dim, H_pad, W_pad]
        out = out[:, :, :H, :W]  # crop
        return out  # [B, C, H, W]


class SAVSS_2D_DirConv(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, conv_size=7, bias=False, conv_bias=False,
                 init_layer_scale=None, default_hw_shape=None, w_size=8, reduction_ratio=2):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.default_hw_shape = default_hw_shape
        self.n_directions = 4
        self.init_layer_scale = init_layer_scale

        # ✅ 用Conv2d代替Linear实现in/out投影（节省flatten显存）
        self.in_proj = nn.Conv2d(self.d_model, self.d_inner * 2, kernel_size=1, bias=bias)
        self.out_proj = nn.Conv2d(self.d_inner, self.d_model, kernel_size=1, bias=bias)

        # DSC的消融实验 换成GBC
        self.GBC= GBC(d_model)

        # 新增共享降维层
        C_red = d_model // reduction_ratio
        self.dir_conv_reduce = ConvGN_SiLU(d_model, C_red, ks=1, stride=1, pad=0)
        self.conv_h = nn.Conv2d(C_red, self.d_inner, kernel_size=(1, conv_size),
                                padding=(0, conv_size // 2), groups=C_red, bias=bias)  # ⚠️ C_red 替换 d_model
        self.conv_v = nn.Conv2d(C_red, self.d_inner, kernel_size=(conv_size, 1),
                                padding=(conv_size // 2, 0), groups=C_red, bias=bias)  # ⚠️ C_red 替换 d_model

        # 保持多尺度各向同性卷积，但输入通道变为 C_red
        self.conv_d1_k7 = nn.Conv2d(C_red, self.d_inner, kernel_size=7,
                                    padding=3, groups=C_red, bias=conv_bias)  # ⚠️ C_red 替换 d_model
        self.conv_d2_k9 = nn.Conv2d(C_red, self.d_inner, kernel_size=9,
                                    padding=4, groups=C_red, bias=conv_bias)  # ⚠️ C_red 替换 d_model

        self.act = nn.SiLU()

        # scan参数
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.fuse_conv = ConvGN_SiLU(4 * self.d_inner, self.d_model, ks=(1, conv_size), stride=1, pad=(0, conv_size // 2), groups=self.d_model)

        # dt初始化
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions + 1, self.d_state))
        nn.init.trunc_normal_(self.direction_Bs, std=0.02)

        # ✅ cluster分支与融合参数
        self.local_cluster = LocalCluster(dim=self.d_model, w_size=w_size, clusters=5)
        self.cluster_proj = nn.Conv2d(self.d_model, self.d_inner, kernel_size=1, bias=True)
        nn.init.zeros_(self.cluster_proj.bias)

        self.branch_channel = nn.Parameter(torch.ones(2, self.d_inner))  # 每通道可学习权重
        self.branch_norm = nn.LayerNorm(self.d_inner)

        if self.init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

    # 消融实验 不蛇形扫描：
    # E1 纯横向
    # def sass(self, H, W):
    #     L = H * W
    #     o1, o2, o3, o4 = [], [], [], []
    #     o1_inv = [-1] * L
    #     o2_inv = [-1] * L
    #     o3_inv = [-1] * L
    #     o4_inv = [-1] * L
    #
    #     # --- o1: horizontal raster from (0,0) ---
    #     for i in range(H):
    #         for j in range(W):
    #             idx = i * W + j
    #             o1_inv[idx] = len(o1)
    #             o1.append(idx)
    #
    #     # --- o2: horizontal raster reversed from (H-1, W-1) ---
    #     for i in reversed(range(H)):
    #         for j in reversed(range(W)):
    #             idx = i * W + j
    #             o2_inv[idx] = len(o2)
    #             o2.append(idx)
    #
    #     # --- o3: same as o1 (再来一遍正向) ---
    #     for i in range(H):
    #         for j in range(W):
    #             idx = i * W + j
    #             o3_inv[idx] = len(o3)
    #             o3.append(idx)
    #
    #     # --- o4: same as o2 (再来一遍反向) ---
    #     for i in reversed(range(H)):
    #         for j in reversed(range(W)):
    #             idx = i * W + j
    #             o4_inv[idx] = len(o4)
    #             o4.append(idx)
    #
    #     return (
    #         (tuple(o1), tuple(o2), tuple(o3), tuple(o4)),
    #         (tuple(o1_inv), tuple(o2_inv), tuple(o3_inv), tuple(o4_inv)),
    #     )

    # E2 纯纵向
    # def sass(self, H, W):
    #     L = H * W
    #     o1, o2, o3, o4 = [], [], [], []
    #     o1_inv = [-1] * L
    #     o2_inv = [-1] * L
    #     o3_inv = [-1] * L
    #     o4_inv = [-1] * L
    #
    #     # --- o1: vertical raster from top to bottom, left to right ---
    #     for j in range(W):
    #         for i in range(H):
    #             idx = i * W + j
    #             o1_inv[idx] = len(o1)
    #             o1.append(idx)
    #
    #     # --- o2: vertical raster reversed from bottom-right ---
    #     for j in reversed(range(W)):
    #         for i in reversed(range(H)):
    #             idx = i * W + j
    #             o2_inv[idx] = len(o2)
    #             o2.append(idx)
    #
    #     # --- o3: same as o1 ---
    #     for j in range(W):
    #         for i in range(H):
    #             idx = i * W + j
    #             o3_inv[idx] = len(o3)
    #             o3.append(idx)
    #
    #     # --- o4: same as o2 ---
    #     for j in reversed(range(W)):
    #         for i in reversed(range(H)):
    #             idx = i * W + j
    #             o4_inv[idx] = len(o4)
    #             o4.append(idx)
    #
    #     return (
    #         (tuple(o1), tuple(o2), tuple(o3), tuple(o4)),
    #         (tuple(o1_inv), tuple(o2_inv), tuple(o3_inv), tuple(o4_inv)),
    #     )

    # E3 4方向混合
    # def sass(self, H, W):
    #     L = H * W
    #     o1, o2, o3, o4 = [], [], [], []
    #     o1_inv = [-1] * L
    #     o2_inv = [-1] * L
    #     o3_inv = [-1] * L
    #     o4_inv = [-1] * L
    #
    #     # --- o1: horizontal raster (row-major) ---
    #     for i in range(H):
    #         for j in range(W):
    #             idx = i * W + j
    #             o1_inv[idx] = len(o1)
    #             o1.append(idx)
    #
    #     # --- o2: vertical raster (column-major) ---
    #     for j in range(W):
    #         for i in range(H):
    #             idx = i * W + j
    #             o2_inv[idx] = len(o2)
    #             o2.append(idx)
    #
    #     # --- o3: main diagonal scan (non-snake) ---
    #     # i + j = diag
    #     for diag in range(H + W - 1):
    #         ii_min = max(0, diag - (W - 1))
    #         ii_max = min(H - 1, diag)
    #         for ii in range(ii_min, ii_max + 1):
    #             jj = diag - ii
    #             idx = ii * W + jj
    #             o3_inv[idx] = len(o3)
    #             o3.append(idx)
    #
    #     # --- o4: anti-diagonal scan (non-snake) ---
    #     # i + (W-1-j) = diag  => j = (W-1) - (diag - i)
    #     for diag in range(H + W - 1):
    #         ii_min = max(0, diag - (W - 1))
    #         ii_max = min(H - 1, diag)
    #         for ii in range(ii_min, ii_max + 1):
    #             jj = diag - ii
    #             j = (W - 1) - jj
    #             idx = ii * W + j
    #             o4_inv[idx] = len(o4)
    #             o4.append(idx)
    #
    #
    #     return (
    #         (tuple(o1), tuple(o2), tuple(o3), tuple(o4)),
    #         (tuple(o1_inv), tuple(o2_inv), tuple(o3_inv), tuple(o4_inv)),
    #     )

    # 蛇形扫描的
    # E4 纯横向
    # def sass(self, H, W):
    #     L = H * W
    #     o1, o2, o3, o4 = [], [], [], []
    #     o1_inv = [-1] * L
    #     o2_inv = [-1] * L
    #     o3_inv = [-1] * L
    #     o4_inv = [-1] * L
    #
    #     # --- 构造横向蛇形基础路径 base ---
    #     base = []
    #     for i in range(H):
    #         if i % 2 == 0:
    #             js = range(W)  # 偶数行：左->右
    #         else:
    #             js = range(W - 1, -1, -1)  # 奇数行：右->左
    #         for j in js:
    #             base.append(i * W + j)
    #
    #     # 安全检查（需要的话可以打开）
    #     # assert len(base) == L and len(set(base)) == L
    #
    #     # --- o1: base ---
    #     for pos, idx in enumerate(base):
    #         o1_inv[idx] = pos
    #         o1.append(idx)
    #
    #     # --- o2: reversed(base) ---
    #     for pos, idx in enumerate(reversed(base)):
    #         o2_inv[idx] = pos
    #         o2.append(idx)
    #
    #     # --- o3: 再来一遍 base ---
    #     for pos, idx in enumerate(base):
    #         o3_inv[idx] = pos
    #         o3.append(idx)
    #
    #     # --- o4: 再来一遍 reversed(base) ---
    #     for pos, idx in enumerate(reversed(base)):
    #         o4_inv[idx] = pos
    #         o4.append(idx)
    #
    #     return (
    #         (tuple(o1), tuple(o2), tuple(o3), tuple(o4)),
    #         (tuple(o1_inv), tuple(o2_inv), tuple(o3_inv), tuple(o4_inv)),
    #     )

    # E5 纯纵向蛇形
    # def sass(self, H, W):
    #     L = H * W
    #     o1, o2, o3, o4 = [], [], [], []
    #     o1_inv = [-1] * L
    #     o2_inv = [-1] * L
    #     o3_inv = [-1] * L
    #     o4_inv = [-1] * L
    #
    #     # --- 构造纵向蛇形基础路径 base ---
    #     base = []
    #     for j in range(W):
    #         if j % 2 == 0:
    #             iseq = range(H)  # 偶数列：上->下
    #         else:
    #             iseq = range(H - 1, -1, -1)  # 奇数列：下->上
    #         for i in iseq:
    #             base.append(i * W + j)
    #
    #     # assert len(base) == L and len(set(base)) == L
    #
    #     # --- o1: base ---
    #     for pos, idx in enumerate(base):
    #         o1_inv[idx] = pos
    #         o1.append(idx)
    #
    #     # --- o2: reversed(base) ---
    #     for pos, idx in enumerate(reversed(base)):
    #         o2_inv[idx] = pos
    #         o2.append(idx)
    #
    #     # --- o3: base ---
    #     for pos, idx in enumerate(base):
    #         o3_inv[idx] = pos
    #         o3.append(idx)
    #
    #     # --- o4: reversed(base) ---
    #     for pos, idx in enumerate(reversed(base)):
    #         o4_inv[idx] = pos
    #         o4.append(idx)
    #
    #     return (
    #         (tuple(o1), tuple(o2), tuple(o3), tuple(o4)),
    #         (tuple(o1_inv), tuple(o2_inv), tuple(o3_inv), tuple(o4_inv)),
    #     )

    # 再补两个只有对角的
    # E7纯对角非蛇形
    # def sass(self, H, W):
    #     L = H * W
    #     o1, o2, o3, o4 = [], [], [], []
    #     o1_inv = [-1] * L
    #     o2_inv = [-1] * L
    #     o3_inv = [-1] * L
    #     o4_inv = [-1] * L
    #
    #     # --- 构造主对角扫描（非蛇形） base_main ---
    #     base_main = []
    #     # i + j = diag
    #     for diag in range(H + W - 1):
    #         ii_min = max(0, diag - (W - 1))
    #         ii_max = min(H - 1, diag)
    #         for ii in range(ii_min, ii_max + 1):
    #             jj = diag - ii
    #             if 0 <= jj < W:
    #                 base_main.append(ii * W + jj)
    #
    #     # --- 构造反对角扫描（非蛇形） base_anti ---
    #     base_anti = []
    #     # i + (W-1-j) = diag  => j = (W-1) - (diag - i)
    #     for diag in range(H + W - 1):
    #         ii_min = max(0, diag - (W - 1))
    #         ii_max = min(H - 1, diag)
    #         for ii in range(ii_min, ii_max + 1):
    #             jj = diag - ii
    #             j = (W - 1) - jj
    #             if 0 <= j < W:
    #                 base_anti.append(ii * W + j)
    #
    #     # --- o1: 主对角 base_main ---
    #     for pos, idx in enumerate(base_main):
    #         o1_inv[idx] = pos
    #         o1.append(idx)
    #
    #     # --- o2: 反对角 base_anti ---
    #     for pos, idx in enumerate(base_anti):
    #         o2_inv[idx] = pos
    #         o2.append(idx)
    #
    #     # --- o3: 主对角反向 ---
    #     for pos, idx in enumerate(reversed(base_main)):
    #         o3_inv[idx] = pos
    #         o3.append(idx)
    #
    #     # --- o4: 反对角反向 ---
    #     for pos, idx in enumerate(reversed(base_anti)):
    #         o4_inv[idx] = pos
    #         o4.append(idx)
    #     return (
    #         (tuple(o1), tuple(o2), tuple(o3), tuple(o4)),
    #         (tuple(o1_inv), tuple(o2_inv), tuple(o3_inv), tuple(o4_inv)),
    #     )

    # E8 纯对角蛇形
    # def sass(self, H, W):
    #     L = H * W
    #     o1, o2, o3, o4 = [], [], [], []
    #     o1_inv = [-1] * L
    #     o2_inv = [-1] * L
    #     o3_inv = [-1] * L
    #     o4_inv = [-1] * L
    #
    #     # --- 构造主对角 diagonal snake base_main ---
    #     base_main = []
    #     # i + j = diag
    #     for diag in range(H + W - 1):
    #         ii_min = max(0, diag - (W - 1))
    #         ii_max = min(H - 1, diag)
    #         line = []
    #         for ii in range(ii_min, ii_max + 1):
    #             jj = diag - ii
    #             if 0 <= jj < W:
    #                 line.append(ii * W + jj)
    #
    #         # 对角线级别蛇形：奇数 diag 反向
    #         if diag % 2 == 1:
    #             line.reverse()
    #
    #         base_main.extend(line)
    #
    #     # --- 构造反对角 anti-diagonal snake base_anti ---
    #     base_anti = []
    #     # i + (W-1-j) = diag => j = (W-1) - (diag - i)
    #     for diag in range(H + W - 1):
    #         ii_min = max(0, diag - (W - 1))
    #         ii_max = min(H - 1, diag)
    #         line = []
    #         for ii in range(ii_min, ii_max + 1):
    #             jj = diag - ii
    #             j = (W - 1) - jj
    #             if 0 <= j < W:
    #                 line.append(ii * W + j)
    #
    #         if diag % 2 == 1:
    #             line.reverse()
    #
    #         base_anti.extend(line)
    #
    #     # --- o1: 主对角蛇形 base_main ---
    #     for pos, idx in enumerate(base_main):
    #         o1_inv[idx] = pos
    #         o1.append(idx)
    #
    #     # --- o2: 反对角蛇形 base_anti ---
    #     for pos, idx in enumerate(base_anti):
    #         o2_inv[idx] = pos
    #         o2.append(idx)
    #
    #     # --- o3: 主对角蛇形反向 ---
    #     for pos, idx in enumerate(reversed(base_main)):
    #         o3_inv[idx] = pos
    #         o3.append(idx)
    #
    #     # --- o4: 反对角蛇形反向 ---
    #     for pos, idx in enumerate(reversed(base_anti)):
    #         o4_inv[idx] = pos
    #         o4.append(idx)
    #
    #     return (
    #         (tuple(o1), tuple(o2), tuple(o3), tuple(o4)),
    #         (tuple(o1_inv), tuple(o2_inv), tuple(o3_inv), tuple(o4_inv)),
    #     )

    # 我的版本：里面的对角扫描开始不是蛇形的，后面加上了
    def sass(self, H, W):
        L = H * W
        o1, o2, o3, o4 = [], [], [], []
        o1_inv = [-1] * L
        o2_inv = [-1] * L
        o3_inv = [-1] * L
        o4_inv = [-1] * L
        # snake
        if H % 2 == 1:
            i, j = H - 1, W - 1
            j_d = "left"
        else:
            i, j = H - 1, 0
            j_d = "right"
        while i > -1:
            idx = i * W + j
            o1_inv[idx] = len(o1); o1.append(idx)
            if j_d == "right":
                if j < W - 1: j += 1
                else: i -= 1; j_d = "left"
            else:
                if j > 0: j -= 1
                else: i -= 1; j_d = "right"
        # vertical snake
        i, j = 0, 0; i_d = "down"
        while j < W:
            idx = i * W + j
            o2_inv[idx] = len(o2); o2.append(idx)
            if i_d == "down":
                if i < H-1: i += 1
                else: j += 1; i_d = "up"
            else:
                if i > 0: i -= 1
                else: j += 1; i_d = "down"
        # diag 不是蛇形的
        # for diag in range(H + W - 1):
        #     for ii in range(min(diag+1, H)):
        #         jj = diag - ii
        #         if jj < W:
        #             idx = ii * W + jj
        #             o3_inv[idx] = len(o3)
        #             o3.append(idx)
        # # anti-diag
        # for diag in range(H + W - 1):
        #     for ii in range(min(diag+1, H)):
        #         jj = diag - ii
        #         if jj < W:
        #             idx = ii * W + (W - jj - 1)
        #             o4_inv[idx] = len(o4)
        #             o4.append(idx)

        # 蛇形扫描的
        for diag in range(H + W - 1):
            line = []
            # 注意：ii 的范围要保证 jj 在 [0, W-1]
            ii_min = max(0, diag - (W - 1))
            ii_max = min(H - 1, diag)
            for ii in range(ii_min, ii_max + 1):
                jj = diag - ii
                idx = ii * W + jj
                line.append(idx)

            if diag % 2 == 1:
                line.reverse()

            for idx in line:
                o3_inv[idx] = len(o3)
                o3.append(idx)

        for diag in range(H + W - 1):
            line = []
            ii_min = max(0, diag - (W - 1))
            ii_max = min(H - 1, diag)
            for ii in range(ii_min, ii_max + 1):
                jj = diag - ii
                j = (W - 1) - jj
                idx = ii * W + j
                line.append(idx)

            if diag % 2 == 1:
                line.reverse()

            for idx in line:
                o4_inv[idx] = len(o4)
                o4.append(idx)
        return (tuple(o1), tuple(o2), tuple(o3), tuple(o4)), (tuple(o1_inv), tuple(o2_inv), tuple(o3_inv), tuple(o4_inv))


    def forward(self, x_norm):
        # 输入 x_norm: [B, H, W, C]
        B, H, W, C = x_norm.shape
        conv_state, ssm_state = None, None

        # 改成 [B, C, H, W]
        x_2d = x_norm.permute(0, 3, 1, 2).contiguous()

        # === ⚠️ 共享降维 ===
        # x_reduced = self.dir_conv_reduce(x_2d)  # [B, C_red, H, W]
        #
        # # 方向卷积 (使用降维后的特征)
        # x_h = self.act(self.conv_h(x_reduced))  # [B, E, H, W]
        # x_v = self.act(self.conv_v(x_reduced))  # [B, E, H, W]
        # x_d1 = self.act(self.conv_d1_k7(x_reduced))  # [B, E, H, W]
        # x_d2 = self.act(self.conv_d2_k9(x_reduced))  # [B, E, H, W]
        #
        # x_concat = torch.cat([x_h, x_v, x_d1, x_d2], dim=1)
        # x_conv = self.fuse_conv(x_concat)  # [B, C, H, W]

        # 消融实验把DSC替换为GBC
        x_conv=self.GBC(x_2d)

        # ✅ in_proj 改为 conv，不再flatten
        xz = self.in_proj(x_conv)  # [B, 2E, H, W]
        x_proj, z_proj = torch.chunk(xz, 2, dim=1)
        # xz = self.in_proj(x_2d)
        # x_proj, z_proj = torch.chunk(xz, 2, dim=1)

        # flatten 仅保留一次
        L = H * W
        x_proj_seq = x_proj.flatten(2).transpose(1, 2)  # [B, L, E]
        z_proj_seq = z_proj.flatten(2).transpose(1, 2)

        # selective scan
        A = -torch.exp(self.A_log.float())
        x_dbl = self.x_proj(x_proj_seq)
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).permute(0, 2, 1).contiguous()
        Bm = Bm.permute(0, 2, 1).contiguous()
        Cm = Cm.permute(0, 2, 1).contiguous()

        orders, inv_orders = self.sass(H, W)
        y_scan = [
            selective_scan_fn(
                x_proj_seq[:, o, :].permute(0, 2, 1).contiguous(),
                dt,
                A,
                (Bm + dB).contiguous(),
                Cm,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ).permute(0, 2, 1)[:, inv_order, :]
            for o, inv_order, dB in zip(orders, inv_orders, [self.direction_Bs[d].view(1, -1, 1).expand(B, -1, 1).to(dtype=Bm.dtype) for d in range(4)])
        ]
        y_scan_result = sum(y_scan)
        y_no_scan = z_proj_seq

        # x_cluster = self.local_cluster(x_conv.detach())  # [B, C, H, W]
        # x_cluster = self.local_cluster(x_2d.detach())  # [B, C, H, W]
        # x_cluster_proj = self.cluster_proj(x_cluster)  # [B, E, H, W]
        # x_cluster_seq = x_cluster_proj.flatten(2).transpose(1, 2)  # [B, L, E]

        # normalize后融合
        y_scan_n = self.branch_norm(y_scan_result)
        y_no_n = self.branch_norm(y_no_scan)
        # x_cl_n = self.branch_norm(x_cluster_seq)

        ch_w = F.softmax(self.branch_channel, dim=0)
        y = (
            ch_w[0].view(1, 1, -1) * y_scan_n
            + ch_w[1].view(1, 1, -1) * y_no_n
            # + ch_w[2].view(1, 1, -1) * x_cl_n
        )

        out_seq = y.transpose(1, 2).reshape(B, self.d_inner, H, W)
        out = self.out_proj(out_seq)  # [B, C, H, W]

        if self.init_layer_scale is not None:
            out = out * self.gamma

        return out


class SAVSS_Layer(nn.Module):
    def __init__(self, embed_dims, use_rms_norm=False, with_dwconv=False, drop_path_rate=0.0, mamba_cfg=None,w_size=8):
        super().__init__()
        self.embed_dims = embed_dims
        if use_rms_norm and RMSNorm is not None:
            self.norm = RMSNorm(embed_dims)
        else:
            self.norm = nn.LayerNorm(embed_dims)
        if mamba_cfg is None:
            mamba_cfg = {}
        mamba_cfg.update({'d_model': embed_dims, 'w_size': w_size})
        self.core = SAVSS_2D_DirConv(**mamba_cfg)
        self.drop_path = DropPath(drop_prob=drop_path_rate) if hasattr(DropPath, '__call__') or True else nn.Identity()

    def forward(self, x):
        # x: [B, C, H, W]
        x_perm = x.permute(0,2,3,1).contiguous()  # [B, H, W, C]
        x_norm = self.norm(x_perm)  # layernorm on last dim
        out = self.core(x_norm)  # returns [B, C, H, W]
        out = self.drop_path(out)
        return x + out

class SAVSS_Stage(nn.Module):
    def __init__(self, depth, dim, next_dim=None, mamba_cfg=None, w_size=8, drop_path_rate=0.0):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(SAVSS_Layer(embed_dims=dim, use_rms_norm=False, with_dwconv=False, drop_path_rate=drop_path_rate, mamba_cfg=mamba_cfg, w_size=w_size))
        self.layers = nn.Sequential(*layers)
        self.down = None
        if next_dim is not None:
            self.down = PatchMerging2D(dim, out_dim=next_dim)
            # self.down = nn.Conv2d(dim, next_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: [B, C, H, W]
        # process with layers (each expects [B,C,H,W])
        x = self.layers(x)
        if self.down is not None:
            # convert to [B,H,W,C] then down, then back
            x_perm = x.permute(0,2,3,1).contiguous()
            x_perm = self.down(x_perm)  # [B, H/2, W/2, C_next]
            x = x_perm.permute(0,3,1,2).contiguous()
        return x

class SAVSS_Backbone(nn.Module):
    def __init__(self, in_ch=3, base_dim=16, depths=[1,1,1,1], mamba_cfg=None, w_sizes=[8, 4, 2, 1]):
        super().__init__()
        self.stem = nn.Sequential(
            ConvGN_SiLU(in_ch, base_dim, ks=3, stride=1, pad=1),
            ConvGN_SiLU(base_dim, base_dim, ks=3, stride=1, pad=1)
        )
        self.num_stages = len(depths)
        dims = [base_dim * (2**(i+1)) for i in range(self.num_stages)]
        current_dim = base_dim  # 输入 Stage 1 的通道数 (来自 initial_stem)
        self.w_sizes = w_sizes
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):  # i = 0, 1, 2, 3
            stage_dim = current_dim

            # next_dim 是 Stage i 结束后下采样得到的通道数
            next_dim = dims[i]
            current_s_size = w_sizes[i]

            # SAVSS_Stage(输入通道, 下采样后输出通道, 深度, ...)
            # 每个 Stage i 都会将特征图分辨率减半
            stage = SAVSS_Stage(depths[i], stage_dim, next_dim, mamba_cfg, current_s_size)
            self.stages.append(stage)

            # 更新下一阶段的输入通道数
            current_dim = next_dim

    def forward(self, x):
        # x: [B, 3, H, W]
        outs = []
        x = self.stem(x)  # [B, base_dim, H, W]
        outs.append(x)  # Output 0: 32 @ 256x256 (用于最后一个跳跃连接)
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs

