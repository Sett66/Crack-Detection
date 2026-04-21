import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.cnn.bricks.transformer import build_dropout
from models.GBC import BottConv,GBC

class DiagonalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, groups=1, anti=False):
        super().__init__(in_channels, out_channels, kernel_size,
                         padding=kernel_size // 2, bias=bias, groups=groups)

        # 构造 mask
        mask = torch.zeros((out_channels, in_channels // groups, kernel_size, kernel_size))
        if anti:  # 反对角线
            for i in range(kernel_size):
                mask[:, :, i, kernel_size - i - 1] = 1.0
        else:  # 主对角线
            for i in range(kernel_size):
                mask[:, :, i, i] = 1.0

        self.register_buffer("mask", mask)

    def forward(self, x):
        # 权重与 mask 相乘，保证只有对角权重有效
        w = self.weight * self.mask
        return nn.functional.conv2d(x, w, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)

def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim

class LocalCluster(nn.Module):
    def __init__(self, dim, w_size=7, clusters=5):
        super().__init__()
        self.dim = dim
        self.w_size = w_size
        self.clusters = clusters
        self.centers_proposal = nn.AdaptiveAvgPool2d((self.clusters, self.clusters))

        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))

        self.f = nn.Conv2d(self.dim // 2, self.dim // 16, kernel_size=1)
        self.v = nn.Conv2d(self.dim // 2, self.dim // 16, kernel_size=1)
        self.p = nn.Conv2d(self.dim // 16, self.dim, kernel_size=1)

    def cluster(self, f, v, Wg, Hg):
        bb, cc, ww, hh = f.shape
        centers = self.centers_proposal(f)
        value_centers = rearrange(self.centers_proposal(v), 'b c w h -> b (w h) c')

        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(bb, cc, -1).permute(0, 2, 1),
                f.reshape(bb, cc, -1).permute(0, 2, 1)
            )
        )
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask

        v2 = rearrange(v, 'b c w h -> b (w h) c')
        out = ((v2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                sim.sum(dim=-1, keepdim=True) + 1.0)

        out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)
        out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        out = rearrange(out, "(b Wg Hg) e w h-> b e (Wg w) (Hg h)", Wg=Wg, Hg=Hg)
        return out

    def forward(self, x_in):
        # x_in: [B, C, H, W]
        x = rearrange(x_in, "b e (Wg w) (Hg h)-> (b Wg Hg) e w h", Wg=self.w_size, Hg=self.w_size)
        x1, x2 = x.chunk(2, dim=1)
        f = self.f(x1)
        v = self.v(x2)
        out = self.cluster(f, v, self.w_size, self.w_size)
        out = self.p(out)
        out = out.flatten(2)  # [B, C, L]
        return out

class SAVSS_2D_DirConv(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, conv_size=7, bias=False, conv_bias=False,
                 init_layer_scale=None, default_hw_shape=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.default_hw_shape = default_hw_shape
        self.n_directions = 4

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        # 输入投影（保持与原来一致）
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        # 四路方向卷积（可带 bias）
        # 输入通道数为 d_model，输出通道数为 d_inner
        self.conv_h = nn.Conv2d(self.d_model, self.d_inner, kernel_size=(1, conv_size),
                                padding=(0, conv_size // 2), groups=self.d_model, bias=bias)
        self.conv_v = nn.Conv2d(self.d_model, self.d_inner, kernel_size=(conv_size, 1),
                                padding=(conv_size // 2, 0), groups=self.d_model, bias=bias)
        # 对角方向用大核近似（可之后换成 rotated conv / separable diag conv）
        self.conv_d1 = nn.Conv2d(self.d_model, self.d_inner, kernel_size=(conv_size, conv_size),
                                 padding=(conv_size // 2, conv_size // 2), groups=self.d_model, bias=conv_bias)
        self.conv_d2 = nn.Conv2d(self.d_model, self.d_inner, kernel_size=(conv_size, conv_size),
                                 padding=(conv_size // 2, conv_size // 2), groups=self.d_model, bias=conv_bias)

        # 主对角条形卷积
        # self.conv_d1 = DiagonalConv2d(self.d_model, self.d_inner,
        #                               kernel_size=conv_size,
        #                               groups=self.d_model, bias=bias, anti=False)
        #
        # # 反对角条形卷积
        # self.conv_d2 = DiagonalConv2d(self.d_model, self.d_inner,
        #                               kernel_size=conv_size,
        #                               groups=self.d_model, bias=bias, anti=True)

        self.conv2d = BottConv(in_channels=self.d_inner, out_channels=self.d_inner, mid_channels=self.d_inner // 16,
                               kernel_size=3, padding=1, stride=1)
        self.act = nn.SiLU()

        # 状态空间投影（保持原逻辑）
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        self.fuse_conv = nn.Conv2d(4 * self.d_inner, self.d_model, kernel_size=1, bias=False)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # A, A_log, D 等参数（与原实现一致）
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        # 方向 bias 参数（4 路 + 1 备用）
        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions + 1, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)

        # === 新增 LocalCluster ===
        # self.local_cluster2 = LocalCluster(dim=self.d_model, w_size=8, clusters=2)
        self.local_cluster5 = LocalCluster(dim=self.d_model, w_size=8, clusters=5)
        self.cluster_proj = nn.Linear(self.d_model, self.d_inner)

        # === 三路融合权重 ===
        # self.branch_weights = nn.Parameter(torch.ones(3))
        self.branch_channel = nn.Parameter(torch.ones(3, self.d_inner))  # E = feature dim
        self.branch_norm = nn.LayerNorm(self.d_inner)
        # # self.branch_weights = nn.Conv2d(3*self.d_inner, 3, kernel_size=3, padding=1)
        # self.branch_logits = nn.Parameter(torch.zeros(3))
        # self.branch_tau = nn.Parameter(torch.tensor(1.0))  # learnable

        nn.init.zeros_(self.cluster_proj.bias)

    def sass(self, hw_shape):
        """产生 4 条扫描路径以及相应的 inverse order（沿用了你原始实现思路）"""
        H, W = hw_shape
        L = H * W
        o1, o2, o3, o4 = [], [], [], []
        o1_inverse = [-1 for _ in range(L)]
        o2_inverse = [-1 for _ in range(L)]
        o3_inverse = [-1 for _ in range(L)]
        o4_inverse = [-1 for _ in range(L)]

        # 蛇形
        if H % 2 == 1:
            i, j = H - 1, W - 1
            j_d = "left"
        else:
            i, j = H - 1, 0
            j_d = "right"
        while i > -1:
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W - 1:
                    j += 1
                else:
                    i -= 1;
                    j_d = "left"
            else:
                if j > 0:
                    j -= 1
                else:
                    i -= 1;
                    j_d = "right"

        # 竖向
        i, j = 0, 0
        i_d = "down"
        while j < W:
            idx = i * W + j
            o2_inverse[idx] = len(o2)
            o2.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i += 1
                else:
                    j += 1;
                    i_d = "up"
            else:
                if i > 0:
                    i -= 1
                else:
                    j += 1;
                    i_d = "down"

        # 对角线
        for diag in range(H + W - 1):
            for i in range(min(diag + 1, H)):
                j = diag - i
                if j < W:
                    idx = i * W + j
                    o3_inverse[idx] = len(o3)
                    o3.append(idx)

        # 反对角线
        for diag in range(H + W - 1):
            for i in range(min(diag + 1, H)):
                j = diag - i
                if j < W:
                    idx = i * W + (W - j - 1)
                    o4_inverse[idx] = len(o4)
                    o4.append(idx)

        return (tuple(o1), tuple(o2), tuple(o3), tuple(o4)), \
               (tuple(o1_inverse), tuple(o2_inverse), tuple(o3_inverse), tuple(o4_inverse))

    def forward(self, x, hw_shape):
        """
        输入: x: [B, L, C]  (C == d_model)
        输出: out: [B, L, C] (保持与原模块一致)
        """
        batch_size, L, C = x.shape
        H, W = hw_shape
        E = self.d_inner

        # -------- 第一步：先条形卷积，还原到原始shape --------
        x_2d = x.reshape(batch_size, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        # 四路条形卷积增强
        x_h = self.act(self.conv_h(x_2d))
        x_v = self.act(self.conv_v(x_2d))
        x_d1 = self.act(self.conv_d1(x_2d))
        x_d2 = self.act(self.conv_d2(x_2d))

        # 将四个条形卷积结果融合回原始维度
        x_concat = torch.cat([x_h, x_v, x_d1, x_d2], dim=1)  # [B, 4*E, H, W]
        x_conv = self.fuse_conv(x_concat)  # [B, C, H, W]

        # 转回序列格式
        x_conv_seq = x_conv.permute(0, 2, 3, 1).reshape(batch_size, L, C)  # [B, L, C]

        # -------- 第二步：投影拆分 --------
        xz = self.in_proj(x_conv_seq)  # [B, L, 2*E]
        A = -torch.exp(self.A_log.float())
        x_proj, z_proj = xz.chunk(2, dim=-1)  # 两个分支都是 [B, L, E]

        # 新增 在扫描前加入DWconv
        # x_proj = x_proj.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        # x_proj = self.act(self.conv2d(x_proj))
        # x_proj = x_proj.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        # -------- 第三步：两个分支 --------
        # 分支1：进行扫描
        x_dbl = self.x_proj(x_proj)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        dt = dt.permute(0, 2, 1).contiguous()
        B = B.permute(0, 2, 1).contiguous()
        C = C.permute(0, 2, 1).contiguous()

        # 生成扫描路径
        orders, inverse_orders = self.sass(hw_shape)

        # 方向 bias
        direction_Bs = []
        for d in range(4):
            vec = self.direction_Bs[d]
            dB = vec.view(1, -1, 1).expand(batch_size, -1, 1)
            direction_Bs.append(dB.to(dtype=B.dtype))

        # 四路选择性扫描
        y_scan = []
        for o, inv_order, dB in zip(orders, inverse_orders, direction_Bs):
            scanned = selective_scan_fn(
                x_proj[:, o, :].permute(0, 2, 1).contiguous(),
                dt,
                A,
                (B + dB).contiguous(),
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            scanned = scanned.permute(0, 2, 1)[:, inv_order, :]
            y_scan.append(scanned)

        # 分支1结果：扫描后的特征
        y_scan_result = sum(y_scan)  # [B, L, E]

        # 分支2：不扫描，直接使用z_proj
        y_no_scan = z_proj  # [B, L, E]

        # -------- 新增：聚类分支 --------
        # x_2d = x.reshape(batch_size, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        # x_cluster2 = self.local_cluster2(x_conv)
        # x_cluster2 = x_cluster2.permute(0, 2, 1)
        # x_cluster2 = self.cluster_proj(x_cluster2)

        x_cluster5 = self.local_cluster5(x_conv)
        x_cluster5 = x_cluster5.permute(0, 2, 1)
        x_cluster5 = self.cluster_proj(x_cluster5)


        # 三路融合
        # weights = F.softmax(self.branch_weights, dim=0)
        weights_ch = F.softmax(self.branch_channel, dim=0)
        #新增Normalize
        y_scan_result = self.branch_norm(y_scan_result)
        y_no_scan = self.branch_norm(y_no_scan)
        x_cluster5 = self.branch_norm(x_cluster5)
        # weights = F.softmax(self.branch_logits / (F.softplus(self.branch_tau) + 1e-6), dim=0)
        y = (weights_ch[0] * y_scan_result +
             weights_ch[1] * y_no_scan +
             weights_ch[2] * x_cluster5)

        # 三路位置/空间自适应融合
        # y_scan_2d = y_scan_result.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)  # [B, E, H, W]
        # y_no_scan_2d = y_no_scan.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)  # [B, E, H, W]
        # x_cluster_2d = x_cluster.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)  # [B, E, H, W]
        # m = torch.cat([y_scan_2d, y_no_scan_2d, x_cluster_2d], dim=1)  # convert [B,L,E] -> [B,E,H,W] first
        # w = self.branch_weights(m)
        # w = F.softmax(w, dim=1)  # along branch dim
        # y_2d = w[:, 0:1] * y_scan_2d + w[:, 1:2] * y_no_scan_2d + w[:, 2:3] * x_cluster_2d
        # y = y_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)  # [B, L, E]

        out = self.out_proj(y)  # [B, L, C]

        if self.init_layer_scale is not None:
            out = out * self.gamma

        return out

class SAVSS_Layer(nn.Module):
    def __init__(
            self,
            embed_dims,
            use_rms_norm,
            with_dwconv,
            drop_path_rate,
            mamba_cfg,
    ):

        super(SAVSS_Layer, self).__init__()
        mamba_cfg.update({'d_model': embed_dims})
        if use_rms_norm:
            self.norm = RMSNorm(embed_dims)
        else:
            self.norm = nn.LayerNorm(embed_dims)

        self.SAVSS_2D = SAVSS_2D_DirConv(**mamba_cfg)
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))

    def forward(self, x, hw_shape):

        mixed_x = self.drop_path(self.SAVSS_2D(self.norm(x), hw_shape))

        return mixed_x + x
