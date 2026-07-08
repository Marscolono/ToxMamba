#!/usr/bin/env python3
"""
Mamba-based classifier for peptide toxicity prediction.
Multi-scale Mamba SSM with PSConv kernels (3, 5, 7) and fusion.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum


# ==============================================================================
# RMS Normalization
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ==============================================================================
# PSConv: Pseudo-Siamese Convolutional Networks
# ==============================================================================
class PSConvNetBase(nn.Module):
    def __init__(self, window_size, filter_num, feature, seq_len):
        super().__init__()
        self.filter_num = filter_num
        self.feature = feature
        self.window_size = window_size
        self.seq_len = seq_len
        self.pad_len = int(self.window_size / 2)
        self.dense_layer_net = nn.ModuleList([
            nn.Sequential(nn.Linear(self.window_size * self.feature, filter_num), nn.ReLU(), nn.Dropout(0.2))
            for _ in range(self.seq_len)
        ])

    def forward(self, input_mat):
        h_d = F.pad(input_mat, (self.pad_len, self.pad_len), "constant", 0)
        h_d = torch.transpose(h_d, 1, 2)
        h_d_temp = []
        for i in range(self.pad_len, self.seq_len + self.pad_len):
            segment = h_d[:, i - self.pad_len: i + self.pad_len + 1, :]
            segment_flat = segment.reshape(-1, self.window_size * self.feature)
            h_d_temp.append(self.dense_layer_net[i - self.pad_len](segment_flat))
        h_d = torch.stack(h_d_temp)
        return torch.transpose(h_d, 0, 1)


class PSConvNet1(PSConvNetBase):
    def __init__(self, window_size=3, filter_num=256, feature=256, seq_len=50):
        super().__init__(window_size, filter_num, feature, seq_len)


class PSConvNet2(PSConvNetBase):
    def __init__(self, window_size=5, filter_num=256, feature=256, seq_len=50):
        super().__init__(window_size, filter_num, feature, seq_len)


class PSConvNet3(PSConvNetBase):
    def __init__(self, window_size=7, filter_num=256, feature=256, seq_len=50):
        super().__init__(window_size, filter_num, feature, seq_len)


# ==============================================================================
# Mamba Blocks (Selective State Space Models)
# ==============================================================================
class MambaBlockBase(nn.Module):
    def __init__(self, args, conv_class, conv_attr_name):
        super().__init__()
        self.args = args
        self.conv_attr_name = conv_attr_name
        setattr(self, conv_attr_name, conv_class())
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.x_proj = nn.Linear(int(args.d_inner), int(args.dt_rank) + int(args.d_state) * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = getattr(self, self.conv_attr_name)(x)
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        return y + u * D


class MambaBlock1(MambaBlockBase):
    def __init__(self, args): super().__init__(args, PSConvNet1, "conv1d1")


class MambaBlock2(MambaBlockBase):
    def __init__(self, args): super().__init__(args, PSConvNet2, "conv1d2")


class MambaBlock3(MambaBlockBase):
    def __init__(self, args): super().__init__(args, PSConvNet3, "conv1d3")


# ==============================================================================
# Residual Blocks
# ==============================================================================
class ResidualBlockBase(nn.Module):
    def __init__(self, args, mixer_class):
        super().__init__()
        self.mixer = mixer_class(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x


class ResidualBlock1(ResidualBlockBase):
    def __init__(self, args): super().__init__(args, MambaBlock1)


class ResidualBlock2(ResidualBlockBase):
    def __init__(self, args): super().__init__(args, MambaBlock2)


class ResidualBlock3(ResidualBlockBase):
    def __init__(self, args): super().__init__(args, MambaBlock3)


# ==============================================================================
# Multi-Representation Fusion
# ==============================================================================
class MultiRepresentationFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Wa = nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)
        self.Wb = nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)
        self.Wc = nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)

    def forward(self, y3, y5, y7):
        Ya = torch.sigmoid(self.Wa * y3)
        Yb = torch.sigmoid(self.Wb * y5)
        Yc = torch.sigmoid(self.Wc * y7)
        Y_combined = F.softmax(torch.cat([Ya, Yb, Yc], dim=-1), dim=-1)
        Y_combined = Y_combined.view(Y_combined.size(0), Y_combined.size(1), 3, -1)
        return Y_combined[:, :, 0] * y3 + Y_combined[:, :, 1] * y5 + Y_combined[:, :, 2] * y7


# ==============================================================================
# S_Pretrain_emb_Mamba_ISM: Full Classifier
# ==============================================================================
class S_Pretrain_emb_Mamba_ISM(nn.Module):
    """Mamba-based peptide toxicity classifier with multi-scale SSM branches."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_inner = int(args.expand * args.d_model)
        if args.dt_rank == 'auto':
            args.dt_rank = math.ceil(args.d_model / 16)
        self.num_class = args.num_class

        self.layers1 = nn.ModuleList([ResidualBlock1(args) for _ in range(args.n_layer)])
        self.layers2 = nn.ModuleList([ResidualBlock2(args) for _ in range(args.n_layer)])
        self.layers3 = nn.ModuleList([ResidualBlock3(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)
        self.fusion_module = MultiRepresentationFusion(dim=args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

        seq_len = 50
        self.classifier = nn.Sequential(
            nn.Linear(args.d_model * seq_len, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, self.num_class),
        )

    def forward(self, input_ids):
        embedded = input_ids[0]
        x = embedded.clone()
        for layer in self.layers1:
            x_1 = layer(x)
        x_1 = self.norm_f(x_1)
        for layer in self.layers2:
            x_2 = layer(x)
        x_2 = self.norm_f(x_2)
        for layer in self.layers3:
            x_3 = layer(x)
        x_3 = self.norm_f(x_3)
        x = self.fusion_module(x_1, x_2, x_3)
        return torch.flatten(x, start_dim=1)

    def train_model(self, x):
        features = self.forward(x)
        return self.classifier(features)
