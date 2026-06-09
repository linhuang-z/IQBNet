# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
}



class BottleneckFreqSpatialRefinement(nn.Module):
    def __init__(self, hidden_size, init_sigma=1.0, reduction=4):
        super(BottleneckFreqSpatialRefinement, self).__init__()
        self.hidden_size = hidden_size
        self.eps = 1e-6

        # Frequency branch
        self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma)))
        self.freq_norm = nn.LayerNorm(hidden_size)

        # Multi-scale depthwise spatial branch
        self.dwconv3 = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=3,
            padding=1, groups=hidden_size, bias=False
        )
        self.dwconv5 = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=5,
            padding=2, groups=hidden_size, bias=False
        )
        self.spatial_fuse = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.GELU(),
        )

        # Axial spatial gating
        self.axis_h = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=(1, 7),
            padding=(0, 3), groups=hidden_size, bias=False
        )
        self.axis_w = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=(7, 1),
            padding=(3, 0), groups=hidden_size, bias=False
        )
        self.axis_gate = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Token-wise adaptive fusion
        mid = max(hidden_size // reduction, 32)
        self.token_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, mid),
            nn.GELU(),
            nn.Linear(mid, hidden_size),
            nn.Sigmoid(),
        )

        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Start as identity for stable fine-tuning.
        self.gamma = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.dwconv3.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.dwconv5.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.axis_h.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.axis_w.weight, mode="fan_out", nonlinearity="relu")
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _make_gaussian_kernel(self, sigma):
        coords = torch.arange(self.hidden_size, device=sigma.device).float()
        coords = coords - self.hidden_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / (g.sum() + self.eps)
        return g.view(1, 1, self.hidden_size)

    def _frequency_stability(self, x):
        sigma = torch.exp(self.log_sigma)
        kernel = self._make_gaussian_kernel(sigma)

        x_fft = torch.fft.fft(x, dim=-1)
        x_fft = torch.fft.fftshift(x_fft, dim=-1)
        x_fft = x_fft * kernel
        x_fft = torch.fft.ifftshift(x_fft, dim=-1)
        x_lp = torch.fft.ifft(x_fft, dim=-1).real

        stability = torch.abs(x_lp) / (torch.abs(x - x_lp) + self.eps)
        stability = torch.log1p(stability)
        stability = self.freq_norm(stability)
        return x_lp * torch.sigmoid(stability)

    def _spatial_context(self, x):
        B, N, D = x.size()
        h = int(np.sqrt(N))
        w = h
        if h * w != N:
            return x

        feat = x.transpose(1, 2).contiguous().view(B, D, h, w)
        local3 = self.dwconv3(feat)
        local5 = self.dwconv5(feat)
        local = self.spatial_fuse(torch.cat([local3, local5], dim=1))

        axis = self.axis_h(local) + self.axis_w(local)
        gate = self.axis_gate(axis)
        refined = local * gate
        refined = refined.flatten(2).transpose(1, 2).contiguous()
        return refined

    def forward(self, x):
        freq_feat = self._frequency_stability(x)
        spatial_feat = self._spatial_context(x)
        gate = self.token_gate(torch.cat([freq_feat, spatial_feat], dim=-1))
        fused = gate * spatial_feat + (1.0 - gate) * freq_feat
        fused = self.out_proj(self.out_norm(fused))
        return x + self.gamma * fused


LazyStrikeRefinement = BottleneckFreqSpatialRefinement

# ==========================================================================
# Innovation: CH-IQE-GQA
# Cross-Head Interactive Query Enhancement for GQA.
#
# This version makes Q interact with "other Q" explicitly:
#   1) Cross-head Q interaction: each query head talks to the other query heads.
#   2) Cross-group Q interaction: GQA query groups talk to the other query groups.
#   3) Q-X spatial/token modulation is kept only as an auxiliary branch.
# ==========================================================================
class QuerySpatialAttention(nn.Module):
    def __init__(self):
        super(QuerySpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, kernel_size=7, padding=3, padding_mode="reflect", bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        return self.sa(torch.cat([x_avg, x_max], dim=1))


class QueryChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(QueryChannelAttention, self).__init__()
        mid = max(dim // reduction, 16)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.ca(self.gap(x))


class QueryPixelAttention(nn.Module):
    def __init__(self, dim):
        super(QueryPixelAttention, self).__init__()
        self.pa = nn.Conv2d(
            2 * dim, dim, kernel_size=7, padding=3,
            padding_mode="reflect", groups=dim, bias=True
        )

    def forward(self, x, prior):
        return self.pa(torch.cat([x, prior], dim=1))


class CrossHeadQueryInteraction(nn.Module):

    def __init__(self, num_heads, head_dim, dropout_rate=0.0):
        super(CrossHeadQueryInteraction, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = num_heads * head_dim

        self.q_proj = nn.Linear(head_dim, head_dim, bias=True)
        self.k_proj = nn.Linear(head_dim, head_dim, bias=True)
        self.v_proj = nn.Linear(head_dim, head_dim, bias=True)
        self.out_proj = nn.Linear(head_dim, head_dim, bias=True)

        self.head_gate = nn.Sequential(
            nn.Linear(self.dim, max(self.dim // 8, 32)),
            nn.GELU(),
            nn.Linear(max(self.dim // 8, 32), num_heads),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(self.dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        # zero init keeps stable pretrained loading
        nn.init.zeros_(self.head_gate[-2].weight)
        nn.init.zeros_(self.head_gate[-2].bias)

    def forward(self, q):
        B, N, C = q.shape
        if C != self.dim:
            return q

        qh = q.view(B, N, self.num_heads, self.head_dim)
        q_proj = self.q_proj(qh)
        k_proj = self.k_proj(qh)
        v_proj = self.v_proj(qh)

        scores = torch.matmul(q_proj, k_proj.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        other_q = torch.matmul(attn, v_proj)
        other_q = self.out_proj(other_q)

        gate = self.head_gate(q).view(B, N, self.num_heads, 1)
        mixed = qh + gate * other_q
        mixed = mixed.contiguous().view(B, N, C)
        mixed = self.norm(mixed)
        return q + self.gamma * mixed


class CrossGroupQueryInteraction(nn.Module):

    def __init__(self, num_heads, head_dim, num_kv_heads=1, dropout_rate=0.0):
        super(CrossGroupQueryInteraction, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.group_heads = max(num_heads // max(num_kv_heads, 1), 1)
        self.dim = num_heads * head_dim
        self.group_dim = self.group_heads * head_dim

        self.q_proj = nn.Linear(self.group_dim, self.group_dim, bias=True)
        self.k_proj = nn.Linear(self.group_dim, self.group_dim, bias=True)
        self.v_proj = nn.Linear(self.group_dim, self.group_dim, bias=True)
        self.out_proj = nn.Linear(self.group_dim, self.group_dim, bias=True)

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(self.dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, q):
        B, N, C = q.shape
        if C != self.dim or self.num_kv_heads <= 1:
            return q

        q_group = q.view(B, N, self.num_kv_heads, self.group_heads, self.head_dim)
        group_tokens = q_group.contiguous().view(B, N, self.num_kv_heads, self.group_dim)

        q_proj = self.q_proj(group_tokens)
        k_proj = self.k_proj(group_tokens)
        v_proj = self.v_proj(group_tokens)

        scores = torch.matmul(q_proj, k_proj.transpose(-1, -2))
        scores = scores / math.sqrt(self.group_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        other_group = torch.matmul(attn, v_proj)
        other_group = self.out_proj(other_group)
        other_group = other_group.view(B, N, self.num_kv_heads, self.group_heads, self.head_dim)

        mixed = q_group + other_group
        mixed = mixed.contiguous().view(B, N, C)
        mixed = self.norm(mixed)
        return q + self.gamma * mixed


class InteractiveQueryEnhancement(nn.Module):
    def __init__(self, dim, num_heads, head_dim, num_kv_heads=1, reduction=8, dropout_rate=0.0):
        super(InteractiveQueryEnhancement, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads

        # Main branch: Q interacts with other Q heads.
        self.cross_head_q = CrossHeadQueryInteraction(
            num_heads=num_heads,
            head_dim=head_dim,
            dropout_rate=dropout_rate,
        )

        # GQA branch: Q groups interact with other Q groups.
        self.cross_group_q = CrossGroupQueryInteraction(
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            dropout_rate=dropout_rate,
        )

        # Auxiliary branch: Q-X interaction retained as spatial/token prior.
        self.sa = QuerySpatialAttention()
        self.ca = QueryChannelAttention(dim, reduction=reduction)
        self.pa = QueryPixelAttention(dim)
        self.fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.norm = nn.LayerNorm(dim)
        self.sigmoid = nn.Sigmoid()
        self.gamma_x = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def _tokens_to_map(self, x):
        B, N, C = x.shape
        h = int(np.sqrt(N))
        if h * h != N:
            return None
        return x.transpose(1, 2).contiguous().view(B, C, h, h)

    def _qx_spatial_modulation(self, q, x):
        q_map = self._tokens_to_map(q)
        x_map = self._tokens_to_map(x)
        if q_map is None or x_map is None:
            return q

        initial = q_map + x_map
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn = self.sigmoid(self.pa(initial, cattn + sattn))

        enhanced = initial + pattn * q_map + (1.0 - pattn) * x_map
        enhanced = self.fuse(enhanced)
        enhanced = enhanced.flatten(2).transpose(1, 2).contiguous()
        enhanced = self.norm(enhanced)
        return q + self.gamma_x * enhanced

    def forward(self, q, x):
        # 1) Q interacts with other Q heads.
        q = self.cross_head_q(q)

        # 2) Q query groups interact with other query groups for GQA.
        q = self.cross_group_q(q)

        # 3) X is only an auxiliary prior, not the main interaction object.
        q = self._qx_spatial_modulation(q, x)
        return q


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis

        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.num_kv_heads = config.transformer.get("num_kv_heads", 2)
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        self.num_groups = self.num_attention_heads // self.num_kv_heads
        self.kv_all_head_size = self.num_kv_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.kv_all_head_size)
        self.value = Linear(config.hidden_size, self.kv_all_head_size)
        self.out = Linear(config.hidden_size, config.hidden_size)

        self.query_enhance = InteractiveQueryEnhancement(
            dim=self.all_head_size,
            num_heads=self.num_attention_heads,
            head_dim=self.attention_head_size,
            num_kv_heads=self.num_kv_heads,
            reduction=8,
            dropout_rate=config.transformer["attention_dropout_rate"],
        )

        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x, num_heads):
        new_x_shape = x.size()[:-1] + (num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_query_layer = self.query_enhance(mixed_query_layer, hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, self.num_attention_heads)
        key_layer = self.transpose_for_scores(mixed_key_layer, self.num_kv_heads)
        value_layer = self.transpose_for_scores(mixed_value_layer, self.num_kv_heads)

        if self.num_kv_heads != self.num_attention_heads:
            key_layer = key_layer.repeat_interleave(self.num_groups, dim=1)
            value_layer = value_layer.repeat_interleave(self.num_groups, dim=1)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch and position embeddings."""
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (
                img_size[0] // 16 // grid_size[0],
                img_size[1] // 16 // grid_size[1],
            )
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (
                (img_size[0] // patch_size_real[0])
                * (img_size[1] // patch_size_real[1])
            )
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor,
            )
            in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(
                self.hidden_size, self.hidden_size
            ).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(
                self.hidden_size, self.hidden_size
            ).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(
                self.hidden_size, self.hidden_size
            ).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(
                self.hidden_size, self.hidden_size
            ).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            # Convert pretrained MHA K/V weights to GQA K/V weights by group averaging.
            if hasattr(self.attn, "num_kv_heads") and self.attn.num_kv_heads != self.attn.num_attention_heads:
                num_heads = self.attn.num_attention_heads
                num_kv_heads = self.attn.num_kv_heads
                num_groups = self.attn.num_groups
                head_size = self.attn.attention_head_size
                hidden_size = self.hidden_size
                kv_size = self.attn.kv_all_head_size

                key_weight = key_weight.view(num_heads, head_size, hidden_size)
                key_weight = key_weight.view(num_kv_heads, num_groups, head_size, hidden_size).mean(dim=1)
                key_weight = key_weight.view(kv_size, hidden_size)

                key_bias = key_bias.view(num_heads, head_size)
                key_bias = key_bias.view(num_kv_heads, num_groups, head_size).mean(dim=1)
                key_bias = key_bias.view(kv_size)

                value_weight = value_weight.view(num_heads, head_size, hidden_size)
                value_weight = value_weight.view(num_kv_heads, num_groups, head_size, hidden_size).mean(dim=1)
                value_weight = value_weight.view(kv_size, hidden_size)

                value_bias = value_bias.view(num_heads, head_size)
                value_bias = value_bias.view(num_kv_heads, num_groups, head_size).mean(dim=1)
                value_bias = value_bias.view(kv_size)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)

            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

        init_sigma = getattr(config, "lazystrike_init_sigma", 1.0)
        self.lazystrike = LazyStrikeRefinement(
            hidden_size=config.hidden_size,
            init_sigma=init_sigma,
        )

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        encoded = self.lazystrike(encoded)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        upsampling_layer = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling_layer)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4 - self.config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config["decoder_channels"][-1],
            out_channels=config["n_classes"],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True)
            )
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"])
            )

            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"])
            )
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"])
            )

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print("load_pretrained: grid-size from %s to %s" % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # BFSR is newly added and has no corresponding ViT pretrained weights.
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(res_weight["conv_root/kernel"], conv=True)
                )
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


CONFIGS = {
    "ViT-B_16": configs.get_b16_config(),
    "ViT-B_32": configs.get_b32_config(),
    "ViT-L_16": configs.get_l16_config(),
    "ViT-L_32": configs.get_l32_config(),
    "ViT-H_14": configs.get_h14_config(),
    "R50-ViT-B_16": configs.get_r50_b16_config(),
    "R50-ViT-L_16": configs.get_r50_l16_config(),
    "testing": configs.get_testing(),
}