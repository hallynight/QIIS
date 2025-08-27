# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Based on Vision Transformer (ViT) in PyTorch by Ross Wightman

INTERPOLATE_MODE = 'bilinear'
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv
import numpy as np
from einops import rearrange as o_rearrange
from .quaternion import QuaternionTransposeConv, QuaternionConv,QuaternionBatchNorm2d


def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()


BatchNorm2d = nn.BatchNorm2d
_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        # 'num_classes': 1000,
        **kwargs
    }


def sep_prompt(x, prompt_length):
    prompt = x[:, :prompt_length, :]
    x = x[:, prompt_length:, :]
    return prompt, x


default_cfgs = {
    # patch models (weights from official Google JAX impl)
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_base_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
    ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),

    # patch models, imagenet21k (weights from official Google JAX impl)
    'vit_tiny_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        num_classes=21843),
    'vit_huge_patch14_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843),

    # SAM trained models (https://arxiv.org/abs/2106.01548)
    'vit_base_patch32_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz'),
    'vit_base_patch16_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz'),

    # deit models (FB weights)
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0),
    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    # ViT ImageNet-21K-P pretraining by MILL
    'vit_base_patch16_224_miil_in21k': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    'vit_base_patch16_224_miil': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm'
            '/vit_base_patch16_224_1k_miil_84_4.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear',
    ),
}


class Attention(nn.Module):
    def __init__(self, chan_nheads, resolution, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dim = dim
        self.resolution = resolution
        pixel_no = int(resolution[0] * resolution[1])
        self.pixel_no = pixel_no

        self.chan_nheads = chan_nheads
        self.token_trans = nn.Linear(dim, pixel_no)
        self.token_trans1 = nn.Linear(pixel_no, dim)
        # self.chan_q = nn.Linear(pixel_no, pixel_no, bias=qkv_bias)
        # self.chan_kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        chan_head_dim = self.pixel_no // self.chan_nheads
        self.chan_scale = chan_head_dim ** -0.5

        self.qfusion = QuaternionConv(4, 8, kernel_size=3, padding=1, stride=1,operation='convolution1d')
        self.deqfusion = QuaternionTransposeConv(8, 4, kernel_size=3, padding=1, stride=1,operation='convolution1d')
        self.qbn = nn.BatchNorm1d(4)
        self.qge = nn.GELU()
        self.qfto = nn.Conv1d(4, 1, kernel_size=1, bias=False)

    def forward(self, x, task_prompts):
        ori_task_prompts = task_prompts 

        ori_x = x 
        x = torch.cat([task_prompts, x], dim=1)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        raw_spa_attn = (q @ k.transpose(-2, -1))
        attn = raw_spa_attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        raw_spa_attn = raw_spa_attn, attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, task_no+1+HxW, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        task_prompts, x = sep_prompt(x, -self.pixel_no)

        # channel-wise attention
        chan_x = ori_x[:, -self.pixel_no:, :]

        chan_prompts = self.token_trans(ori_task_prompts)
        _, nT, _ = chan_prompts.shape

        # Original token as QKV
        q = chan_prompts
        k = chan_x.permute(0, 2, 1)
        v = k

        # multi-head attention
        nh = nw = int(np.sqrt(self.chan_nheads))
        win_h = self.resolution[0] // nh
        win_w = self.resolution[1] // nw
        q = rearrange(q, 'b t (nh h nw w) -> b (nh nw) t (h w)', nh=nh, nw=nw, h=win_h, w=win_w)
        k = rearrange(k, 'b t (nh h nw w) -> b (nh nw) t (h w)', nh=nh, nw=nw, h=win_h, w=win_w)
        v = rearrange(v, 'b t (nh h nw w) -> b (nh nw) t (h w)', nh=nh, nw=nw, h=win_h, w=win_w)

        raw_chan_attn = (q @ k.transpose(-2, -1))
        attn = raw_chan_attn * self.chan_scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        chan_x = (attn @ v)  # (B, H, task_no, HxW)
        chan_x = rearrange(chan_x, 'b (nh nw) t (h w) -> b t (nh h nw w)', nh=nh, nw=nw, h=win_h, w=win_w)
        raw_chan_attn = rearrange(raw_chan_attn, 'b (nh nw) t c -> b t c nh nw', nh=nh, nw=nw)
        raw_chan_attn = raw_chan_attn, rearrange(attn, 'b (nh nw) t c -> b t c nh nw', nh=nh, nw=nw)

        ch_prompts = self.token_trans1(chan_prompts)
        zero_fea = torch.zeros(task_prompts.shape).to(task_prompts.device)
        quater_para = torch.stack([zero_fea, ori_task_prompts, task_prompts, ch_prompts],dim=2).to(task_prompts.device)
        quater_out = torch.empty(task_prompts.shape).to(task_prompts.device)
        for idx, batch in enumerate(quater_para):
            fe = self.qfusion(batch)
            fe = self.deqfusion(fe)
            quater_out[idx] = self.qfto(self.qge(self.qbn(fe))).squeeze(dim=1)
        raw_attn = [raw_spa_attn, raw_chan_attn]
        return x, raw_attn, quater_out



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x1 = self.fn(x)
        return x1 + x


class Offset(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.p_conv = nn.Conv2d(dim, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=1)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)
        self.opt = nn.Conv2d(2 * self.kernel_size * self.kernel_size, dim * 2, kernel_size=3, padding=1, stride=1,
                             groups=2)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1), indexing="xy")
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride), indexing="xy")
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def forward(self, x):
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        p = self._get_p(offset, dtype)  # 1,18,107,140
        p = self.opt(p)
        return p


class ChlSpl(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
    ):
        super(ChlSpl, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))

        self.get_offset = Offset(dim=in_channels, kernel_size=3)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        offset = torch.empty(1, self.in_channels * 2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input: Tensor) -> Tensor:
        """
            input: Tensor[b,c,h,w]
        """
        offset_2 = self.get_offset(input)
        B, C, H, W = input.size()

        return deform_conv2d_tv(input, offset_2, self.weight, self.bias, stride=self.stride, padding=self.padding,
                                dilation=self.dilation)

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class Block(nn.Module):

    def __init__(self, chan_nheads, resolution, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, patch_size=16):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(chan_nheads, resolution, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()  # 随机丢弃防止过拟合  nn.Identity()什么都不做 输入什么输出什么
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, task_prompts):

        x_attn, attn_weight, task_prompts_attn = self.attn(self.norm1(x), self.norm1(task_prompts))

        x = x + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        task_prompts = task_prompts + self.drop_path(task_prompts_attn)
        task_prompts = task_prompts + self.drop_path(self.mlp(self.norm2(task_prompts)))

        return x, attn_weight, task_prompts




class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)
        return x


class QuaternionFusionResidual(nn.Module):
    def __init__(self, inputchannel, outputchannel, fn):
        super().__init__()
        self.conv1 = QuaternionConv(inputchannel, outputchannel, kernel_size=3, stride=1, padding=1)
        self.fn = fn

    def forward(self, x):
        x1 = self.fn(x)
        x = self.conv1(x)
        return x1 + x



class QuaternionFusion(nn.Module):
    def __init__(self, inputchannel, outputchannel, use_qbn, drop_rate, gamma=1.0, use_beta=True):
        super(QuaternionFusion,self).__init__()
        self.act_fn = nn.GELU()
        self.use_qbn = use_qbn

        self.conv1 = QuaternionConv(inputchannel, outputchannel,kernel_size=3, stride=1, padding=1)
        self.conv2 = QuaternionConv(inputchannel, inputchannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = QuaternionBatchNorm2d(inputchannel, gamma_init=gamma,
                                         beta_param=use_beta) if self.use_qbn else nn.BatchNorm2d(16)
        self.dp = nn.Dropout(p=drop_rate)

        self.conv3 = QuaternionConv(inputchannel, outputchannel, kernel_size=3, stride=1, padding=1)
        self.bn3 = QuaternionBatchNorm2d(outputchannel, gamma_init=gamma,
                                         beta_param=use_beta) if self.use_qbn else nn.BatchNorm2d(16)


    def forward(self, x):
        x_o = self.conv1(x)

        x = self.bn2(self.conv2(x))
        x = self.act_fn(x)

        x = self.self.dp(self.conv3(x))
        x = self.bn3(x + x_o)
        x = self.act_fn(x)
        return x



class TaskPrompter(nn.Module):
    """ TaskPrompter built upon ViT
    """

    def __init__(self, p, select_list, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, chan_nheads=1, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):

        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))  # one cls token from pretrained weights on ImageNet
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.resolution = [int(img_size[0] / patch_size), int(img_size[1] / patch_size)]
        self.blocks = nn.Sequential(*[
            Block(
                chan_nheads,
                self.resolution,
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.select_list = select_list
        self.num_layers = 4
        assert len(select_list) == self.num_layers - 1
        task_no = len(p.TASKS.NAMES)
        self.resolution = [int(img_size[0] / patch_size), int(img_size[1] / patch_size)]
        pixel_no = int(self.resolution[0] * self.resolution[1])
        self.pixel_no = pixel_no
        self.p = p

        # multi-task prompt learning
        self.prompt_len = p.prompt_len
        self.prompts_len = task_no * p.prompt_len
        self.task_prompts = nn.Parameter(torch.ones(self.prompts_len, embed_dim))
        trunc_normal_(self.task_prompts, mean=1., std=1.)

        self.cs_fea = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_chans, in_chans, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(in_chans),
                Residual(nn.Sequential(
                    ChlSpl(in_chans, in_chans, (1, 3), 1, 0),
                    nn.GELU(),
                    nn.BatchNorm2d(in_chans)
                )),
            ) for i in range(3)],
        )
        dw_stride = patch_size // 4
        self.cs_conv1 = nn.Conv2d(in_chans, 64 * 4, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.cs_bn1 = nn.BatchNorm2d(64 * 4)
        self.cs_act1 = nn.GELU()
        self.cs_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = int(64 * 4)
        trans_dw_stride = patch_size // 4

        # self.trans_patch_conv = nn.Conv2d(stage_1_channel, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.cs_conf = FCUDown(inplanes=stage_1_channel, outplanes=embed_dim, dw_stride=dw_stride)

        self.fea_fuse = nn.ModuleList()
        if p.use_ctr:
            self.ctr_attn_conv = nn.ModuleList()
        self.fea_decode_spa = nn.ModuleList()
        self.fea_decode_chan = nn.ModuleList()
        self.fea_decode_pat = nn.ModuleList()
        self.fea_decode_cross = nn.ModuleList()
        attn_conv_expansion = 1
        prompt_dim = num_heads * p.prompt_len
        tar_dim = p.embed_dim
        final_embed_dim = p.final_embed_dim
        for i_layer in range(self.num_layers):
            self.fea_fuse.append(nn.ModuleDict())
            if p.use_ctr:
                self.ctr_attn_conv.append(nn.ModuleDict())
            self.fea_decode_spa.append(nn.ModuleDict())
            self.fea_decode_chan.append(nn.ModuleDict())
            self.fea_decode_pat.append(nn.ModuleDict())
            self.fea_decode_cross.append(nn.ModuleDict())
            for task in p.TASKS.NAMES:
                # self.e1 = QuaternionConv(final_embed_dim, final_embed_dim, kernel_size=3, padding=1)
                # self.fea_fuse[i_layer][task] = nn.Sequential(nn.Conv2d(tar_dim * 2, final_embed_dim, kernel_size=1),
                #                                              nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=3,
                #                                                        padding=1), BatchNorm2d(final_embed_dim),
                #                                              nn.GELU(),
                #                                              nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=1))

                self.fea_fuse[i_layer][task] = nn.Sequential(
                    QuaternionConv(tar_dim * 4, final_embed_dim, kernel_size=3, padding=1, stride=1),
                    Residual(nn.Sequential(
                                QuaternionConv(final_embed_dim, final_embed_dim, kernel_size=3, padding=1, stride=1),
                                QuaternionBatchNorm2d(final_embed_dim, gamma_init=1.0,beta_param=True),
                                nn.GELU())),
                    QuaternionBatchNorm2d(final_embed_dim, gamma_init=1.0, beta_param=True),
                    nn.GELU(),
                    # QuaternionConv(final_embed_dim, final_embed_dim, kernel_size=1, padding=1, stride=1)
                )

                # self.fea_fuse[i_layer][task] = QuaternionFusion(tar_dim * 4, final_embed_dim, use_qbn=True, drop_rate=drop_rate)

                if p.use_ctr:
                    self.ctr_attn_conv[i_layer][task] = nn.Sequential(
                        nn.Conv2d(prompt_dim, prompt_dim * attn_conv_expansion, kernel_size=1, padding=0), nn.GELU(),
                        nn.Conv2d(prompt_dim * attn_conv_expansion, 1, kernel_size=1, padding=0))
                self.fea_decode_spa[i_layer][task] = nn.Sequential(
                    nn.Conv2d(embed_dim, tar_dim, kernel_size=1, padding=0))
                self.fea_decode_chan[i_layer][task] = nn.Sequential(
                    nn.Conv2d(embed_dim, tar_dim, kernel_size=1, padding=0))
                self.fea_decode_cross[i_layer][task] = nn.Sequential(
                    nn.Conv2d(embed_dim, tar_dim, kernel_size=1, padding=0))
                self.fea_decode_pat[i_layer][task] = nn.Sequential(
                    nn.Conv2d(embed_dim, tar_dim, kernel_size=1, padding=0))

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward(self, x):
        x_o = x
        x_cs = self.cs_fea(x)
        x_base = self.cs_maxpool(self.cs_act1(self.cs_bn1(self.cs_conv1(x_cs))))
        x_cs = self.cs_conf(x_base)
        x = self.patch_embed(x)  # shape: [2,1008,1024]
        x = x + x_cs
        x = self.pos_drop(x + self.pos_embed[:, 1:])  # shape: [2,1008,1024]
        # put prompts in it
        task_prompts = self.task_prompts[None].expand(x.shape[0], -1, -1)  # shape[2,4,1024]
        # multi-scale backbone feature
        all_tasks = self.p.TASKS.NAMES  # all_tasks:['semseg','depth', 'normals', 'edge']
        task_fea = {task: 0 for task in all_tasks}  # {'depth':0,'edge':0, 'normals':0, 'semseg':0}
        info = {}  # pass information through the pipeline

        for idx, blk in enumerate(self.blocks):
            x, attn_weight, task_prompts = blk(x, task_prompts)
            if idx + 1 in self.select_list:
                # extract task-specific feature at this layer
                il = np.sum(idx >= (np.array(self.select_list) - 1)) - 1  # [0,1,2]
                _cur_task_fea, info = self.cal_task_feature(x, attn_weight, il, info,False,x_o)
                for t_idx, task in enumerate(self.p.TASKS.NAMES):
                    task_fea[task] += _cur_task_fea[task]

        x = self.norm(x)

        # extract task-specific feature at the last layer
        il = self.num_layers - 1
        _cur_task_fea, info = self.cal_task_feature(x, attn_weight, il, info,True,x_o)
        for t_idx, task in enumerate(self.p.TASKS.NAMES):
            task_fea[task] += _cur_task_fea[task]
            task_fea[task] = F.interpolate(task_fea[task], scale_factor=4, mode=INTERPOLATE_MODE)

        return task_fea, info

    def cal_task_feature(self, x, attn_weight, il, info,vis_flag,ori_fig):
        ''' Calculate task feature at this layer
        '''
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.resolution[0], w=self.resolution[1])
        task_fea = {}
        spa_attn, chan_attn = attn_weight
        chan_task_fea = {}
        chan_cross_fea = {}
        chan_patch_fea = {}
        spa_attn, softmax_spa_attn = spa_attn
        chan_attn, softmax_chan_attn = chan_attn


        for t_idx, task in enumerate(self.p.TASKS.NAMES):
            # task feature extraction with spatial attention
            cur_attn_weight = spa_attn[:, :, t_idx * self.prompt_len:(t_idx + 1) * self.prompt_len, :]
            cur_attn_weight = cur_attn_weight[:, :, :, self.prompts_len:]
            cur_attn_weight = rearrange(cur_attn_weight, 'b nh np (h w) -> b (nh np) h w', h=self.resolution[0],
                                        w=self.resolution[1])

            bs, nheads = cur_attn_weight.shape[0:2]
            cur_task_fea = []
            head_channel_no = self.embed_dim // nheads
            for hea in range(nheads):
                cur_head_attn = cur_attn_weight[:, hea:hea + 1, :, :]
                cur_task_fea.append(cur_head_attn * x[:, head_channel_no * hea:head_channel_no * (hea + 1), :, :])
            # 16个 2 64 28 36  ->  2 1024 28 36
            cur_task_fea = torch.cat(cur_task_fea, dim=1) + x
            cur_task_fea = self.fea_decode_spa[il][task](cur_task_fea)

            task_fea[task] = cur_task_fea

            # cross task feature
            cro_attn_weight = spa_attn[:, :, :, t_idx * self.prompt_len:(t_idx + 1) * self.prompt_len]
            cro_attn_weight = cro_attn_weight[:, :, self.prompts_len:, :].transpose(2, 3)
            cro_attn_weight = rearrange(cro_attn_weight, 'b nh np (h w) -> b (nh np) h w', h=self.resolution[0],
                                        w=self.resolution[1])
            cro_task_fea = []
            for hea in range(nheads):
                cro_head_attn = cro_attn_weight[:, hea:hea + 1, :, :]
                cro_task_fea.append(cro_head_attn * x[:, head_channel_no * hea:head_channel_no * (hea + 1), :, :])

            cro_task_fea = torch.cat(cro_task_fea, dim=1) + x
            cro_task_fea = self.fea_decode_cross[il][task](cro_task_fea)

            chan_cross_fea[task] = cro_task_fea

            # task feature extraction with channel-wise attention
            # chan_attn: (b t c nh nw)
            cur_attn_weight = chan_attn[:, t_idx]  # (b, c, nh, nw)
            bs, _, nh, nw = cur_attn_weight.shape
            nheads = nh * nw
            win_h = self.resolution[0] // nh
            win_w = self.resolution[1] // nw
            cur_task_fea = []

            for h_idx in range(nh):
                cur_row = []
                for w_idx in range(nw):
                    _patch = x[:, :, h_idx * win_h:(h_idx + 1) * win_h, w_idx * win_w:(w_idx + 1) * win_w]
                    _attn = cur_attn_weight[:, :, h_idx, w_idx]
                    _attn = _attn.unsqueeze(-1).unsqueeze(-1)
                    cur_row.append(_attn * _patch)
                cur_task_fea.append(torch.cat(cur_row, dim=3))
            cur_task_fea = torch.cat(cur_task_fea, dim=2) + x
            cur_task_fea = self.fea_decode_chan[il][task](cur_task_fea)
            chan_task_fea[task] = cur_task_fea

            cur_task_fea = self.fea_decode_pat[il][task](x)
            chan_patch_fea[task] = cur_task_fea
            # combined_fea = torch.cat([task_fea[task], chan_task_fea[task]], dim=1)
            # zero_fea = torch.zeros(chan_patch_fea[task].shape).to(chan_patch_fea[task].device)
            # combined_fea = torch.stack([task_fea[task], chan_task_fea[task]], dim=1)

            #             combined_fea = torch.empty(cur_task_fea.shape[0],2 * cur_task_fea.shape[1],cur_task_fea.shape[2],cur_task_fea.shape[3]).to(cur_task_fea.device)
            #             for i in range(0, cur_task_fea.shape[1], 2):
            #                 channels_tensor1 = chan_task_fea[task][:, i:i + 2, :, :]
            #                 channels_tensor2 = task_fea[task][:, i:i + 2, :, :]

            #                 combined_channels = torch.cat((channels_tensor1, channels_tensor2), dim=1)

            #                 combined_fea[:, 2 * i:2 * i + 4, :, :] = combined_channels

            batch_size, channels, height, width = chan_task_fea[task].shape
            combined_fea = torch.empty(batch_size, channels * 4, height, width, device=chan_task_fea[task].device)

            combined_fea[:, 0::4, :, :] = chan_cross_fea[task]
            combined_fea[:, 1::4, :, :] = chan_task_fea[task]
            combined_fea[:, 2::4, :, :] = task_fea[task]
            combined_fea[:, 3::4, :, :] = chan_patch_fea[task]

            if vis_flag:
                show_fig(chan_cross_fea[task][0].cpu(),ori_fig[0].cpu())
                show_fig(chan_task_fea[task][0].cpu(),ori_fig[0].cpu())
                show_fig(task_fea[task][0].cpu(),ori_fig[0].cpu())
                show_fig(chan_patch_fea[task][0].cpu(),ori_fig[0].cpu())
                test = chan_cross_fea[task] + chan_task_fea[task] + task_fea[task] + chan_patch_fea[task]
                # show_fig(test[0].cpu(),ori_fig[0].cpu())

            combined_fea = self.fea_fuse[il][task](combined_fea)
            task_fea[task] = combined_fea
            if vis_flag:
                show_fig(combined_fea[0].cpu(),ori_fig[0].cpu())
                print("ok")

        # Cross-Task Reweighting: multi-task intereaction based on affinity map of task prompts
        if self.p.use_ctr:
            new_task_fea = {}
            assert self.prompt_len == 1
            for t_idx, task in enumerate(self.p.TASKS.NAMES):
                cur_attn_weight = spa_attn[:, :, t_idx:(t_idx + 1), :self.prompts_len]  # (B, nH, 1, nT)
                cur_attn_weight = self.ctr_attn_conv[il][task](cur_attn_weight)
                new_task_fea[task] = sum(
                    [cur_attn_weight[:, :, :, target_idx:target_idx + 1] * task_fea[target_task] for
                     target_idx, target_task in enumerate(self.p.TASKS.NAMES)])
            task_fea = new_task_fea

        return task_fea, info


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):

            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):

            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    # model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    # if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
    #     model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
    #     model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        # for j, b in enumerate(block.blocks.children()):
        #     b_prefix = f'{prefix}Transformer/encoderblock_{i}/blocks_{j}/'
        #     for inde, bo in enumerate(b.children()):
        #         print("a")

        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_task_prompter(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        TaskPrompter, variant, pretrained,
        default_cfg=default_cfg,
        # representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


def taskprompter_vit_large_patch16_384(pretrained=False, **kwargs):
    """ Based on ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(select_list=range(6, 24, 6), patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                        chan_nheads=kwargs['p'].chan_nheads, **kwargs)
    model = _create_task_prompter('vit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


def taskprompter_vit_base_patch16_384(pretrained=False, **kwargs):
    """ Based on ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(select_list=range(3, 12, 3), patch_size=16, embed_dim=768, depth=12, num_heads=12,
                        chan_nheads=kwargs['p'].chan_nheads, **kwargs)
    model = _create_task_prompter('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


class ConvHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.mt_proj = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1), BatchNorm2d(in_channels),
                                     nn.GELU())
        trunc_normal_(self.mt_proj[0].weight, std=0.02)

        self.linear_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.linear_pred(self.mt_proj(x))


class DEConvHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.mt_proj = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2, padding=0), BatchNorm2d(in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1), BatchNorm2d(in_channels // 2), nn.GELU()
        )

        self.linear_pred = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)
        trunc_normal_(self.mt_proj[0].weight, std=0.02)
        trunc_normal_(self.mt_proj[3].weight, std=0.02)
        trunc_normal_(self.linear_pred.weight, std=0.02)

    def forward(self, x):
        return self.linear_pred(self.mt_proj(x))

def show_fig(feature_map,img_tensor):
    import matplotlib.pyplot as plt
    import torchvision.transforms as T
    aggregated_features = feature_map.mean(dim=0) 


    resized_features = T.Resize((512, 512))(aggregated_features.unsqueeze(0))
    resized_features = resized_features.squeeze(0)

    resized_features = (resized_features - resized_features.min()) / (
            resized_features.max() - resized_features.min())

    heatmap_np = resized_features.numpy()

    img_np = img_tensor.permute(1, 2, 0).numpy() 

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(img_np) 
    ax.imshow(heatmap_np, cmap='jet', alpha=0.7) 
    ax.axis('off')
    plt.show()

def show_fusion_fea(feature_map,img_np):
    import matplotlib.pyplot as plt
    import torchvision.transforms as T


    aggregated_features = feature_map.mean(dim=0)

    resized_features = T.Resize((512, 512))(aggregated_features.unsqueeze(0))
    resized_features = resized_features.squeeze(0) 

    resized_features = (resized_features - resized_features.min()) / (
            resized_features.max() - resized_features.min())

    heatmap_np = resized_features.numpy() 

    img_tensor = torch.rand(3, 512, 512)
    img_np = img_tensor.permute(1, 2, 0).numpy() 

    fig, ax = plt.subplots(figsize=(10, 10),constrained_layout=True)
    ax.imshow(img_np)  
    ax.imshow(heatmap_np, cmap='jet', alpha=0.5)
    ax.axis('off') 
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0) 
    plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0) 
    plt.show()