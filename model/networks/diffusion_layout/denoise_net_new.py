import math
from random import random
from functools import partial
from collections import namedtuple
from tkinter.messagebox import NO
from tkinter.tix import Tree

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# helpers functions

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None, pool=False):
    if pool:
        return nn.Sequential(
            # nn.Upsample(scale_factor = 2, mode = 'nearest'),

            nn.Conv1d(dim, default(dim_out, dim), 1)
        )
    else:
        return nn.Identity()


def Downsample(dim, dim_out=None, pool=False):
    if pool:
        return nn.Sequential(
            # return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

            nn.Conv1d(dim, default(dim_out, dim), 1)
        )
    else:
        return nn.Identity()


class WeightStandardizedConv2d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class PreNormCross(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, context):
        x = self.norm(x)
        return self.fn(x, context)


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 1, padding=0)  # 3-->1
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            if len(time_emb.shape) == 2:
                time_emb = rearrange(time_emb, 'b c -> b c 1')
            else:
                # BxNxC --> BxCxN
                time_emb = torch.permute(time_emb, (0, 2, 1))  # context case
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h=self.heads)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)


class AttentionCross(nn.Module):
    def __init__(self, dim, context_dim=None, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        if context_dim is None:
            context_dim = dim

        self.to_q = nn.Conv1d(dim, hidden_dim * 1, 1, bias=False)
        self.to_kv = nn.Conv1d(context_dim, hidden_dim * 2, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x, context):  # =None):
        b, c, n = x.shape

        q = self.to_q(x)
        # if context is None:
        #     context = x
        context = torch.permute(context, (0, 2, 1))
        kv = self.to_kv(context).chunk(2, dim=1)
        q = rearrange(q, 'b (h c) n -> b h c n', h=self.heads)
        k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), kv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h=self.heads)
        return self.to_out(out)


class Unet1D(nn.Module):
    def __init__(
            self,
            dim=256,  #
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            self_condition=False,
            bbox_separate=False,
            merge_bbox=False,
            class_dim=21,
            translation_dim=3,
            size_dim=3,
            angle_dim=1,
            conditioning_key='crossattn',
            crossattn_dim=0,
            concat_dim=0,
            modulate_time_context_instanclass=False,
            relation_condition=False,
            text_dim=256,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        self.bbox_separate = bbox_separate
        self.class_dim = class_dim
        self.translation_dim = translation_dim
        self.size_dim = size_dim
        self.angle_dim = angle_dim
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.conditioning_key = conditioning_key
        self.context_dim = crossattn_dim if self.conditioning_key == "crossattn" else concat_dim
        self.modulate_time_context_instanclass = modulate_time_context_instanclass
        self.rel_condition = relation_condition
        self.text_dim = text_dim
        if self.bbox_separate:
            self.bbox_embedf = Unet1D._encoder_mlp(dim, self.bbox_dim)

            input_channels = dim
            print('separate unet1d encoder of /translation/size/angle')

        else:
            input_channels = channels
            print('unet1d encoder of all object properties')

        init_dim = default(init_dim, dim)
        self.init_layer = nn.Conv1d(input_channels, init_dim, 1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in + self.context_dim, dim_in, time_emb_dim=time_dim) if self.conditioning_key in [
                    "concat", "hybrid"] else block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                AttentionCross(dim_in, context_dim=self.context_dim) if self.conditioning_key in ["crossattn",
                                                                                                  "hybrid"] else nn.Identity(),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 1)  # 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block_time1 = block_klass(mid_dim + self.context_dim, mid_dim,
                                      time_emb_dim=time_dim) if self.conditioning_key in ["concat",
                                                                                          "hybrid"] else block_klass(
            mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attncross = AttentionCross(mid_dim, context_dim=self.context_dim) if self.conditioning_key in [
            "crossattn", "hybrid"] else nn.Identity()
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block_time2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in + self.context_dim, dim_out,
                            time_emb_dim=time_dim) if self.conditioning_key in ["concat", "hybrid"] else block_klass(
                    dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                AttentionCross(dim_out, context_dim=self.context_dim) if self.conditioning_key in ["crossattn",
                                                                                                   "hybrid"] else nn.Identity(),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 1)  # 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)

        if self.bbox_separate:
            self.bbox_hidden2output = Unet1D._decoder_mlp(dim, self.bbox_dim)
            print('separate unet1d decoder of /translation/size/angle')

        else:
            self.final_conv = nn.Conv1d(dim, self.out_dim, 1)
            print('unet1d decoder of all object properties')

    @staticmethod
    def _encoder_mlp(hidden_size, input_size, dropout_rate=0):
        mlp_layers = [
            nn.Conv1d(input_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(hidden_size, hidden_size * 2, 1),
            nn.BatchNorm1d(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(hidden_size * 2, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
        ]
        return nn.Sequential(*mlp_layers)

    @staticmethod
    def _decoder_mlp(hidden_size, output_size, dropout_rate=0):
        mlp_layers = [
            nn.Conv1d(hidden_size, hidden_size * 2, 1),
            nn.BatchNorm1d(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden_size * 2, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden_size, output_size, 1),
        ]
        return nn.Sequential(*mlp_layers)

    def forward(self, x, beta, context=None, context_cross=None):
        batch_size, data_dim = x.size()
        context = context.reshape(batch_size,-1,1)
        context_cross = context_cross.reshape(batch_size, -1, 1)
        if self.conditioning_key == 'concat':
            context = context_cross # cancel the context, and only remain context_cross

        x = x.unsqueeze(2)
        if self.bbox_separate:
            x_bbox = self.bbox_embedf(x)
            x = x_bbox

        x = self.init_layer(x)

        r = x.clone()
        t = self.time_mlp(beta)

        h = []

        ## unet-1D: context here should be the feature of each node, context_cross is the relation feature
        # downsampling blocks
        for block_time1, attncross, block_time2, attn, downsample in self.downs:
            if self.conditioning_key in ["concat", "hybrid"]:
                x = torch.cat((x, context), dim=1)
            x = block_time1(x, t)
            h.append(x)

            x = attncross(x, context_cross) if self.conditioning_key in ["crossattn", "hybrid"] else attncross(x)
            x = block_time2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        # middle block
        if self.conditioning_key in ["concat", "hybrid"]:
            x = torch.cat((x, context), dim=1)
        x = self.mid_block_time1(x, t)
        x = self.mid_attncross(x, context_cross) if self.conditioning_key in ["crossattn", "hybrid"] else self.mid_attncross(x)
        x = self.mid_attn(x)
        x = self.mid_block_time2(x, t)

        # upsampling blocks
        for block_time1, attncross, block_time2, attn, upsample in self.ups:
            if self.conditioning_key in ["concat", "hybrid"]:
                x = torch.cat((x, context), dim=1)
            x = block_time1(torch.cat((x, h.pop()), dim=1), t)
            x = attncross(x, context_cross) if self.conditioning_key in ["crossattn", "hybrid"] else attncross(x)
            x = block_time2(torch.cat((x, h.pop()), dim=1), t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)

        if self.bbox_separate:
            out_bbox = self.bbox_hidden2output(x)
            out = out_bbox
        else:
            out = self.final_conv(x)

        out = out.squeeze(2)
        return out