# https://github.com/linyiheng123/MEMatte

# throw all file into 1 and using less dependancies
# we dont train so copy eval branch only
# also, ignore ViT teacher

# license: copy from timm, detectron2, fvcore, MEMatte so whatever their licenses are, applied here
# contact respective original authors if needed

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor, einsum
from torch.nn.modules.batchnorm import _BatchNorm

import numpy as np
import cv2
from einops import rearrange, repeat, pack, unpack

import collections
from collections import namedtuple
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Generator, Tuple
from functools import partial
from itertools import repeat
import math
import warnings

@dataclass
class ShapeSpec:
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    """

    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None

class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self):
        """
        Subclasses must override this method, but adhere to the same return type.

        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
        pass

    @property
    def size_divisibility(self) -> int:
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 0

    @property
    def padding_constraints(self) -> Dict[str, int]:
        """
        This property is a generalization of size_divisibility. Some backbones and training
        recipes require specific padding constraints, such as enforcing divisibility by a specific
        integer (e.g., FPN) or padding to a square (e.g., ViTDet with large-scale jitter
        in :paper:vitdet). `padding_constraints` contains these optional items like:
        {
            "size_divisibility": int,
            "square_size": int,
            # Future options are possible
        }
        `size_divisibility` will read from here if presented and `square_size` indicates the
        square padding size if `square_size` > 0.

        TODO: use type of Dict[str, int] to avoid torchscipt issues. The type of padding_constraints
        could be generalized as TypedDict (Python 3.8+) to support more types in the future.
        """
        return {}

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


# utils.py
RouterReturn = namedtuple('RouterReturn', ['indices', 'scores', 'routed_tokens', 'routed_mask'])

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seq_len = tensor.shape[dim]
    m = seq_len / multiple
    if m.is_integer():
        return tensor, seq_len

    remainder = math.ceil(m) * multiple - seq_len
    pad_offset = (0,) * (-1 - dim) * 2
    padded_tensor = F.pad(tensor, (*pad_offset, 0, remainder), value = value)
    return padded_tensor, seq_len

def batched_gather(x, indices):
    batch_range = create_batch_range(indices, indices.ndim - 1)
    return x[batch_range, indices]

def identity(t):
    return t

def l2norm(t):
    return F.normalize(t, dim = -1)

# tensor helpers

def create_batch_range(t, right_pad_dims = 1):
    b, device = t.shape[0], t.device
    batch_range = torch.arange(b, device = device)
    pad_dims = ((1,) * right_pad_dims)
    return batch_range.reshape(-1, *pad_dims)


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size # 80, 120
    k_h, k_w = k_size # 80, 120
    Rh = get_rel_pos(q_h, k_h, rel_pos_h) # torch.Size([80, 80, 64])
    Rw = get_rel_pos(q_w, k_w, rel_pos_w) # torch.Size([120, 120, 64])

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh) # torch.Size([6, 80, 120, 80])
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw) # torch.Size([6, 80, 120, 120])

    attn = ( # 2048*2048这里会爆, rel_h: [6, 128, 128, 128], rel_w:[6, 128, 128, 128], None扩充后都乘128：6*128*128*128*128*4 = 6,442,450,944
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


def get_abs_pos(abs_pos, has_cls_token, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x

class Router(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy=None):
        x = self.in_conv(x) # 16, 1024, 384
        B, N, C = x.size()
        local_x = x[:,:, :C//2] # 16, 1024, 192
        global_x = (x[:,:, C//2:]).sum(dim=1, keepdim=True) / N # 16, 1, 192
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1) # 16, 1024, 384
        return self.out_conv(x) # 16, 1024, 2

# vit.py

# import

# meta's fvcore.nn.weight_init
class weight_init():
    @staticmethod
    def c2_xavier_fill(module: nn.Module) -> None:
        """
        Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
        Also initializes `module.bias` to 0.

        Args:
            module (torch.nn.Module): module to initialize.
        """
        # Caffe2 implementation of XavierFill in fact
        # corresponds to kaiming_uniform_ in PyTorch
        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Union[Module, Tensor]`.
        nn.init.kaiming_uniform_(module.weight, a=1)
        if module.bias is not None:
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Union[Module,
            #  Tensor]`.
            nn.init.constant_(module.bias, 0)

    @staticmethod
    def c2_msra_fill(module: nn.Module) -> None:
        """
        Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
        Also initializes `module.bias` to 0.

        Args:
            module (torch.nn.Module): module to initialize.
        """
        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Union[Module, Tensor]`.
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Union[Module,
            #  Tensor]`.
            nn.init.constant_(module.bias, 0)
            
            
# timm.layers
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    assert( traning == False )
    return x # no training


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
        
def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

class env():
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
    
BatchNorm2d = torch.nn.BatchNorm2d
    
class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)
        self.register_buffer("num_batches_tracked", None)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
            res.num_batches_tracked = module.num_batches_tracked
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

    @classmethod
    def convert_frozenbatchnorm2d_to_batchnorm2d(cls, module: nn.Module) -> nn.Module:
        """
        Convert all FrozenBatchNorm2d to BatchNorm2d

        Args:
            module (torch.nn.Module):

        Returns:
            If module is FrozenBatchNorm2d, returns a new module.
            Otherwise, in-place convert module and return it.

        This is needed for quantization:
            https://fb.workplace.com/groups/1043663463248667/permalink/1296330057982005/
        """

        res = module
        if isinstance(module, FrozenBatchNorm2d):
            res = torch.nn.BatchNorm2d(module.num_features, module.eps)

            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data.clone().detach()
            res.running_var.data = module.running_var.data.clone().detach()
            res.eps = module.eps
            res.num_batches_tracked = module.num_batches_tracked
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozenbatchnorm2d_to_batchnorm2d(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

class NaiveSyncBatchNorm(BatchNorm2d):
    """
    In PyTorch<=1.5, ``nn.SyncBatchNorm`` has incorrect gradient
    when the batch size on each worker is different.
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    This is a slower but correct alternative to `nn.SyncBatchNorm`.

    Note:
        There isn't a single definition of Sync BatchNorm.

        When ``stats_mode==""``, this module computes overall statistics by using
        statistics of each worker with equal weight.  The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (N, H, W). This mode does not support inputs with zero batch size.

        When ``stats_mode=="N"``, this module computes overall statistics by weighting
        the statistics of each worker by their ``N``. The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (H, W). It is slower than ``stats_mode==""``.

        Even though the result of this module may not be the true statistics of all samples,
        it may still be reasonable because it might be preferrable to assign equal weights
        to all workers, regardless of their (H, W) dimension, instead of putting larger weight
        on larger images. From preliminary experiments, little difference is found between such
        a simplified implementation and an accurate computation of overall mean & variance.
    """

    def __init__(self, *args, stats_mode="", **kwargs):
        super().__init__(*args, **kwargs)
        assert stats_mode in ["", "N"]
        self._stats_mode = stats_mode

    def forward(self, input):
        assert( not self.traning )
        return super().forward(input)

class LayerNormDetectron2(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            "naiveSyncBN": NaiveSyncBatchNorm,
            # expose stats_mode N as an option to caller, required for zero-len inputs
            "naiveSyncBN_N": lambda channels: NaiveSyncBatchNorm(channels, stats_mode="N"),
            "LN": lambda channels: LayerNormDetectron2(channels),
        }[norm]
    return norm(out_channels)

def check_if_dynamo_compiling():
    if env.TORCH_VERSION >= (2, 1):
        from torch._dynamo import is_compiling

        return is_compiling()
    else:
        return False

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            # Dynamo doesn't support context managers yet
            is_dynamo_compiling = check_if_dynamo_compiling()
            if not is_dynamo_compiling:
                with warnings.catch_warnings(record=True):
                    if x.numel() == 0 and self.training:
                        # https://github.com/pytorch/pytorch/issues/12013
                        assert not isinstance(
                            self.norm, torch.nn.SyncBatchNorm
                        ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of `forward()` method must be NCHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.

    Attribute:
        in_channels (int):
        out_channels (int):
        stride (int):
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        """
        Make this block not trainable.
        This method sets all parameters to `requires_grad=False`,
        and convert all BatchNorm layers to FrozenBatchNorm

        Returns:
            the block itself
        """
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self
    
def patch_batchnorm(module: nn.Module) -> List:
    """Patch all batchnorm instances (1d, 2d, 3d, sync_bn, etc.) of a module
       so that they don't track running stats when torch.no_grad() is enabled.

       This is important in activation checkpointing to ensure stats are tracked
       correctly as if there were no activation checkpointing. The reason is
       that activation checkpointing runs the forward function twice, first
       with torch.no_grad(), then with torch.grad().

    Args:
        module (nn.Module):
            The module to be patched in-place.

    Returns:
        (list):
            A list of hook handles, late can be freed.
    """

    def pre_forward(module: _BatchNorm, input: Tensor) -> None:
        if torch.is_grad_enabled():
            return
        module._track_running_stats_backup = module.track_running_stats
        module.track_running_stats = False

    def post_forward(module: _BatchNorm, input: Tensor, result: Tensor) -> None:
        if torch.is_grad_enabled():
            return
        module.track_running_stats = module._track_running_stats_backup

    hooks = []
    for name, child in module.named_modules():
        # _BatchNorm is base for bn1d, bn2d, bn3d and sync_bn, apex_sync_bn, etc.
        if isinstance(child, _BatchNorm) and not hasattr(child, "disable_patch_batchnorm"):
            # Register the pre/post hooks.
            pre_handle = child.register_forward_pre_hook(pre_forward)
            post_handle = child.register_forward_hook(post_forward)
            hooks += [pre_handle, post_handle]
    return hooks
    
def unpack_non_tensors(
    tensors: Tuple[torch.Tensor, ...], packed_non_tensors: Optional[Dict[str, List[Any]]]
) -> Tuple[Any, ...]:
    """See split_non_tensors."""
    if packed_non_tensors is None:
        return tensors
    assert isinstance(packed_non_tensors, dict), type(packed_non_tensors)
    mixed: List[Any] = []
    is_tensor_list = packed_non_tensors["is_tensor"]
    objects = packed_non_tensors["objects"]
    assert len(tensors) + len(objects) == len(is_tensor_list), (
        f"len(tensors) {len(tensors)} len(objects) {len(objects)} " f"len(is_tensor_list) {len(is_tensor_list)}"
    )
    obj_i = tnsr_i = 0
    for is_tensor in is_tensor_list:
        if is_tensor:
            mixed.append(tensors[tnsr_i])
            tnsr_i += 1
        else:
            mixed.append(objects[obj_i])
            obj_i += 1
    return tuple(mixed)  
  
def checkpoint_wrapper(
    module: nn.Module,
    offload_to_cpu: bool = False,
) -> nn.Module:
    """
    A friendlier wrapper for performing activation checkpointing.

    Compared to the PyTorch version, this version:

        - wraps an nn.Module, so that all subsequent calls will use checkpointing
        - handles keyword arguments in the forward
        - handles non-Tensor outputs from the forward
        - supports offloading activations to CPU

    Usage::

        checkpointed_module = checkpoint_wrapper(my_module, offload_to_cpu=True)
        a, b = checkpointed_module(x, y=3, z=torch.Tensor([1]))

    To understand the benefits of checkpointing and the `offload_to_cpu` flag,
    let's divide activations into 2 types: inner activations and outer
    activations w.r.t. the checkpointed modules. The inner ones are saved
    by activation checkpointing, the outer ones are saved by offload_to_cpu.

    In terms of GPU memory savings:

        - When inner ones are large in size and outer ones are small,
          checkpointing helps a lot, offload_to_cpu may help a little.
        - When inner ones are small and outer ones are large,
          checkpointing helps little, offload_to_cpu helps a lot.
        - When both inner and outer are large, both help and the
          benefit is additive.

    ..Note::

        The first and last layers are not likely to benefit from the `offload_to_cpu` flag
        because (1) there are typically other references to the first layer's input, so
        the GPU memory won't be freed; (2) the input to the last layer is immediately
        used by the backward pass and won't result in memory savings.

    Args:
        module (nn.Module):
            The module to be wrapped
        offload_to_cpu (bool):
            Whether to offload activations to CPU.

    Returns:
        (nn.Module):
            Wrapped module
    """
    # Patch the batchnorm layers in case there are any in this module.
    patch_batchnorm(module)

    # The use of weakref here is to prevent creating a ref cycle: m -> m.forward -> m.
    # When such cycle exists, gc won't collect the module when the module is freed.
    # That causes GPU memory to be leaked. See the unit test for how we catch that.
    #
    # We prefer this over a class wrapper since the class wrapper would have to
    # proxy a lot of fields and methods.
    module.forward = functools.partial(  # type: ignore
        _checkpointed_forward, type(module).forward, weakref.ref(module), offload_to_cpu
    )
    return module

def _checkpointed_forward(
    original_forward: Any, weak_self: Any, offload_to_cpu: bool, *args: Any, **kwargs: Any
) -> Any:
    module = weak_self()

    # If gradients are disabled, just use original `.forward()` method directly.
    if not torch.is_grad_enabled() or thread_local.is_checkpointing_disabled:
        return original_forward(module, *args, **kwargs)

    # Autograd Functions in PyTorch work best with positional args, since
    # the backward must return gradients (or None) for every input argument.
    # We can flatten keyword arguments to make this easier.
    args = (module,) + args
    kwarg_keys, flat_args = pack_kwargs(*args, **kwargs)
    parent_ctx_dict: Dict[str, Any] = {
        "offload": offload_to_cpu,
    }
    # Dummy tensor with grad is used to ensure the backward pass is called. This is needed
    # when original_forward's input are non-tensor (i.e. a tuple). Using this dummy tensor
    # avoids requiring users to set their input tensors's requires_grad flag. In the case
    # of tuple type inputs, setting the flag won't even trigger the backward pass.
    #
    # One implication of this is that since we always feed in a dummy tensor
    # needing grad, then the output will always require grad, even if it originally
    # wouldn't, such as if the module and original input both do not require grad.
    # We get around this by saving the desired requires_grad value in output and
    # detaching the output if needed.
    output = CheckpointFunction.apply(
        torch.tensor([], requires_grad=True), original_forward, parent_ctx_dict, kwarg_keys, *flat_args
    )
    output_requires_grad = parent_ctx_dict["output_requires_grad"]
    if not isinstance(output, torch.Tensor):
        # If output should not require grad, then detach it, since otherwise it will
        # always have requires_grad = True due to our dummy tensor input above that
        # requires_grad
        output = [x.detach() if not output_requires_grad else x for x in output]

        packed_non_tensor_outputs = parent_ctx_dict["packed_non_tensor_outputs"]
        if packed_non_tensor_outputs:
            output = unpack_non_tensors(output, packed_non_tensor_outputs)

    else:
        # If output should not require grad, then detach it, since otherwise it will
        # always have requires_grad = True due to our dummy tensor input above that
        # requires_grad
        if not output_requires_grad:
            output = output.detach()

    return output

# vit.py: end of import
    
class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N, _ = policy.size() # 128, 197, 1
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye # 目的是将对角线上的token计算attention
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)


    def forward(self, x, policy, H, W): # total FLOPs: 4 * B * hw * dim * dim +  2 * B * hw * hw * dim
        if x.ndim == 4:
            B, H, W, _ = x.shape
            N = H*W
        else:      
            B, N, _ = x.shape
        
        
        # qkv with shape (3, B, nHead, H * W, C) self.qkv.flops: b * hw * dim * dim * 3
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) # 6.341787648 reshape和permute没有FLOPs
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, N, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1) # 5.637144576 (B * hw * hw * dim) 14 * 1024*1024*384

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn.reshape(B, self.num_heads, N, N), policy).reshape(B*self.num_heads, N, N)

        #  # 5.637144576 (B * hw * hw * dim)
        if x.ndim == 4:
            x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1) 
        else:
            x = (attn @ v).view(B, self.num_heads, N, -1).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x) # 2.113929216 (B * hw * dim * dim) 14 * 1024 * 384 * 384

        return x
        
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
            
class ResBottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm="LN",
        act_layer=nn.GELU,
        conv_kernels=3,
        conv_paddings=1,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            conv_kernels,
            padding=conv_paddings,
            bias=False,
        )
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        out = x + out
        return out

class ECA(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=self.kernel_size(),
            padding=(self.kernel_size() - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels) / self.gamma) + self.b / self.gamma))
        out = k if k % 2 else k + 1
        return out

    def forward(self, x):

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)



class LTRM(nn.Module):
    """Efficient Block to replace the Original Transformer Block"""
    def __init__(self, dim, expand_ratio = 2, kernel_size = 5):
        super(LTRM, self).__init__()
        self.fc1 = nn.Linear(dim, dim*expand_ratio)
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv2d(dim*expand_ratio, dim*expand_ratio, kernel_size=(kernel_size, kernel_size), groups=dim*expand_ratio, padding=(kernel_size//2, kernel_size//2))
        self.act2 = nn.GELU()
        self.fc2 = nn.Linear(dim*expand_ratio, dim)
        self.eca = ECA(dim)
    
    def forward(self, x, prev_msa=None):
        x = self.act1(self.fc1(x))
        x = self.act2(self.dwconv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        x =  self.fc2(x)
        y = x + self.eca(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return y
        
class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_cc_attn = False,
        use_residual_block=False,
        use_convnext_block=False,
        input_size=None,
        res_conv_kernel_size=3,
        res_conv_padding=1,
        use_efficient_block = False,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size
        self.use_efficient_block = use_efficient_block
        if use_efficient_block:
            self.efficient_block = LTRM(dim = dim)

        self.use_residual_block = use_residual_block
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
                act_layer=act_layer,
                conv_kernels=res_conv_kernel_size,
                conv_paddings=res_conv_padding,
            )
        self.use_convnext_block = use_convnext_block
        if use_convnext_block:
            self.convnext = ConvNextBlock(dim = dim)

        if use_cc_attn:
            self.attn = CrissCrossAttention(dim)

    def msa_forward(self, x, policy, H, W):
        # shortcut = x
        x = self.norm1(x) # 0.27525
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

            x = self.attn(x, policy, H = H, W = W)
            
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            x = self.attn(x, policy, H = H, W = W)


        x =  self.drop_path(x)

        return x

    def mlp_forward(self, x): 
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, policy=None):
        B, H, W, C = x.shape
        N = H * W
        # B, N, C = x.shape
        shortcut = x
        
        if self.use_efficient_block:
                msa = self.efficient_block(x)
                selected_indices = policy.squeeze(-1).bool()
                # if True in selected_indices:
                if torch.any(selected_indices == 1):
                    selected_x = x.reshape(B, -1, C)[selected_indices].unsqueeze(0)
                    slow_msa = self.msa_forward(selected_x, policy=None, H = H, W = W)
                    msa.masked_scatter_(selected_indices.reshape(B, H, W, 1), slow_msa)
        else:
            msa = self.msa_forward(x, policy, H, W).reshape(B, H, W, C)
        x = shortcut + msa
        x = x + self.mlp_forward(x)

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.use_convnext_block:
            x = self.convnext(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x
        
class ViT(Backbone):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        window_block_indexes=(),
        residual_block_indexes=(),
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature="last_feat",
        res_conv_kernel_size=3, 
        res_conv_padding=1,
        topk = 1.,
        multi_score = False,
        router_module = "Ours",
        skip_all_block = False,
        max_number_token = 18500,
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_feature (str): name of the feature from the last block.
        """
        super().__init__()

        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.topk = topk
        self.multi_score = multi_score
        self.skip_all_block = skip_all_block
        self.window_block_indexes = window_block_indexes
        self.max_number_token = max_number_token
        self.router_module = router_module
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=True if (i in window_block_indexes and use_rel_pos == True) else False,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size // patch_size, img_size // patch_size),
                res_conv_kernel_size=res_conv_kernel_size,
                res_conv_padding=res_conv_padding,
                use_efficient_block=False if (i == 0 or i in window_block_indexes)  else True,
            )
            if use_act_checkpoint:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)
        
        self.routers = nn.ModuleList()
        for i in range(depth - len(window_block_indexes)):
            router = Router(embed_dim = embed_dim)
            self.routers.append(router)


        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        assert not self.training
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        B, H, W, C = x.shape
        N = H * W
        prev_decision = torch.ones(B, N, 1, dtype = x.dtype, device = x.device)
        out_pred_prob = []
        out_hard_keep_decision = []
        count = 0
        
        for i, blk in enumerate(self.blocks): # x.shape = torch.Size([16, 32, 32, 384])
            if i in self.window_block_indexes or i == 0:
                x = blk(x, policy = None)
                continue
            pred_score = self.routers[count](x.reshape(B, -1, C), prev_decision).reshape(B, -1, 2) # B, N, 2
            count = count + 1
            """gumbel_softmax"""
            # hard_keep_decision = F.gumbel_softmax(pred_score, hard = True)[:, :, 0:1] # torch.Size([1, N, 1])
                
            """argmax"""
            hard_keep_decision = torch.zeros_like(pred_score[..., :1])
            hard_keep_decision[pred_score[..., 0] > pred_score[..., 1]] = 1.

            if hard_keep_decision.sum().item() > self.max_number_token:
                print("=================================================================")
                print(f"Decrease the number of tokens in Global attention: {self.max_number_token}")
                print("=================================================================")
                _, sort_index = torch.sort((pred_score[..., 0] - pred_score[..., 1]), descending=True)
                sort_index = sort_index[:, :self.max_number_token]
                hard_keep_decision = torch.zeros_like(pred_score[..., :1])
                hard_keep_decision[:, sort_index.squeeze(0), :] = 1.

            # threshold = torch.quantile(pred_score[..., :1], 0.85)
            # hard_keep_decision = (pred_score[..., :1] >= threshold).float()
                
            # out_pred_prob.append(pred_score[..., 0].reshape(B, x.shape[1], x.shape[2]))

            out_pred_prob.append(pred_score.reshape(B, x.shape[1], x.shape[2], 2))
            out_hard_keep_decision.append(hard_keep_decision.reshape(B, x.shape[1], x.shape[2], 1))
            x = blk(x, policy = hard_keep_decision)

        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}

        return outputs['last_feat'], out_pred_prob, out_hard_keep_decision

# detail_capture.py
class Basic_Conv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """
    def __init__(
        self,
        in_chans,
        out_chans,
        stride=2,
        padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class ConvStream(nn.Module):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """
    def __init__(
        self,
        in_chans = 4,
        out_chans = [48, 96, 192],
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        
        self.conv_chans = out_chans.copy()
        self.conv_chans.insert(0, in_chans)
        
        for i in range(len(self.conv_chans)-1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i+1]
            self.convs.append(
                Basic_Conv3x3(in_chan_, out_chan_)
            )
    
    def forward(self, x):
        out_dict = {'D0': x}
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            name_ = 'D'+str(i+1)
            out_dict[name_] = x
        
        return out_dict

class Fusion_Block(nn.Module):
    """
    Simple fusion block to fuse feature from ConvStream and Plain Vision Transformer.
    """
    def __init__(
        self,
        in_chans,
        out_chans,
    ):
        super().__init__()
        self.conv = Basic_Conv3x3(in_chans, out_chans, stride=1, padding=1)

    def forward(self, x, D):
        F_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.cat([D, F_up], dim=1)
        out = self.conv(out)

        return out    

class Matting_Head(nn.Module):
    """
    Simple Matting Head, containing only conv3x3 and conv1x1 layers.
    """
    def __init__(
        self,
        in_chans = 32,
        mid_chans = 16,
    ):
        super().__init__()
        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_chans, mid_chans, 3, 1, 1),
            nn.BatchNorm2d(mid_chans),
            nn.ReLU(True),
            nn.Conv2d(mid_chans, 1, 1, 1, 0)
            )

    def forward(self, x):
        x = self.matting_convs(x)

        return x

class Detail_Capture(nn.Module):
    """
    Simple and Lightweight Detail Capture Module for ViT Matting.
    """
    def __init__(
        self,
        in_chans = 384,
        img_chans=4,
        convstream_out = [48, 96, 192],
        fusion_out = [256, 128, 64, 32],
    ):
        super().__init__()
        assert len(fusion_out) == len(convstream_out) + 1

        self.convstream = ConvStream(in_chans = img_chans)
        self.conv_chans = self.convstream.conv_chans

        self.fusion_blks = nn.ModuleList()
        self.fus_channs = fusion_out.copy()
        self.fus_channs.insert(0, in_chans)
        for i in range(len(self.fus_channs)-1):
            self.fusion_blks.append(
                Fusion_Block(
                    in_chans = self.fus_channs[i] + self.conv_chans[-(i+1)],
                    out_chans = self.fus_channs[i+1],
                )
            )

        self.matting_head = Matting_Head(
            in_chans = fusion_out[-1],
        )

    def forward(self, features, images):
        detail_features = self.convstream(images)
        for i in range(len(self.fusion_blks)):
            d_name_ = 'D'+str(len(self.fusion_blks)-i-1)
            features = self.fusion_blks[i](features, detail_features[d_name_])
        
        phas = torch.sigmoid(self.matting_head(features))

        return {'phas': phas}

# mematte.py
class MEMatte(nn.Module):
    def __init__(self,
                 *,
                 teacher_backbone, # ignore
                 backbone,
                 criterion, # ignore
                 pixel_mean,
                 pixel_std,
                 input_format,
                 size_divisibility,
                 decoder,
                 distill = True,
                 distill_loss_ratio = 1.,
                 token_loss_ratio = 1.,
                 balance_loss = "MSE",
                 ):
        super(MEMatte, self).__init__()
        self.teacher_backbone = None
        self.backbone = backbone
        self.criterion = None
        self.input_format = input_format
        self.balance_loss = balance_loss
        self.size_divisibility = size_divisibility
        self.decoder = decoder
        self.distill = distill
        self.distill_loss_ratio = distill_loss_ratio
        self.token_loss_ratio = token_loss_ratio
        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
    
    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, patch_decoder=True):
        assert not self.training
        images, targets, H, W = self.preprocess_inputs(batched_inputs)

        features, out_pred_prob, out_hard_keep_decision = self.backbone(images)
        if patch_decoder:
            outputs = self.patch_inference(features=features, images=images)
        else:
            outputs = self.decoder(features, images)
    
        outputs['phas'] = outputs['phas'][:,:,:H,:W]

        return outputs, out_pred_prob, out_hard_keep_decision
            
        """测试flops"""
        # features = self.backbone(images)
        # # outputs = self.decoder(features, images)
        # # outputs['phas'] = outputs['phas'][:,:,:H,:W]
        # return features

        """测试decoder flops"""

        


    def patch_inference(self, features, images):
        patch_size = 512
        overlap = 64
        image_size = patch_size + 2 * overlap
        feature_patch_size = patch_size // 16
        feature_overlap = overlap // 16
        features_size = feature_patch_size + 2 * feature_overlap
        B, C, H, W = images.shape
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        pad_images = F.pad(images.permute(0,2,3,1), (0,0,0,pad_w,0,pad_h)).permute(0,3,1,2)
        _, _, pad_H, pad_W = pad_images.shape

        _, _, H_fea, W_fea = features.shape
        pad_fea_h = (feature_patch_size - H_fea % feature_patch_size) % feature_patch_size
        pad_fea_w = (feature_patch_size - W_fea % feature_patch_size) % feature_patch_size
        pad_features = F.pad(features.permute(0,2,3,1), (0,0,0,pad_fea_w,0,pad_fea_h)).permute(0,3,1,2)
        _, _, pad_fea_H, pad_fea_W = pad_features.shape
        
        h_patch_num = pad_images.shape[2] // patch_size
        w_patch_num = pad_images.shape[3] // patch_size

        outputs = torch.zeros_like(pad_images[:,0:1,:,:])

        for i in range(h_patch_num):
            for j in range(w_patch_num):
                start_top = i * patch_size
                end_bottom = start_top + patch_size
                start_left = j*patch_size
                end_right = start_left + patch_size
                coor_top = start_top if (start_top - overlap) < 0 else (start_top - overlap)
                coor_bottom = end_bottom if (end_bottom + overlap) > pad_H else (end_bottom + overlap)
                coor_left = start_left if (start_left - overlap) < 0 else (start_left - overlap)
                coor_right = end_right if (end_right + overlap) > pad_W else (end_right + overlap)
                selected_images = pad_images[:,:,coor_top:coor_bottom, coor_left:coor_right]

                fea_start_top = i * feature_patch_size
                fea_end_bottom = fea_start_top + feature_patch_size
                fea_start_left = j*feature_patch_size
                fea_end_right = fea_start_left + feature_patch_size
                coor_top_fea = fea_start_top if (fea_start_top - feature_overlap) < 0 else (fea_start_top - feature_overlap)
                coor_bottom_fea = fea_end_bottom if (fea_end_bottom + feature_overlap) > pad_fea_H else (fea_end_bottom + feature_overlap)
                coor_left_fea = fea_start_left if (fea_start_left - feature_overlap) < 0 else (fea_start_left - feature_overlap)
                coor_right_fea = fea_end_right if (fea_end_right + feature_overlap) > pad_fea_W else (fea_end_right + feature_overlap)
                selected_fea = pad_features[:,:,coor_top_fea:coor_bottom_fea, coor_left_fea:coor_right_fea]



                outputs_patch = self.decoder(selected_fea, selected_images)

                coor_top = start_top if (start_top - overlap) < 0 else (coor_top + overlap)
                coor_bottom = coor_top + patch_size
                coor_left = start_left if (start_left - overlap) < 0 else (coor_left + overlap)
                coor_right = coor_left + patch_size

                coor_out_top = 0 if (start_top - overlap) < 0 else overlap
                coor_out_bottom = coor_out_top + patch_size
                coor_out_left = 0 if (start_left - overlap) < 0 else overlap
                coor_out_right = coor_out_left + patch_size

                outputs[:, :, coor_top:coor_bottom, coor_left:coor_right] = outputs_patch['phas'][:,:,coor_out_top:coor_out_bottom,coor_out_left:coor_out_right]

        outputs = outputs[:,:,:H, :W]
        return {'phas':outputs}

    def preprocess_inputs(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = batched_inputs["image"].to(self.device)
        trimap = batched_inputs['trimap'].to(self.device)
        images = (images - self.pixel_mean) / self.pixel_std

        if 'fg' in batched_inputs.keys():
            trimap[trimap < 85] = 0
            trimap[trimap >= 170] = 1
            trimap[trimap >= 85] = 0.5

        images = torch.cat((images, trimap), dim=1)
        
        B, C, H, W = images.shape
        if images.shape[-1]%32!=0 or images.shape[-2]%32!=0:
            new_H = (32-images.shape[-2]%32) + H
            new_W = (32-images.shape[-1]%32) + W
            new_images = torch.zeros((images.shape[0], images.shape[1], new_H, new_W)).to(self.device)
            new_images[:,:,:H,:W] = images[:,:,:,:]
            del images
            images = new_images

        if "alpha" in batched_inputs:
            phas = batched_inputs["alpha"].to(self.device)
        else:
            phas = None

        return images, dict(phas=phas), H, W

def CreateMEMatteModel(max_number_token=18500, embed_dim = 384, num_heads = 6, decoder_in_chans = 384):
    model = MEMatte(
        teacher_backbone = None,
        backbone = ViT(
            in_chans=4,
            img_size=512,
            patch_size=16,
            embed_dim=embed_dim,
            depth=12,
            num_heads=num_heads,
            drop_path_rate=0,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[
                # 2, 5, 8 11 for global attention
                0,
                1,
                3,
                4,
                6,
                7,
                9,
                10,
            ],
            residual_block_indexes=[2, 5, 8, 11],
            use_rel_pos=True,
            out_feature="last_feat",
            topk = 0.25,
            max_number_token = max_number_token
        ),
        criterion = None,
        pixel_mean = [123.675 / 255., 116.280 / 255., 103.530 / 255.],
        pixel_std = [58.395 / 255., 57.120 / 255., 57.375 / 255.],
        input_format = "RGB",
        size_divisibility=32,
        decoder = Detail_Capture(in_chans = decoder_in_chans)
    )
    model.eval()
    return model

def CreateMEMatteS(max_number_token=18500):
    model = CreateMEMatteModel(max_number_token=max_number_token)
    return model
    
def CreateMEMatteB(max_number_token=18500):
    model = CreateMEMatteModel(max_number_token=max_number_token, embed_dim = 768, num_heads = 12, decoder_in_chans = 768)
    return model

if __name__ == '__main__':
    
    checkpoint_dir='./MEMatte_ViTS_DIM.pth'
    device = torch.device('cpu')
    weights = torch.load(checkpoint_dir,map_location=device, weights_only=True)
    model = CreateMEMatteS()
    model.load_state_dict(weights)
    model.to(device)
    
    # test on local image
    from PIL import Image
    from torchvision.transforms import functional as FF
    import numpy as np
    import cv2
    img = Image.open('input.jpg').convert('RGB')
    tris = Image.open('trimask.png').convert('L')

    img = FF.to_tensor(img)
    tris = FF.to_tensor(tris)
    img = img[None,...]
    tris = tris[None,...]
    data = {'image': img, 'trimap': tris}
    for k in data:
        data[k].to(model.device)
    with torch.no_grad():
        output, _, _ = model(data, patch_decoder=True)
    print(output['phas'].shape)
    output = output['phas'][0][0].detach().numpy()
    output = (output*255).clip(0,255).astype(np.uint8)
    cv2.imwrite('refinemask3.png', output)