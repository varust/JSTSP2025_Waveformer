import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Optional
import math
import warnings
import numbers
from torch import einsum
import numpy as np
from torch import Tensor, int32
from torch.nn import Module
from einops import rearrange, pack, unpack
import pywt
from pytorch_wavelets import DWTForward, DWTInverse
import matplotlib.pyplot as plt

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class LocalMSA0(nn.Module):
    """
    The Local MSA partitions the input into non-overlapping windows of size M × M, treating each pixel within the window as a token, and computes self-attention within the window.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.pos_emb = nn.Parameter(
            torch.Tensor(1, num_heads, window_size[0] * window_size[1], window_size[0] * window_size[1]))

        trunc_normal_(self.pos_emb)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x.shape

        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b c (h b0) (w b1) -> (b h w) (b0 b1) c',
                                          b0=self.window_size[0], b1=self.window_size[1]), (q, k, v))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))


        q *= self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = rearrange(out, '(b h w) (b0 b1) c -> b c (h b0) (w b1)', h=h // self.window_size[0],
                        w=w // self.window_size[1],
                        b0=self.window_size[0])
        out = self.project_out(out)

        return out

def visualize_single_feature_map(tensor, title='Feature Map', cmap='jet', save_path=None, vmax=None):
    """
    可视化形状为 (1, 1, H, W) 的特征图

    参数:
        tensor: torch.Tensor, 形状为 (1, 1, H, W)
        title: 图像标题
        cmap: 使用的颜色图（如 'viridis', 'gray', 'hot' 等）
        save_path: 如果指定，则保存图像到该路径；否则直接显示
    """
    assert tensor.ndim == 4 and tensor.shape[0] == 1 and tensor.shape[1] == 1, \
        f'输入形状应为 (1, 1, H, W)，但收到的是 {tensor.shape}'

    feature_map = tensor.squeeze().detach().cpu().numpy()  # (H, W)

    plt.figure(figsize=(5, 5))
    if vmax:
        im = plt.imshow(feature_map, cmap=cmap, vmax=vmax, vmin=0.0)
    else:
        im = plt.imshow(feature_map, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f'保存到: {save_path}')
    else:
        plt.show()

    plt.close()

class LocalMSA(nn.Module):
    """
    The Local MSA partitions the input into non-overlapping windows of size M × M, treating each pixel within the window as a token, and computes self-attention within the window.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.x_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.sobel = SobelEdge(dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1))

        self.pos_emb = nn.Parameter(
            torch.Tensor(1, num_heads, window_size[0] * window_size[1], window_size[0] * window_size[1]))

        trunc_normal_(self.pos_emb)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x.shape
        var = self.sobel(self.x_dwconv(x)).mean(1, keepdims=True)
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b c (h b0) (w b1) -> (b h w) (b0 b1) c',
                                          b0=self.window_size[0], b1=self.window_size[1]), (q, k, v))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        var_windows = rearrange(var, 'b 1 (h m1) (w m2) -> (b h w) (m1 m2)', m1=self.window_size[0], m2=self.window_size[1])
        var_expanded = var_windows.unsqueeze(1).expand(-1, self.num_heads, -1) * self.temperature
        diag_bias = torch.diag_embed(var_expanded)

        q *= self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb + diag_bias
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = rearrange(out, '(b h w) (b0 b1) c -> b c (h b0) (w b1)', h=h // self.window_size[0],
                        w=w // self.window_size[1],
                        b0=self.window_size[0])
        out = self.project_out(out)

        return out


# class NonLocalMSA(nn.Module):
#     """
#     The Non-Local MSA divides the input into N × N non-overlapping windows, treating each window as a token, and computes self-attention across the windows.
#     """
#
#     def __init__(self,
#                  dim,
#                  num_heads,
#                  window_num
#                  ):
#         super().__init__()
#         self.dim = dim
#         self.window_num = window_num
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#         #self.token_size = [2, 2]
#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
#         self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
#         self.window_size = [8,8]
#         self.pos_emb = nn.Parameter(
#            torch.Tensor(1, num_heads, self.dim//self.num_heads, self.dim//self.num_heads))
#         trunc_normal_(self.pos_emb)
#
#     def forward(self, x):
#         """
#         x: [b,c,h,w]
#         return out: [b,c,h,w]
#         """
#         b, c, h, w = x.shape
#
#         q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
#
#         q, k, v = map(lambda t: rearrange(t, 'b c (h b0) (w b1) -> (b h w) (b0 b1) c',
#                                           b0=self.window_size[0], b1=self.window_size[1]), (q, k, v))
#
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
#
#         q *= self.scale
#         sim = einsum('b h n d, b h m p -> b h d p', q, k)
#         #print(sim.shape)
#         sim = sim + self.pos_emb
#         attn = sim.softmax(dim=-1)
#
#         out = einsum('b h d p, b h n p -> b h n d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#
#         out = rearrange(out, '(b h w) (b0 b1) c -> b c (h b0) (w b1)', h=h // self.window_size[0],
#                         w=w // self.window_size[1],
#                         b0=self.window_size[0])
#         out = self.project_out(out)
#
#         return out

class SobelEdge(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel_x = torch.tensor([[[-1., 0., 1.],
                                  [-2., 0., 2.],
                                  [-1., 0., 1.]]], dtype=torch.float32).expand(channels, 1, 3, 3)
        kernel_y = torch.tensor([[[-1., -2., -1.],
                                  [ 0.,  0.,  0.],
                                  [ 1.,  2.,  1.]]], dtype=torch.float32).expand(channels, 1, 3, 3)
        self.register_buffer('sobel_x', kernel_x)
        self.register_buffer('sobel_y', kernel_y)

    def forward(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return grad_mag

class NonLocalMSA(nn.Module):
    """
    The Local MSA partitions the input into non-overlapping windows of size M × M, treating each pixel within the window as a token, and computes self-attention within the window.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.pos_emb = nn.Parameter(
            torch.Tensor(1, num_heads, dim // num_heads, dim // num_heads))

        trunc_normal_(self.pos_emb)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x.shape
        x = x.roll(shifts=4, dims=2).roll(shifts=4, dims=3)
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b c (h b0) (w b1) -> (b h w) (b0 b1) c',
                                          b0=self.window_size[0], b1=self.window_size[1]), (q, k, v))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h d n', h=self.num_heads), (q, k, v))


        q *= self.scale
        sim = einsum('b h i n, b h j n -> b h i j', q, k)
        #print(sim.shape,self.pos_emb.shape)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j n -> b h i n', attn, v)
        out = rearrange(out, 'b h d n -> b n (h d)')

        out = rearrange(out, '(b h w) (b0 b1) c -> b c (h b0) (w b1)', h=h // self.window_size[0],
                        w=w // self.window_size[1],
                        b0=self.window_size[0])
        out = self.project_out(out)
        out = out.roll(shifts=-4, dims=2).roll(shifts=-4, dims=3)
        return out

def FFN_FN(ffn_name, dim):
    if ffn_name == "Gated_Dconv_FeedForward":
        return Gated_Dconv_FeedForward(dim, ffn_expansion_factor=2.66)
    elif ffn_name == "FeedForward":
        return FeedForward(dim=dim)


class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66):
        super(Gated_Dconv_FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=False)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=True)

        self.act_fn = nn.GELU()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: [b, c, h, w]
        return out: [b, c, h, w]
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        return x


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x)
        return out


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # x: (b, c, h, w)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, layernorm_type='WithBias'):
        super().__init__()
        self.fn = fn
        self.layernorm_type = layernorm_type
        if layernorm_type == 'BiasFree' or layernorm_type == 'WithBias':
            self.norm = LayerNorm(dim, layernorm_type)
        else:
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        if self.layernorm_type == 'BiasFree' or self.layernorm_type == 'WithBias':
            x = self.norm(x)
        else:
            h, w = x.shape[-2:]
            x = to_4d(self.norm(to_3d(x)), h, w)
        return self.fn(x, *args, **kwargs)
def find_duplicate_vectors(tensor):
    # 将张量转换为列表，每个元素是一个向量
    vectors = tensor.view(-1, tensor.size(-1)).tolist()

    # 创建一个字典来存储向量和它们的索引
    vector_dict = {}
    indices = [None] * len(vectors)

    for index, vector in enumerate(vectors):
        # 将向量转换为元组，因为列表是可变的，不能作为字典的键
        vector_tuple = tuple(vector)
        # 检查向量是否已经在字典中
        if vector_tuple in vector_dict:
            # 如果在，添加索引到对应的列表中
            vector_dict[vector_tuple].append(index)
        else:
            # 如果不在，创建新的键值对
            vector_dict[vector_tuple] = [index]
            # 并且在索引列表中记录这个索引
            indices[index] = len(vector_dict) - 1

    # 创建一个列表来存储每个向量的索引
    index_list = []
    for vector_tuple in vector_dict:
        index_list.append(vector_dict[vector_tuple])

    return index_list, vector_dict


# helper functions

def exists(v):
    return v is not None


def default(*args):
    for arg in args:

        if exists(arg):
            return arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


# main class

def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r

def feature_shuffle(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim-1, index=index)
    return shuffled_x

class WindowAttention(nn.Module):
    r"""
    Shifted Window-based Multi-head Self-Attention

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, mask=None):
        r"""
        Args:
            qkv: Input query, key, and value tokens with shape of (num_windows*b, n, c*3)
            rpi: Relative position index
            mask (0/-inf):  Mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q, k, v = map(lambda t: rearrange(t, 'b h (w n) c -> (b n) h w c',
                                          w=self.window_size[0]*self.window_size[1]), (q, k, v))

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        out = attn @ v
        out = rearrange(out,'(b n) h w c -> b (n w) (h c)',b = b_)
        #x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(out)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'

    def flops(self, n):
        flops = 0
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        #  x = (attn @ v)
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += n * self.dim * self.dim
        return flops

class ATD_CA(nn.Module):
    r"""
    Adaptive Token Dictionary Cross-Attention.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_tokens (int): Number of tokens in external token dictionary. Default: 64
        reducted_dim (int, optional): Reducted dimension number for query and key matrix. Default: 4
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_tokens=64, reducted_dim=10, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.rc = reducted_dim
        self.qkv_bias = qkv_bias

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.wa = nn.Linear(dim, dim, bias=qkv_bias)

        #self.scale = nn.Parameter(torch.ones([self.num_tokens]) * 0.5, requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_decode):
        r"""
        Args:
            x: input features with shape of (b, n, c)
            td: token dicitionary with shape of (b, m, c)
            x_size: size of the input x (h, w)
        """
        #h, w = x_size
        # b, n, c = x.shape
        # b, m, c = x_decode.shape
        # rc = self.rc

        # Q: b, n, c
        q = self.wq(x)
        # K: b, m, c
        k = self.wk(x)
        # V: b, m, c
        v = self.wv(x)

        #a = self.wa(x_decode) # bs tokens_num tokens_dim

        # Q @ K^T
        attn1 = (F.normalize(q, dim=-1).transpose(-2, -1) @ F.normalize(k, dim=-1))  # b, n, n_tk
        #scale = torch.clamp(self.scale, 0, 1)
        #attn1 = attn1 * (1 + scale * np.log(self.num_tokens))
        attn1 = self.softmax(attn1)

        # attn2 = (F.normalize(a, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))  # b, n, n_tk
        # scale = torch.clamp(self.scale, 0, 1)
        #
        # #print(attn1.shape,attn2.shape) torch.Size([4, 81920, 128]) torch.Size([4, 128, 81920])
        # attn2 = attn2.permute(0,2,1) * (1 + scale * np.log(self.num_tokens))
        # attn2 = self.softmax(attn2.permute(0,2,1))

        # Attn * V
        # x = (attn2 @ v)#.reshape(b, n, c)
        x = v @ attn1
        return x#, attn1

    def flops(self, n):
        n_tk = self.num_tokens
        flops = 0
        # qkv = self.wq(x)
        flops += n * self.dim * self.rc
        # k = self.wk(gc)
        flops += n_tk * self.dim * self.rc
        # v = self.wv(gc)
        flops += n_tk * self.dim * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += n * self.dim * self.rc
        #  x = (attn @ v)
        flops += n * n_tk * self.dim

        return flops

class AC_MSA(nn.Module):
    r"""
    Adaptive Category-based Multihead Self-Attention.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_tokens (int): Number of tokens in external dictionary. Default: 64
        num_heads (int): Number of attention heads. Default: 4
        category_size (int): Number of tokens in each group for global sparse attention. Default: 128
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_tokens=64, num_heads=4, category_size=128, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.category_size = category_size

        # self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((1, 1))), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, indices):
        """
        Args:
            x: input features with shape of (b, HW, c)
            mask: similarity map with shape of (b, HW, m)
            x_size: size of the input x
        """

        #H, W = x_size
        b, n, c3 = qkv.shape
        c = c3 // 3
        b, h, w = indices.shape
        tk_id = indices.flatten(1)
        #b, n, m = sim.shape
        gs = min(n, self.category_size)  # group size
        ng = (n + gs - 1) // gs
        # classify features into groups based on similarity map (sim)
        #tk_id = torch.argmax(sim, dim=-1, keepdim=False)
        # sort features by type
        x_sort_values, x_sort_indices = torch.sort(tk_id, dim=-1)
        x_sort_indices_reverse = index_reverse(x_sort_indices)
        shuffled_qkv = feature_shuffle(qkv, x_sort_indices)  # b, n, c3
        pad_n = ng * gs - n
        paded_qkv = torch.cat((shuffled_qkv, torch.flip(shuffled_qkv[:, n - pad_n:n, :], dims=[1])), dim=1)
        y = paded_qkv.reshape(b, -1, gs, c3)

        qkv = y.reshape(b, ng, gs, 3, self.num_heads, c // self.num_heads).permute(3, 0, 1, 4, 2, 5)  # 3, b, ng, nh, gs, c//nh
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Q @ K^T
        attn = (q @ k.transpose(-2, -1))  # b, ng, nh, gs, gs

        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(qkv.device)).exp()
        attn = attn * logit_scale

        # softmax
        attn = self.softmax(attn)  # b, ng, nh, gs, gs

        # Attn * V
        y = (attn @ v).permute(0, 1, 3, 2, 4).reshape(b, n + pad_n, c)[:, :n, :]

        x = feature_shuffle(y, x_sort_indices_reverse)
        x = self.proj(x)

        return x

    def flops(self, n):
        flops = 0

        # attn = (q @ k.transpose(-2, -1))
        flops += n * self.dim * self.category_size
        #  x = (attn @ v)
        flops += n * self.dim * self.category_size
        # x = self.proj(x)
        flops += n * self.dim * self.dim

        return flops

class FSQTransformerLayer(nn.Module):
    r"""
    FSQ Transformer Layer

    Args:
        dim (int): Number of input channels.
        idx (int): Layer index.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        category_size (int): Category size for AC-MSA.
        num_tokens (int): Token number for each token dictionary.
        reducted_dim (int): Reducted dimension number for query and key matrix.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        is_last (bool): True if this layer is the last of a ATD Block. Default: False
    """

    def __init__(self,
                 dim,
                 num_heads,
                 category_size,
                 num_tokens,
                 reducted_dim,
                 qkv_bias=True
                 ):
        super().__init__()

        self.dim = dim
        self.window_size = 8
        self.shift_size = 0
        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_win = WindowAttention(dim = dim,window_size=[16,16],num_heads=num_heads)
        # self.attn_aca = AC_MSA(
        #     self.dim,
        #     num_tokens=num_tokens,
        #     num_heads=num_heads,
        #     category_size=category_size,
        #     qkv_bias=qkv_bias,
        # )


    def forward(self, x,indices):
        b, c, h, w = x.shape
        c3 = 3 * c
        x = x.flatten(2).permute(0,2,1) # 256*320 == 81920

        shortcut = x
        qkv = self.wqkv(x)

        x_attn = self.attn_win(qkv)
        # AC_MSA
        # x_aca = self.attn_aca(qkv, indices)

        x = shortcut + x_attn #+ x_attn
        x = x.permute(0,2,1).reshape(b,c,h,w)

        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.input_resolution if input_resolution is None else input_resolution

        # qkv = self.wqkv(x)
        flops += self.dim * 3 * self.dim * h * w

        # W-MSA/SW-MSA, ATD-CA, AC-MSA
        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn_win.flops(self.window_size * self.window_size)
        flops += self.attn_atd.flops(h * w)
        flops += self.attn_aca.flops(h * w)

        # mlp
        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
        flops += h * w * self.dim * self.convffn_kernel_size ** 2 * self.mlp_ratio

        return flops

class Resblock(nn.Module):
    def __init__(self,dim):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(dim,dim,3,1,1)
        self.conv2 = nn.Conv2d(dim,dim,3,1,1)
        self.gelu = nn.GELU()
    def forward(self, x):
        inter = self.conv1(x)
        inter = self.gelu(inter)
        inter = self.conv2(inter)+x
        return inter


class _Memory_Block(nn.Module):
    def __init__(self, hdim, kdim, moving_average_rate=0.999):
        super().__init__()
        kernel_size = 3
        self.c = hdim
        self.k = kdim

        self.moving_average_rate = moving_average_rate

        self.units = nn.Embedding(kdim, hdim)
        self.relu = nn.ReLU()
        self.power = nn.Parameter(torch.zeros(size=(1, 1, 1, hdim)))
        self.alpha = 4
        self.g = nn.Linear(hdim, hdim)
        self.dwc = nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=kernel_size,
                             groups=hdim, padding=kernel_size // 2)
    def update(self, x, score, m=None):
        '''
            x: (n, c)
            e: (k, c)
            score: (n, k)
        '''
        if m is None:
            m = self.units.weight.data
        x = x.detach()
        embed_ind = torch.max(score, dim=1)[1]  # (n, )
        embed_onehot = F.one_hot(embed_ind, self.k).type(x.dtype)  # (n, k)
        embed_onehot_sum = embed_onehot.sum(0)
        embed_sum = x.transpose(0, 1) @ embed_onehot  # (c, k)
        embed_mean = embed_sum / (embed_onehot_sum + 1e-6)
        new_data = m * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)
        if self.training:
            self.units.weight.data = new_data
        return new_data

    def forward(self, x, update_flag=True):
        '''
          x: (b, c, h, w)
          embed: (k, c)
        '''

        b, c, h, w = x.size()
        assert c == self.c
        k, c = self.k, self.c
        kernel_function = nn.ReLU()
        power = 1 + self.alpha * nn.functional.sigmoid(self.power)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, c)  # (n, c)
        g = self.g(x)
        m = self.units.weight.data  # (k, c)

        xn = F.normalize(x, dim=1)  # (n, c)
        mn = F.normalize(m, dim=1)  # (k, c)
        score = torch.matmul(xn, mn.t())  # (n, k)

        if update_flag:
            m = self.update(x, score, m)
            mn = F.normalize(m, dim=1)  # (k, c)
            #score = torch.matmul(xn, mn.t())  # (n, k)

        q_pos = kernel_function(xn) ** power
        q_neg = kernel_function(-xn) ** power
        k_pos = kernel_function(mn) ** power
        k_neg = kernel_function(-mn) ** power
        q_sim = torch.cat([q_pos, q_neg],dim=-1)
        q_opp = torch.cat([q_neg, q_pos],dim=-1)
        k = torch.cat([k_pos, k_neg],dim=-1)
        v1, v2 = torch.chunk(m, 2, dim=-1)

        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        #kv = (k.transpose(-2, -1)) @ (v1)
        kv = (k.transpose(-2, -1) * (x.shape[0] ** -0.5)) @ (v1 * (x.shape[0] ** -0.5))
        x_sim = q_sim @ kv * z
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        #kv = (k.transpose(-2, -1)) @ (v2)
        kv = (k.transpose(-2, -1) * (x.shape[0] ** -0.5)) @ (v2 * (x.shape[0] ** -0.5))
        x_opp = q_opp @ kv * z

        x = torch.cat([x_sim, x_opp], dim=-1)
        g = g.view(b, h, w, c).permute(0, 3, 1, 2)
        #*g.unsqueeze(0).unsqueeze(0)
        #x = x.transpose(1, 2).reshape(b, h*w, c)
        #soft_label = F.softmax(score, dim=1)
        #out = torch.matmul(soft_label, m)  # (n, c)
        out = x.view(b, h, w, c).permute(0, 3, 1, 2)
        out = out+ self.dwc(out)
        out = out*g

        return out, score

class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.rescale = nn.Parameter(torch.ones(num_heads, 1))
        self.x_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape
        var = torch.var(self.x_dwconv(x), dim=(-2, -1), keepdim=True)
        c_head = c // self.num_heads
        var = var.view(b, self.num_heads, c_head)

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        diag_bias = torch.diag_embed(self.rescale * var)

        attn = attn + diag_bias
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class LLBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            window_size=(8, 8),
            window_num=(8, 8),
            num_blocks=1,
            layernorm_type="WithBias",
    ):
        super().__init__()
        self.window_size = window_size
        self.window_num = window_num
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, LocalMSA(dim=dim, window_size=window_size, num_heads=num_heads),
                        layernorm_type=layernorm_type),
                # PreNorm(dim, NonLocalMSA(dim=dim, num_heads=num_heads, window_size=window_num),
                #         layernorm_type=layernorm_type),
                # PreNorm(dim, ChannelAttention(dim=dim, num_heads=num_heads, bias=False),
                #         layernorm_type=layernorm_type),
                PreNorm(dim, FFN_FN(ffn_name="Gated_Dconv_FeedForward", dim=dim), layernorm_type=layernorm_type),
                # PreNorm(dim, FFN_FN(ffn_name="Gated_Dconv_FeedForward", dim=dim), layernorm_type=layernorm_type),
                # PreNorm(dim, FFN_FN(ffn_name="Gated_Dconv_FeedForward", dim=dim), layernorm_type=layernorm_type)

            ]))

        self.non_linear = nn.Sigmoid()

    def forward(self, x):
        # x = self.fea_encoding(x)
        for (local_msa, ffn) in self.blocks:
            x = x + local_msa(x)
            x = x + ffn(x)
        return x

class MEMBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            window_size=(8, 8),
            window_num=(8, 8),
            num_blocks=2,
            layernorm_type="WithBias",
    ):
        super().__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(mode='zero', wave='haar')
        self.window_size = window_size
        self.window_num = window_num
        self.hf_conv = PreNorm(dim, FFN_FN(ffn_name="Gated_Dconv_FeedForward", dim=dim), layernorm_type=layernorm_type)
        self.hf_conv2 = PreNorm(dim, FFN_FN(ffn_name="Gated_Dconv_FeedForward", dim=dim), layernorm_type=layernorm_type)
        self.LLblock = LLBlock(dim=dim)
        # self.LLblock = PreNorm(dim, LocalMSA(dim=dim, window_size=window_size, num_heads=num_heads), layernorm_type=layernorm_type)
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, LocalMSA(dim=dim, window_size=window_size, num_heads=num_heads),
                        layernorm_type=layernorm_type),
                # FSQTransformerLayer(dim=dim,
                #                     num_heads=num_heads,
                #                     category_size=dim,
                #                     num_tokens=128,
                #                     reducted_dim=dim // 2),
                PreNorm(dim, NonLocalMSA(dim=dim, num_heads=num_heads, window_size=window_num),
                        layernorm_type=layernorm_type),
                PreNorm(dim, ChannelAttention(dim=dim, num_heads=num_heads, bias=False),
                        layernorm_type=layernorm_type),
                PreNorm(dim, FFN_FN(ffn_name="Gated_Dconv_FeedForward", dim=dim), layernorm_type=layernorm_type),
                PreNorm(dim, FFN_FN(ffn_name="Gated_Dconv_FeedForward", dim=dim), layernorm_type=layernorm_type),
                PreNorm(dim, FFN_FN(ffn_name="Gated_Dconv_FeedForward", dim=dim), layernorm_type=layernorm_type)

            ]))
        self.r = 6
        self.levels = [8 for _ in range(self.r)]
        #self.levels = [16 for _ in range(self.r)]
        # self.fsq = FSQ(levels=self.levels)

        self.non_linear = nn.Sigmoid()
        #self.project_in = nn.Linear(dim,self.r)
        # self.fea_encoding = nn.Sequential(nn.Conv2d(dim,dim,3,1,1),nn.GELU(),nn.Conv2d(dim,dim,3,1,1))
        # self.pool = nn.AvgPool2d(4, 4)
    def forward(self, x):
        # x = self.fea_encoding(x)
        xh, yh = self.dwt(x)
        xh = self.LLblock(xh)
        LH = self.hf_conv(yh[0][:, :, 0])  # LH
        HL = self.hf_conv(yh[0][:, :, 1])  # HL
        HH = self.hf_conv2(yh[0][:, :, 2])  # HH
        yh_new = [torch.stack([LH, HL, HH], dim=2)]
        x = x +self.idwt((xh, yh_new))

        for (local_msa, nonlocal_msa, ca, ffn1, ffn2, ffn3) in self.blocks:
            x = x + local_msa(x)
            x = x + ffn1(x)
            x = ca(x)
            x = x + ffn2(x)
            x = nonlocal_msa(x)
            x = x + ffn3(x)
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=2,
            window_size=(8, 8),
            window_num=(8, 8),
            num_blocks=1,
            layernorm_type="WithBias",
    ):
        super().__init__()

        self.window_size = window_size
        self.window_num = window_num

        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, LocalMSA(dim=dim, window_size=window_size, num_heads=num_heads),
                        layernorm_type=layernorm_type),
                PreNorm(dim, FFN_FN(ffn_name="Gated_Dconv_FeedForward", dim=dim), layernorm_type=layernorm_type)

            ]))

    def forward(self, x):
        for (local_msa, ffn1) in self.blocks:
            x = x + ffn1(x)
            x = x + local_msa(x)

        return x
class Resblock(nn.Module):
    def __init__(self,dim):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(dim,dim,3,1,1)
        self.conv2 = nn.Conv2d(dim,dim,3,1,1)
        self.gelu = nn.GELU()
    def forward(self, x):
        inter = self.conv1(x)
        inter = self.gelu(inter)
        inter = self.conv2(inter)+x
        return inter
class Encoder(nn.Module):
    def __init__(self,dim):
        super(Encoder, self).__init__()
        self.block1 = Block(dim)
        self.block2 = Block(dim)
        self.down = nn.Conv2d(dim,dim,4,2,1)
        self.gelu = nn.GELU()
    def forward(self, x):
        inter = self.block1(x)
        inter = self.down(inter)
        inter = self.block2(inter)
        return inter
class Decoder(nn.Module):
    def __init__(self,dim):
        super(Decoder, self).__init__()
        self.block1 = Block(dim)
        self.block2 = Block(dim)
        self.up = nn.ConvTranspose2d(dim, dim, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.gelu = nn.GELU()
    def forward(self, x):
        inter = self.block1(x)
        inter = self.up(inter)
        inter = self.block2(inter)
        return inter
def get_WB_filter(size):
    """make a 2D weight bilinear kernel suitable for WB_Conv"""
    ligne = []
    colonne = []
    for i in range(size):
        if (i + 1) <= np.floor(math.sqrt(16)):
            ligne.append(i + 1)
            colonne.append(i + 1)
        else:
            ligne.append(ligne[i - 1] - 1.0)
            colonne.append(colonne[i - 1] - 1.0)
    BilinearFilter = np.zeros(size * size)
    for i in range(size):
        for j in range(size):
            BilinearFilter[(j + i * size)] = (ligne[i] * colonne[j] / 16)
    filter0 = np.reshape(BilinearFilter, (size, size))
    return torch.from_numpy(filter0).float()
def mask_input(GT_image, msfa_size):
    mask = np.zeros((GT_image.shape[0], GT_image.shape[1], msfa_size ** 2), dtype=np.float32)
    for i in range(0,msfa_size):
        for j in range(0,msfa_size):
            mask[i::msfa_size, j::msfa_size, i*msfa_size+j] = 1
    input_image = mask * GT_image
    return input_image
class spa_spe_var4(nn.Module):

    def __init__(self):
        super(spa_spe_var4, self).__init__()
        num_blocks = 6
        dim = 48
        kdim = 512
        moving_average_rate = 0.999
        self.denoiser = nn.ModuleList([])
        for _ in range(num_blocks):
            self.denoiser.append(
                MEMBlock(
                    dim=dim,
                    num_heads=2,
                    window_size=[8,8],
                    window_num=[8,8],
                    layernorm_type="WithBias",
                    num_blocks=1
                )
            )#LNLT(in_dim=16, out_dim=16, dim=16, num_blocks=(1, 1, 1, 1, 1))
        #self.apply(self._init_weights)
        #self.memory = _Memory_Block(dim, kdim, moving_average_rate)
        self.embedding = nn.Conv2d(16,dim,3,1,1)
        self.proj = nn.Conv2d(dim,16,3,1,1)
        #
        # self.memory = nn.ModuleList([])
        # self.alpha = nn.ModuleList([])
        # self.beta = nn.ModuleList([])
        # self.fusion = nn.ModuleList([])
        # for _ in range(num_blocks//2):
        #     self.alpha.append(
        #         nn.Conv2d(dim,dim,3,1,1)
        #     )
        #     self.beta.append(
        #         nn.Conv2d(dim,dim,3,1,1)
        #     )
        #     self.fusion.append(
        #         nn.Conv2d(2*dim,dim,3,1,1)
        #     )
        #     self.memory.append(
        #         _Memory_Block(dim, kdim, moving_average_rate)
        #     )
        # self.alpha = nn.Conv2d(dim,dim,3,1,1)
        # self.beta = nn.Conv2d(dim,dim,3,1,1)
        # self.fusion = nn.Conv2d(dim*2,dim,3,1,1)

        self.apply(self._init_weights)
        self.WB_Conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3, bias=False,
                                 groups=16)
        cout, cin, h, w = self.WB_Conv.weight.data.size()
        self.WB_Conv.weight.data = get_WB_filter(7).view(1, 1, h, w).repeat(cout, cin, 1, 1)
        self.WB_Conv.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, y, mask=None):
        """
        :param y: [b,256,310]
        :param phi: [b,28,256,310]
        :param Phi_PhiT: [b,256,310]
        :return: z_crop: [b,28,256,256]
        """
        alpha_list = []
        beta_list = []
        mask = torch.zeros(y.shape[0],16,y.shape[2],y.shape[3]).to(y.device)
        msfa_size = 4
        for i in range(0, msfa_size):
            for j in range(0, msfa_size):
                mask[:, i * msfa_size + j,i::msfa_size, j::msfa_size] = 1
        input_image = mask * y
        #y1 = y.repeat(1, mask.size(1), 1, 1)
        Xt1 = self.WB_Conv(input_image)
        # print(self.WB_Conv.weight.data)
        fea = self.embedding(Xt1)
        for i in range(len(self.denoiser)):
            # if i<=2:
            #     mem,_ = self.memory[i](fea)
            #     alpha = self.alpha[i](mem)
            #     beta = self.beta[i](mem)
            #     #mem_affine = alpha*fea+beta
            #     alpha_list.append(alpha)
            #     beta_list.append(beta)
            #     #fea = self.fusion(torch.cat([mem_affine,fea],dim=1))
            # if i>2:
            #     mem_affine = alpha_list[5-i] * fea + beta_list[5-i]
            #     fea = self.fusion[i-3](torch.cat([mem_affine,fea],dim=1))
            fea = fea + self.denoiser[i](fea)
        Xt = self.proj(fea) + Xt1
        # Xt1 = Xt.permute(2, 3, 0, 1).cpu().numpy()
        # plt.imshow(Xt1[:, :, 0, 0], cmap='gray')
        # plt.show()
        return Xt
    # def forward(self, y, mask=None):
    #     """
    #     :param y: [b,256,310]
    #     :param phi: [b,28,256,310]
    #     :param Phi_PhiT: [b,256,310]
    #     :return: z_crop: [b,28,256,256]
    #     """
    #     y1 = y.repeat(1, mask.size(1), 1, 1)
    #     Xt = self.WB_Conv(y1 * mask)#+y1 * mask
    #     f0 = self.embedding(Xt)
    #     fea = f0 + self.denoiser[0](f0)
    #     for i in range(len(self.denoiser)-1):
    #         fea = fea + self.denoiser[i+1](fea)
    #     fea = f0 + self.conv(fea)
    #     Xt = self.proj(fea) + Xt
    #
    #
    #     return Xt


if __name__ == '__main__':
    import scipy.io as sio
    # from thop import profile
    import os
    import time
    import matplotlib.pyplot as plt
    from fvcore.nn import FlopCountAnalysis
    import hdf5storage
    from collections import OrderedDict
    from matplotlib.colors import ListedColormap
    from collections import Counter
    def load_raw(filepath):
        mat = hdf5storage.loadmat(filepath)
        img = mat['mosaic']
        return img
    def load_target(filepath):
        mat = hdf5storage.loadmat(filepath)
        # ARAD Dataset
        img = mat['cube']
        norm_factor = mat['norm_factor']

        # Chikusei Dataset
        # data = mat['crop_gt']
        # img = data[0,0]['cube']
        # norm_factor = data[0,0]['norm_factor']
        return img, norm_factor
    def gen_mask(msfa, size, device):
        _, w, h = size
        p = msfa.size(1)
        quotient_w, remainder_w = divmod(w, p)
        quotient_h, remainder_h = divmod(h, p)
        if remainder_w == 0 and remainder_h == 0:
            mask = msfa.repeat(1, quotient_w, quotient_h)
        else:
            mask = torch.zeros(size=size)  # .to(device)
            mask[:, :(quotient_w * p), :(quotient_h * p)] = msfa.repeat(1, quotient_w, quotient_h)
            if remainder_w > 0:
                mask[:, -remainder_w:, :] = mask[:, 0:remainder_w, :]
            if remainder_h > 0:
                mask[:, :, -remainder_h:] = mask[:, :, 0:remainder_h]
        return mask.numpy()


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = spa_spe_var4().cuda()
    # model.load_state_dict(new_state_dict,strict=False)
    input = torch.randn(1, 1, 256, 256).cuda().float()
    # gt = torch.from_numpy(target).unsqueeze(0).permute(0,3,1,2).float()#.cuda()
    mask = torch.randn(1, 16, 256, 256).cuda().float()
    # print(mask.shape)
    # model = model.cuda()
    # print(model)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))

    for i in range(1):
        with torch.no_grad():
            start = time.time()
            x_rec = model(input, mask)
            # print(x1.shape,x2.shape)
            end = time.time()
            print('time:', end - start)
    print(x_rec.shape)
    with torch.no_grad():
        # flops, params = profile(model, inputs=(input, (mask)))
        # print("FLOPs=", str(flops / (1024*1024*1024)) + '{}'.format("G"))
        # print("params=", str(params / (1024*1024)) + '{}'.format("M"))
        #
        flops = FlopCountAnalysis(model, inputs=((input, mask)))
        n_param = sum([p.nelement() for p in model.parameters()])
        print("FLOPs=", str(flops.total() / (1024 * 1024 * 1024)) + '{}'.format("G"))
        print("params=", str(n_param / (1024 * 1024)) + '{}'.format("M"))
    '''os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    height = 256
    width = 256
    msfa_path = 'G:/Dataset/MSFA_16.mat'#'/home/liumengzu/Data/MSFA/MSFA_16.mat'#'/home/liumengzu/Data/MSFA/MSFA_16.mat'#'G:/Dataset/MSFA_16.mat'#'/home/liumengzu/Data/MSFA/MSFA_16.mat'
    raw_path = 'G:/Dataset/ARAD_testing/test_mosaic/ARAD_1K_0901.raw.mat'
    target_path = 'G:/Dataset/ARAD_testing/test_spectral_16/ARAD_1K_0901_16.mat'
    model_path = 'G:/Code/my_test_code/2025/demosaicing/light/model_checkpoint/DERNN_v2_spafsq2v1.pth'
    raw = load_raw(raw_path)
    target,_ = load_target(target_path)
    model_state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    # k为module.xxx.weight, v为权重 多卡训练会在保存的model前面加module，这里把module去掉了
    for k, v in model_state_dict['model'].items():
        name = k[7:]  # 截取`module.`后面的xxx.weight
        new_state_dict[name] = v
    mask = gen_mask(torch.from_numpy(sio.loadmat(msfa_path)['msfa']).permute(2, 0, 1),
                         (16, 480, 512), 'cuda')
    print(raw.shape,target.shape)
    model = spa_memv3_polaq1().cuda()
    #model.load_state_dict(new_state_dict,strict=False)
    input = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).cuda().float()
    gt = torch.from_numpy(target).unsqueeze(0).permute(0,3,1,2).float()#.cuda()
    mask = torch.from_numpy(mask).unsqueeze(0).cuda().float()
    #print(mask.shape)
    # model = model.cuda()
    #print(model)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))

    for i in range(1):
        with torch.no_grad():
            start = time.time()
            x_rec = model(input, mask)
            # print(x1.shape,x2.shape)
            end = time.time()
            print('time:', end - start)
    print(x_rec.shape)
    with torch.no_grad():
        # flops, params = profile(model, inputs=(input, (mask)))
        # print("FLOPs=", str(flops / (1024*1024*1024)) + '{}'.format("G"))
        # print("params=", str(params / (1024*1024)) + '{}'.format("M"))
        #
        flops = FlopCountAnalysis(model, inputs=((x, m)))
        n_param = sum([p.nelement() for p in model.parameters()])
        print("FLOPs=", str(flops.total() / (1024 * 1024 * 1024)) + '{}'.format("G"))
        print("params=", str(n_param / (1024 * 1024)) + '{}'.format("M"))
    # flops, params = profile(model, (input, mask,))
    # print('flops: %.2f M, params: %.2f M' % (flops / (1024 * 1024 * 1024), params / (1024 * 1024 * 1024)))  # 16464.41 M, params: 0.92 M

    # 9stg: flops: 83193.33 M, params: 4.75 M
    # 2stg: flops: 19361.43 M, params: 1.09 M'''

