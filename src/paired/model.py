import functools
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn


def pairs(channels):
    return zip(channels[:-1], channels[1:])


def default_norm(num_groups, in_channels):
    return nn.GroupNorm(num_groups, in_channels)


def default_act():
    return nn.SiLU()


def norm_act_drop_conv(in_channels, out_channels, num_groups, p):
    """builds layers of norm, act, drop, conv order for resblocks"""
    norm = default_norm(num_groups, in_channels)
    act = default_act()
    drop = nn.Dropout2d(p)
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    if p > 0:
        return nn.Sequential(norm, act, drop, conv)
    else:
        return nn.Sequential(norm, act, conv)


class SinusoidalPositionEmbeddings(nn.Module):
    r"""Transformer position encoding

    Args:
        dim (int): number of dimensions of the position embedding, :math:`d_\text{emb}`
    """

    embeddings: Tensor

    def __init__(self, dim) -> None:
        super().__init__()

        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = embeddings.unsqueeze(dim=0)

        self.register_buffer("embeddings", embeddings)

    def forward(self, t):
        r"""
        Args:
            t (torch.Tensor): timestep of shape :math:`(N,)`

        Returns:
            (torch.Tensor): Positional Embedding of shape :math:`(N, d_\text{emb})`
        """

        embeddings = t.unsqueeze(dim=1) * self.embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def DownSample(c_in, c_out):
    """Downsample blocks

    Args:
        c_in (int): number of input channels
        c_out (int): number of output channels

    Returns:
        (nn.Conv2d): down sampling layer using 2d convolutions
    """

    return nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)


class UpSample(nn.Module):
    """Upsample blocks

    Args:
        c_in (int): number of input channels
        c_out (int): number of output channels

    """

    def __init__(self, c_in, c_out) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2.0)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        r"""
        Args:
            x (torch.Tensor): image of shape :math:`(N, C_\text{in}, H, W)`

        Returns:
            (torch.Tensor): downsampled feature map of shape :math:`(N, C_\text{out}, H//2, W//2)`
        """
        x = self.upsample(x)
        return self.conv(x)


class MultiHeadAttention(nn.Module):
    r"""Self Attention with groupnorm

    Args:
        dim (int): equivalent to :math:`d_\text{model}`
        num_groups (int): number of groups in :code:`nn.GroupNorm`
    """

    def __init__(self, dim, num_groups, num_heads):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads

        self.norm = default_norm(num_groups, dim)

        self.scale = dim**-0.5
        self.qkv_proj = nn.Conv2d(dim, 3 * dim, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward_attention(self, x):
        qkv = self.qkv_proj(x)
        qkv = rearrange(qkv, "b (head c) h w -> (b head) (h w) c", head=self.num_heads)
        query, key, value = qkv.chunk(3, dim=2)
        key = rearrange(key, "bhead hw c -> bhead c hw") * self.scale
        score = torch.bmm(query, key)
        attention = F.softmax(score, dim=2)
        out = torch.bmm(attention, value)
        out = rearrange(
            out, "(head b) (h w) c -> b (head c) h w", head=self.num_heads, h=x.size(2)
        )
        return self.proj(out)

    def forward(self, x):
        r"""
        Args:
            x (torch.Tensor): image of shape :math:`(N, C_\text{in}, H, W)`

        Returns:
            (torch.Tensor): feature maps of shape :math:`(N, C_\text{in}, H, W)`
        """
        h = self.norm(x)
        h = self.forward_attention(h)
        return h + x


class ResBlock(nn.Module):
    """3x3 basic resblocks with group norm, dropout and timestep embeddings

    Args:
        c_in (int): number of input channels
        c_out (int): number of output channels
        with_attention (bool): whether to add attention block
        emb_dim (int): input timestep embedding dimension
        num_groups (int): number of groups in :code:`nn.GroupNorm`
        p (float): dropout rate in :code:`nn.Dropout2d`
    """

    def __init__(
        self,
        c_in,
        c_out,
        with_attention=False,
        num_heads=4,
        emb_dim=512,
        num_groups=32,
        p=0.1,
    ) -> None:
        super().__init__()

        self.conv1 = norm_act_drop_conv(c_in, c_out, num_groups, p=0.0)
        self.norm = default_norm(num_groups, c_out)

        self.condition = nn.Sequential(
            nn.Linear(emb_dim, c_out * 2),
            Rearrange("b c -> b c 1 1"),
        )

        self.conv2 = norm_act_drop_conv(c_out, c_out, num_groups, p)[1:]

        if c_in != c_out:
            self.residual = nn.Conv2d(c_in, c_out, kernel_size=1)
        else:
            self.residual = nn.Identity()

        if with_attention:
            self.attention = MultiHeadAttention(c_out, num_groups, num_heads=num_heads)
        else:
            self.attention = nn.Identity()

    def forward(self, x, c):
        r"""
        Args:
            x (torch.Tensor): image of shape :math:`(N, C_\text{in}, H, W)`
            c (torch.Tensor): timestep embedding of shape :math:`(N, d_\text{emb})`

        Returns:
            (torch.Tensor): feature map of shape :math:`(N, C_\text{out}, H, W)`
        """

        h = torch.utils.checkpoint.checkpoint(self.conv1, x, use_reentrant=False)
        shift, scale = self.condition(c).chunk(2, dim=1)
        h = self.norm(h) * (scale + 1) + shift
        h = self.conv2(h)
        h += self.residual(x)
        h = torch.utils.checkpoint.checkpoint(self.attention, h, use_reentrant=False)
        return h


class UNet(nn.Module):
    r"""U-Net for predicting noise in images and learning variance

    Args:
        in_channels (int): input channels of image
        pos_dim (int): dimension of position embedding
        emb_dim (int): dimension of timestep embedding
        num_groups (int): number of groups in :code:`nn.GroupNorm`
        dropout (float): dropout rate in :code:`nn.Dropout2d`
        channels_per_depth (Tuple[int, ...]): channels per depth
        num_blocks (int): number of resblocks to use in each depth
        attention_depths (Tuple[int, ...]): depths to use attention blocks
    """

    def __init__(
        self,
        x_channels=141,
        y_channels=1,
        pos_dim=128,
        emb_dim=512,
        num_groups=32,
        dropout=0.3,
        channels_per_depth=(256, 256, 256, 256),
        num_blocks=3,
        attention_depths=(2, 3),
    ):
        super().__init__()

        # configure channels, downsample_layers
        input_dim = channels_per_depth[0]
        channels = [input_dim]
        for c in channels_per_depth:
            channels += [c] * num_blocks

        max_depth = len(channels_per_depth)
        downsample_layers = [num_blocks * i for i in range(1, max_depth)]
        self.condition = nn.Sequential(
            SinusoidalPositionEmbeddings(pos_dim),
            nn.Linear(pos_dim, emb_dim),
            default_act(),
            nn.Linear(emb_dim, emb_dim),
            default_act(),
        )

        self.input_conv = nn.Conv2d(
            x_channels + y_channels, channels[0], kernel_size=3, stride=1, padding=1
        )

        default_resblock = functools.partial(
            ResBlock, emb_dim=emb_dim, num_groups=num_groups, p=dropout
        )

        down_layers = []

        depth = 1
        for i, (c_in, c_out) in enumerate(pairs(channels)):
            layer_num = i + 1

            down_layers += [default_resblock(c_in, c_out, depth in attention_depths)]

            if layer_num in downsample_layers:
                down_layers += [DownSample(c_out, c_out)]

                depth += 1

        depth = max_depth
        # if last down_layer is DownSample
        if down_layers[-1] == len(channels) - 1:
            up_layers = [UpSample(channels[-1], channels[-1])]
            depth -= 1
        else:
            up_layers = []

        for i, (c_in, c_out) in enumerate(pairs(channels[::-1])):
            with_attention = depth in attention_depths
            layer_num = len(channels) - 1 - i

            up_layers += [default_resblock(2 * c_in, c_out, with_attention)]

            if (layer_num - 1) in downsample_layers:
                up_layers += [
                    default_resblock(2 * c_out, c_out, with_attention),
                    UpSample(c_out, c_out),
                ]

                depth -= 1

        up_layers += [
            default_resblock(2 * channels[0], channels[0], 1 in attention_depths)
        ]

        self.down_layers = nn.ModuleList(down_layers)
        self.up_layers = nn.ModuleList(up_layers)

        c_out = channels[-1]
        self.middle_layers = nn.ModuleList(
            [
                default_resblock(c_out, c_out, with_attention=True),
                default_resblock(c_out, c_out, with_attention=False),
            ]
        )

        self.output_conv = norm_act_drop_conv(
            channels[0], x_channels + y_channels, num_groups, p=0.0
        )

        self.x_channels = x_channels
        self.y_channels = y_channels

    def forward(self, x, y, c):
        r"""Predicts noise from x

        Args:
            x (torch.Tensor): image of shape :math:`(N, C, H, W)`
            y (torch.Tensor): image of shape :math:`(N, C, H, W)`
            c (torch.Tensor): timestep of shape :math:`(N,)`

        Returns:
            (torch.Tensor): estimated noise in input image x
        """
        t = self.condition(c)

        x = torch.cat((x, y), dim=1)

        x = self.input_conv(x)

        outputs = [x]

        for f in self.down_layers:
            if isinstance(f, ResBlock):
                x = f(x, t)
            else:
                x = f(x)

            outputs.append(x)

        for f in self.middle_layers:
            x = f(x, t)

        for f in self.up_layers:
            if isinstance(f, ResBlock):
                x = torch.cat([x, outputs.pop()], dim=1)
                x = f(x, t)
            else:
                x = f(x)

        x = self.output_conv(x)
        x, y = torch.split(
            x, split_size_or_sections=(self.x_channels, self.y_channels), dim=1
        )
        return x, y
