import math
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from ultralytics.nn.modules import C3, DSConv, C2f

#论文地址： https://arxiv.org/pdf/2507.02748  ICCV 2025 CV任务通用
__all__ = ['MultipoleAttention','C2f_MultipoleAttention','C3k2_MultipoleAttention',
           'C3k2_MultipoleBlock','C2f_MultipoleBlock',
           'C2PSA_MultipoleAttention','C2PSA_MultipoleAttention_DyT',
           'DSC3k2_MultipoleAttention','DSC3k2_MultipoleBlock'
           ]
class FeedForward(nn.Module):
    """
    MLP block with pre-layernorm, GELU activation, and dropout.
    """

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    """
    Global multi-head self-attention block with optional projection.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = (
            dim_head * heads
        )  # the total dimension used inside the multi-head attention. When concatenating all heads, the combined dimension is dim_head × heads
        project_out = not (
            heads == 1 and dim_head == dim
        )  # if we're using just 1 head and its dimension equals dim, then we can skip the final linear projection.

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)  # Applies LN over the last dimension.

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """
        Expected input shape: [B, L, C]
        """
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(
            3, dim=-1
        )  # chunk splits into 3 chuncks along the last dimension, this gives Q, K, V
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LocalAttention2D(nn.Module):
    """
    Windowed/local attention for 2D grids using unfold & fold.
    """

    def __init__(self, kernel_size, stride, dim, heads, dim_head, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride  # kernel_size
        self.dim = dim
        padding = 0

        self.norm = nn.LayerNorm(dim)

        self.Attention = AttentionBlock(
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
        )

        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        x = rearrange(
            x, "B H W C -> B C H W"
        )  # Rearrange to [B, C, H, W] for unfolding

        # unfold into local 2D patches
        patches = self.unfold(x)  # [B, C*K*K, L] where W is the number of patches

        patches = rearrange(
            patches,
            "B (C K1 K2) L -> (B L) (K1 K2) C",
            K1=self.kernel_size,
            K2=self.kernel_size,
        )

        patches = self.norm(patches)

        # Intra-Window self.attention
        out = self.Attention(patches)  # [B*L, K*K, C]

        # Reshape back to [B, C*K*K, L]
        out = rearrange(
            out,
            "(B L) (K1 K2) C -> B (C K1 K2) L",
            B=B,
            K1=self.kernel_size,
            K2=self.kernel_size,
        )

        # Fold back to [B, C, H, W] with overlap
        fold = nn.Fold(
            output_size=(H, W), kernel_size=self.kernel_size, stride=self.stride
        )
        out = fold(out)

        # Normalize overlapping regions
        norm = self.unfold(torch.ones((B, 1, H, W), device=x.device))  # [B, K*K, L]
        norm = fold(norm)  # [B, 1, H, W]
        out = out / norm

        # Reshape to [B, H, W, C]
        out = rearrange(out, "B C H W -> B H W C")

        return out


class MultipoleAttention(nn.Module):  #B C H W Multipole Attention 多极注意力
    """
    Hierarchical local attention across multiple scales with down/up-sampling.
    """

    def __init__(
        self,
        in_channels,
        image_size,
        local_attention_kernel_size=2,
        local_attention_stride=2,
        downsampling="conv",
        upsampling= "conv",
        sampling_rate=2,
        heads=4,
        dim_head=16,
        dropout=0.1,
        channel_scale=1,
    ):
        super().__init__()

        # self.levels = int(math.log(image_size, sampling_rate))  # math.log(x, base)
        self.levels = 2
        channels_conv = [in_channels * (channel_scale**i) for i in range(self.levels)]

        # A shared local attention layer for all levels
        self.Attention = LocalAttention2D(
            kernel_size=local_attention_kernel_size,
            stride=local_attention_stride,
            dim=channels_conv[0],
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        if downsampling == "avg_pool":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.AvgPool2d(kernel_size=sampling_rate, stride=sampling_rate),
                Rearrange("B C H W -> B H W C"),
            )

        elif downsampling == "conv":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Conv2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )

        if upsampling == "avg_pool":
            current = image_size

            for _ in range(self.levels):
                assert (
                    current % sampling_rate == 0
                ), f"Image size not divisible by sampling_rate size at level {_}: current={current}, sampling_ratel={sampling_rate}"
                current = current // sampling_rate

            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Upsample(scale_factor=sampling_rate, mode="nearest"),
                Rearrange("B C H W -> B H W C"),
            )

        elif upsampling == "conv":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.ConvTranspose2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )

    def forward(self, x):
        # x: [B, H, W, C], returns the same shape
        # Level 0
        x = x.permute(0,2,3,1)
        x_in = x

        x_out = []
        x_out.append(self.Attention(x_in))

        # Levels from 1 to L
        for l in range(1, self.levels):
            x_in = self.down(x_in)
            x_out_down = self.Attention(x_in)
            x_out.append(x_out_down)

        res = x_out.pop()
        for l, out_down in enumerate(x_out[::-1]):
            res = out_down + (1 / (l + 1)) * self.up(res)

        return res.permute(0,3,2,1)

class Multipole_Attention_BHWC(nn.Module):  #B H W C Multipole Attention 多极注意力
    """
    Hierarchical local attention across multiple scales with down/up-sampling.
    """

    def __init__(
        self,
        in_channels,
        image_size,
        local_attention_kernel_size=2,
        local_attention_stride=2,
        downsampling="conv",
        upsampling= "conv",
        sampling_rate=2,
        heads=4,
        dim_head=16,
        dropout=0.1,
        channel_scale=1,
    ):
        super().__init__()

        # self.levels = int(math.log(image_size, sampling_rate))  # math.log(x, base)
        self.levels = 2
        channels_conv = [in_channels * (channel_scale**i) for i in range(self.levels)]

        # A shared local attention layer for all levels
        self.Attention = LocalAttention2D(
            kernel_size=local_attention_kernel_size,
            stride=local_attention_stride,
            dim=channels_conv[0],
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        if downsampling == "avg_pool":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.AvgPool2d(kernel_size=sampling_rate, stride=sampling_rate),
                Rearrange("B C H W -> B H W C"),
            )

        elif downsampling == "conv":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Conv2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )

        if upsampling == "avg_pool":
            current = image_size

            for _ in range(self.levels):
                assert (
                    current % sampling_rate == 0
                ), f"Image size not divisible by sampling_rate size at level {_}: current={current}, sampling_ratel={sampling_rate}"
                current = current // sampling_rate

            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Upsample(scale_factor=sampling_rate, mode="nearest"),
                Rearrange("B C H W -> B H W C"),
            )

        elif upsampling == "conv":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.ConvTranspose2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )

    def forward(self, x):
        # x: [B, H, W, C], returns the same shape
        # Level 0
        x_in = x

        x_out = []
        x_out.append(self.Attention(x_in))

        # Levels from 1 to L
        for l in range(1, self.levels):
            x_in = self.down(x_in)
            x_out_down = self.Attention(x_in)
            x_out.append(x_out_down)

        res = x_out.pop()
        for l, out_down in enumerate(x_out[::-1]):
            res = out_down + (1 / (l + 1)) * self.up(res)

        return res

class MultipoleBlock(nn.Module): # 多极注意力块
    """
    Transformer block stacking multiple Multipole_Attention2D + FeedForward layers.
    """

    def __init__(
        self,
        in_channels,
        image_size,
        kernel_size=2,  # Local attention patch size
        local_attention_stride=2,  # stride（与 kernel_size 相同）
        downsampling= "conv",  # 使用卷积做下采样
        upsampling= "conv",  # 使用反卷积做上采样
        sampling_rate=2,  # 每层下采样/上采样缩放因子
        depth=2,  # 堆叠层数
        heads=4,  # 注意力头数
        dim_head=16,  # 每个头的维度
        att_dropout=0.1,  # 注意力 dropout
        channel_scale=1,  # 多尺度通道扩展倍率（设为1保持通道数一致）

    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.layers = nn.ModuleList([])
        mlp_dim = int(4*in_channels)  # FeedForward中间层维度
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Multipole_Attention_BHWC(
                            in_channels,
                            image_size,
                            kernel_size,
                            local_attention_stride,
                            downsampling,
                            upsampling,
                            sampling_rate,
                            heads,
                            dim_head,
                            att_dropout,
                            channel_scale,
                        ),
                        FeedForward(in_channels, mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        """
        Expected input shape: [B, H, W, C]
        """
        x = x.permute(0,2,3,1)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x).permute(0,3,1,2)

#----------------
#------------
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Bottleneck_MultipoleAttention(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, HW,shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = MultipoleAttention(c2,HW)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))

class C2f_MultipoleAttention(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1,HW=20, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_MultipoleAttention(self.c, self.c,HW, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
class C2f_MultipoleBlock(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""


    def __init__(self, c1, c2, n=1,HW=20, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(MultipoleBlock(self.c,HW) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
class C3k_MultipoleAttention(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1,HW=20, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, HW,shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_MultipoleAttention(c_, c_, HW,shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
class C3k_MultipoleBlock(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1,HW=20, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, HW,shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MultipoleBlock(c_,HW) for _ in range(n)))
class C3k2_MultipoleAttention(C2f_MultipoleAttention):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1,HW=20, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, HW,shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_MultipoleAttention(self.c, self.c, 2, HW,shortcut, g) if c3k else Bottleneck_MultipoleAttention(self.c, self.c, HW,shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
class C3k2_MultipoleBlock(C2f_MultipoleAttention):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1,HW=20, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, HW,shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_MultipoleBlock(self.c, self.c, 2, HW,shortcut, g) if c3k else MultipoleBlock(self.c,HW) for _ in range(n)
        )

#-----对YOLOv11中的 C2PSA模块改进--------
class PSABlock_MultipoleAttention(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, HW=20,  shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = MultipoleAttention(c, HW)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""

        x = x + self.attn(x)if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last=False, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x
class PSABlock_TSSA_DyT(PSABlock_MultipoleAttention):
    def __init__(self, c, HW, shortcut=True):
        super().__init__(c, HW,  shortcut)

        self.dyt1 = DynamicTanh(normalized_shape=c)
        self.dyt2 = DynamicTanh(normalized_shape=c)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x)) if self.add else self.attn(self.dyt1(x))
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        return x

class C2PSA_MultipoleAttention(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, HW=20,e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock_MultipoleAttention(self.c, HW) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class C2PSA_MultipoleAttention_DyT(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, HW=20,e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock_TSSA_DyT(self.c, HW) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
#YOLOv13---------------
class DSBottleneck_MultipoleAttention(nn.Module):
    def __init__(self, c1, c2, HW,shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DSConv(c1, c_, k1, s=1, p=None, d=1)
        self.cv2 = DSConv(c_, c2, k2, s=1, p=None, d=d2)
        self.add = shortcut and c1 == c2
        self.att = MultipoleAttention(c2,HW)

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return  self.att(x + y) if self.add else  self.att(y)

class DSC3k_MultipoleAttention(C3):
    def __init__( self,c1,c2,n=1,HW=20,shortcut=True,g=1,e=0.5,k1=3,k2=5,d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)

        self.m = nn.Sequential(
            *(
                DSBottleneck_MultipoleAttention(
                    c_, c_,HW,
                    shortcut=shortcut,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )
        )
class DSC3k_MultipoleBlock(C3):
    def __init__( self,c1,c2,n=1,HW=20,shortcut=True,g=1,e=0.5,k1=3,k2=5,d2=1):
        super().__init__(c1, c2, n,shortcut, g, e)
        c_ = int(c2 * e)

        self.m = nn.Sequential(
            *(
                MultipoleBlock(c_,HW)
                for _ in range(n)
            )
        )
class DSC3k2_MultipoleAttention(C2f):

    def __init__(self,c1,c2,n=1,HW=20,dsc3k=False,e=0.5,g=1,shortcut=True,k1=3,k2=7,d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:
            self.m = nn.ModuleList(DSC3k_MultipoleAttention(self.c, self.c,n=2,HW=HW,shortcut=shortcut,g=g,e=1.0,k1=k1,k2=k2,d2=d2)for _ in range(n))
        else:
            self.m = nn.ModuleList(DSBottleneck_MultipoleAttention(self.c, self.c,HW,shortcut=shortcut,e=1.0,k1=k1,k2=k2,d2=d2)for _ in range(n))

class DSC3k2_MultipoleBlock(C2f):

    def __init__(self,c1,c2,n=1,HW=20,dsc3k=False,e=0.5,g=1,shortcut=True,k1=3,k2=7,d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:
            self.m = nn.ModuleList(DSC3k_MultipoleBlock(self.c, self.c,n=2,HW=HW,shortcut=shortcut,g=g,e=1.0,k1=k1,k2=k2,d2=d2) for _ in range(n))
        else:
            self.m = nn.ModuleList(MultipoleBlock(self.c,HW) for _ in range(n))


# AAAI2025 HFP模块的二次创新，HLPFM在我的二次创新模块交流群，可以直接去跑实验发小论文！
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input = torch.randn(1, 32, 64, 64)
    # 创建 Multipole_Attention模块
    MA = MultipoleAttention(in_channels=32,image_size=64)  #第一个模块 ：多极注意力
    # 将输入图像传入Multipole_Attention模块进行处理
    output = MA(input)
    # 输出结果的形状
    # 打印输入和输出的形状
    print('Ai缝合即插即用模块永久更新_Multipole_Attention_input_size:', input.size())
    print('Ai缝合即插即用模块永久更新_Multipole_Attention_output_size:', output.size())

    MABlock = DSC3k2_MultipoleBlock(32,32,1,64)  #第二个模块 ：多极注意力块
    # 将输入图像传入MultipoleBlock模块进行处理
    output = MABlock (input)
    # 输出结果的形状
    # 打印输入和输出的形状
    print('Ai缝合即插即用模块永久更新_MultipoleBlock_input_size:', input.size())
    print('Ai缝合即插即用模块永久更新_MultipoleBlock_output_size:', output.size())


