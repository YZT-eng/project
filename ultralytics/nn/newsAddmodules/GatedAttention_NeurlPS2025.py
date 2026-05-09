import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ultralytics.nn.modules import C2f, C3, Conv
 
__all__ = ['GatedAttention','C3k2_GatedAttention']
 
# MultiHeadAttention 类定义
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
 
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
 
    def forward(self, x, mask=None):
        """
        x: (B, T, d_model)
        """
        B, T, _ = x.shape
 
        q = self.w_q(x)  # (B, T, d_model)
        k = self.w_k(x)
        v = self.w_v(x)
 
        # 拆成多头
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
 
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
 
        # scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
 
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
 
        attn_probs = F.softmax(attn_scores, dim=-1)  # 每一行和=1
 
        y = torch.matmul(attn_probs, v)  # (B, n_heads, T, d_head)
 
        # 合并多头
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.w_o(y)  # (B, T, d_model)
        return out, attn_probs  # 返回最终输出和注意力权重
 
 
# GatedAttention 类定义
class GatedAttention(nn.Module):
    def __init__(self, d_model, n_heads=8,headwise=True, elementwise=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
 
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
 
        # === 门控相关部分 ===
        self.headwise = headwise
        self.elementwise = elementwise
 
        if headwise and not elementwise:
            gate_out_dim = n_heads
        elif headwise and elementwise:
            gate_out_dim = d_model
        else:
            raise NotImplementedError
 
        self.gate_proj = nn.Linear(d_model, gate_out_dim)
        # 一般还会有 pre-norm，这里简单用 LayerNorm 表示
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')
        """
        x: (B, T, d_model)
        """
        B, T, _ = x.shape
        x_norm = self.ln(x)  # pre-norm
 
        q = self.w_q(x_norm)
        k = self.w_k(x_norm)
        v = self.w_v(x_norm)
 
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
 
        q = split_heads(q)   # (B, H, T, Dh)
        k = split_heads(k)
        v = split_heads(v)
        mask = None
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
 
        attn_probs = F.softmax(attn_scores, dim=-1)
        y = torch.matmul(attn_probs, v)  # (B, H, T, Dh) = SDPA 输出 Y
 
        # === Gated Attention 关键部分 ===
        gate_in = x_norm  # (B, T, d_model)
 
        gate_raw = self.gate_proj(gate_in)  # (B, T, gate_out_dim)
        gate = torch.sigmoid(gate_raw)
 
        if self.headwise and self.elementwise:
            gate = gate.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        elif self.headwise and not self.elementwise:
            gate = gate.transpose(1, 2).unsqueeze(-1)
        else:
            raise NotImplementedError
        # 乘性门控：Y' = Y ⊙ σ(X W_θ)
        y = y * gate
        # 合并多头并做输出投影
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.w_o(y)
        return rearrange(out, 'b (h w) c->b c h w', h=h)
class Bottleneck_GatedAttention(nn.Module):
    """Standard bottleneck."""
 
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = GatedAttention(c2)
 
    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))
 
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
 
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_GatedAttention(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
 
class C3k2_GatedAttention(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
 
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_GatedAttention(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
 
 


