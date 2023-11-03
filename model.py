import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # Number of heads for the query and key
    n_kv_heads: Optional[int] = None  # number of heads for the K and V
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-6  # for the rms so it's never 0

    # KV cache args
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # implementation is from the paper as embedding must be even
    assert head_dim % 2 == 0, "head_dum must be divisible by 2"
    # Build the theta params
    # According to the formula theta_i = 10000 ^ (-2(-i-1)/dim) for i = [1,2...dim/2]
    # Shape: (Head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Construct the positions (the "m" parameter)
    # Shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product
    # Shape: (seq_len) outer_product * (Head_dum / 2) -> (Seq_len, Head_dim / 2)
    freqs = torch.out(m, theta).float()
    # Compute complex numbers in the polar form c = R * exp(i * m * theta), where R = 1 as follows:
    # (Seq_Len, Head_dim / 2) -> (Seq_Len, Head_dim /2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, Seq_Len, H, Head_dim) -> (B, Seq_Len, H, Head_dim / 2, 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (Seq_Len, Head_dim / 2) -> (1, Seq_Len, 1, Head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, Seq_Len, H, Head_dim / 2) * (1, Seq_Len, 1, Head_dim / 2) = (B, Seq_Len, H, Head_dim /2)
    x_rotated = x_complex * freqs_complex
    # (B, Seq_Len, H, Head_dim / 2) -> (B, Seq_Len, H, Head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_dim/2, 2) -> (B, Seq_Len, H, Head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization before the self-attention block
        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        # Normalization before the feed-forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, Seq_len, Dim) + (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        h = x + \
            self.attention.forward(
                self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Gamma param
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


class Transformer(nn.module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "vocab_size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (Batch,seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Makes sure only one token is passed at a time"

        # (Batch,seq_len) -> (Batch,seq_len,dim)
        h = self.tok_embeddings(tokens)

        # retrieve the pairs (m,theta) corresponding to the positions [start_pos, start_pos+seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]

        # Consecutively apply all encoder blocks
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
