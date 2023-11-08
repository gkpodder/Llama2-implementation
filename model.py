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


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            # (B, Seq_Len, N_KV_Heads, Head_dim)
            x[:, :, None, :, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
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


class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        # indicates the number of heads for the key and values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the keys and values should be repeated to match the head of the queries
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads *
                            self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads *
                            self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim)

        # Apply the Wq, Wk, Wv matrices to queries, keys and values
        # (B, 1, Dim) -> (B, 1, H_Q * Head_dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B, 1, H_Q * Head_dim) -> (B, 1, H_Q, Head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_dim) -> (B, 1, H_KV, Head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Does not change the shape of the tensor
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Replace the entry in the cache for this token
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv

        # Retrieve all the cached keys and values so far
        # (B, Seq_Len, H_KV, Head_dim)
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]

        # Repeat the heads of the K and V to reach the number of heads of the queries
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, Seq_Len, Head_dim) -> (B, H_Q, 1, Head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_dim) * (B, H_Q, Head_dim, Seq_len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / \
            math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len) * (B, H_Q, Seq_Len_KV, Head_dim) -> (B, H_Q, 1, Head_dim)
        output = torch.matmul(scores, values)

        # (B, H_Q, 1, Head_dim) -> (B, 1, H_Q, Head_dim) -> (B, 1, dim)
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1))
        return self.wo(output)


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
