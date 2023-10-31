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
