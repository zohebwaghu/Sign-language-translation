"""
F2 — Temporal Transformer Encoder.

Contextualises per-frame features across the temporal dimension so the
model understands gesture dynamics rather than treating frames independently.

Architecture:
  (B, T, d_model)  → sinusoidal PE → 4-layer TransformerEncoder → LayerNorm → (B, T, d_model)
"""

import math
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import D_MODEL, N_HEADS, N_ENC_LAYERS, DIM_FEEDFORWARD, DROPOUT, MAX_FRAMES


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learned) sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Registered as a buffer so it moves with the module to GPU/CPU without
    being treated as a trainable parameter.
    """

    def __init__(self, d_model: int = D_MODEL, max_len: int = 512, dropout: float = DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)           # (max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()   # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                         # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model) with positional encoding added and dropout applied.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    """
    4-layer Transformer Encoder over the time dimension.

    Accepts a key_padding_mask to prevent attention to padded (zero) frames,
    ensuring padding doesn't corrupt the contextualised representations.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_ENC_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        max_len: int = 512,
    ):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # expects (B, T, d_model) — matches our layout
            norm_first=False,   # post-norm (standard)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)  per-frame visual features.
            src_key_padding_mask: (B, T) bool tensor — True for PADDED positions
                                  (torch convention: True = ignore this position).

        Returns:
            (B, T, d_model)  temporally contextualised features.
        """
        x = self.pos_enc(x)                        # add positional encoding
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x
