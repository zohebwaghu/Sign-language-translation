"""
F3 — Transformer Decoder with autoregressive generation.

Built from scratch (no pretrained LM) — our novelty is the
landmark-aware spatial attention in the encoder, not a pretrained decoder.

Supports:
  - Teacher forcing during training
  - Greedy decoding at inference
  - Beam search decoding (k=4, configurable) — typically +1-2 BLEU over greedy
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import (
    D_MODEL, N_HEADS, N_DEC_LAYERS, DIM_FEEDFORWARD, DROPOUT,
    PAD_ID, BOS_ID, EOS_ID, MAX_TEXT_LEN, BEAM_SIZE,
)


class TextDecoder(nn.Module):
    """
    4-layer Transformer Decoder with:
      - Token + positional embeddings
      - Causal (upper-triangular) self-attention mask
      - Cross-attention over temporal encoder output
      - Output projection to vocab_size logits
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_DEC_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        max_len: int = MAX_TEXT_LEN,
        pad_id: int = PAD_ID,
    ):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.pad_id     = pad_id
        self.max_len    = max_len

        # ── Embeddings ───────────────────────────────────────────────────
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb   = nn.Embedding(max_len + 1, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # ── Decoder layers ───────────────────────────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── Output projection ────────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Weight tying (token_emb ↔ output_proj) reduces parameters ~15%
        self.output_proj.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask: True = IGNORE (torch convention)."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Combine token + learned positional embeddings."""
        B, S = token_ids.shape
        positions = torch.arange(S, device=token_ids.device).unsqueeze(0)   # (1, S)
        return self.emb_dropout(self.token_emb(token_ids) + self.pos_emb(positions))

    # ── Forward (training) ────────────────────────────────────────────────

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass.

        Args:
            tgt:    (B, S)    Target token IDs — input = tgt[:, :-1], target = tgt[:, 1:].
                              The caller handles the shift; pass the full sequence here
                              and let the causal mask do the work.
            memory: (B, T, d_model)  Encoder output (temporal features).
            tgt_key_padding_mask:    (B, S) True for PAD positions in tgt.
            memory_key_padding_mask: (B, T) True for padded frames in memory.

        Returns:
            logits: (B, S, vocab_size)
        """
        B, S = tgt.shape
        causal = self._causal_mask(S, tgt.device)   # (S, S)

        tgt_emb = self._embed(tgt)                   # (B, S, d_model)

        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )                                            # (B, S, d_model)

        return self.output_proj(out)                 # (B, S, vocab_size)

    # ── Inference: greedy decoding ────────────────────────────────────────

    @torch.no_grad()
    def generate_greedy(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        max_len: int = MAX_TEXT_LEN,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
    ) -> torch.Tensor:
        """
        Greedy autoregressive decoding.

        Args:
            memory: (B, T, d_model)

        Returns:
            token_ids: (B, max_len) including BOS, padded with PAD after EOS.
        """
        B = memory.size(0)
        device = memory.device

        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(
                generated, memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )                                           # (B, S, V)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            next_token = next_token.masked_fill(done.unsqueeze(1), PAD_ID)
            generated = torch.cat([generated, next_token], dim=1)
            done = done | (next_token.squeeze(1) == eos_id)
            if done.all():
                break

        # Pad to max_len
        if generated.size(1) < max_len:
            pad = torch.full((B, max_len - generated.size(1)), PAD_ID, dtype=torch.long, device=device)
            generated = torch.cat([generated, pad], dim=1)

        return generated[:, :max_len]

    # ── Inference: beam search decoding ───────────────────────────────────

    @torch.no_grad()
    def generate_beam(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        max_len: int = MAX_TEXT_LEN,
        beam_size: int = BEAM_SIZE,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
        length_penalty: float = 0.6,
    ) -> torch.Tensor:
        """
        Beam search decoding (+1-2 BLEU over greedy).

        Processes one sample at a time for simplicity.

        Args:
            memory:    (B, T, d_model)
            beam_size: Number of beams.

        Returns:
            token_ids: (B, max_len) best beam per sample.
        """
        B = memory.size(0)
        device = memory.device
        results = []

        for b in range(B):
            mem = memory[b:b+1]                      # (1, T, d_model)
            mem_mask = (
                memory_key_padding_mask[b:b+1]
                if memory_key_padding_mask is not None else None
            )

            # Beams: list of (score, token_ids_list)
            beams = [(0.0, [bos_id])]
            completed = []

            for step in range(max_len - 1):
                candidates = []
                for score, seq in beams:
                    if seq[-1] == eos_id:
                        completed.append((score, seq))
                        continue

                    tgt = torch.tensor([seq], dtype=torch.long, device=device)
                    logits = self.forward(tgt, mem.expand(1, -1, -1), memory_key_padding_mask=mem_mask)
                    log_probs = F.log_softmax(logits[0, -1], dim=-1)   # (V,)

                    topk_probs, topk_ids = log_probs.topk(beam_size)
                    for lp, tid in zip(topk_probs.tolist(), topk_ids.tolist()):
                        candidates.append((score + lp, seq + [tid]))

                # Keep top-k beams (length-penalised)
                candidates.sort(
                    key=lambda x: x[0] / ((len(x[1]) ** length_penalty) + 1e-6),
                    reverse=True,
                )
                beams = candidates[:beam_size]

                if all(s[-1] == eos_id for _, s in beams):
                    completed.extend(beams)
                    break

            completed.extend(beams)
            best_score = -float("inf")
            best_seq = [bos_id, eos_id]
            for score, seq in completed:
                norm = score / ((len(seq) ** length_penalty) + 1e-6)
                if norm > best_score:
                    best_score = norm
                    best_seq = seq

            # Pad to max_len
            best_seq = best_seq[:max_len]
            if len(best_seq) < max_len:
                best_seq += [PAD_ID] * (max_len - len(best_seq))
            results.append(best_seq)

        return torch.tensor(results, dtype=torch.long, device=device)

    def generate(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        max_len: int = MAX_TEXT_LEN,
        beam_size: int = BEAM_SIZE,
    ) -> torch.Tensor:
        """Unified generation entry point — uses beam search if beam_size > 1."""
        if beam_size > 1:
            return self.generate_beam(
                memory, memory_key_padding_mask, max_len, beam_size
            )
        return self.generate_greedy(memory, memory_key_padding_mask, max_len)
