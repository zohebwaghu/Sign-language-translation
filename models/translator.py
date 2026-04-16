"""
F4 — SignLanguageTranslator: full end-to-end model.

Combines:
  F1  VisualEncoder     (ResNet-18 + SpatialAttention)
  F2  TemporalEncoder   (4-layer Transformer Encoder)
  F3  TextDecoder       (4-layer Transformer Decoder + beam search)
"""

import torch
import torch.nn as nn
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import D_MODEL, MAX_TEXT_LEN, BEAM_SIZE, PAD_ID, BOS_ID, EOS_ID
from models.visual_encoder import VisualEncoder
from models.temporal_encoder import TemporalEncoder
from models.text_decoder import TextDecoder


class SignLanguageTranslator(nn.Module):
    """
    Full ASL-to-English translation model.

    Forward pass (training):
        frames  → VisualEncoder → (B, T, d_model)
        heatmaps ─────────────↗
        ↓
        TemporalEncoder        → (B, T, d_model) contextualised
        ↓
        TextDecoder (teacher-forcing) → (B, S, vocab_size) logits

    Inference:
        translate(frames, heatmaps) → List[str]
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.visual_encoder   = VisualEncoder(d_model=d_model, freeze_backbone=freeze_backbone)
        self.temporal_encoder = TemporalEncoder(d_model=d_model)
        self.text_decoder     = TextDecoder(vocab_size=vocab_size, d_model=d_model)
        self.vocab_size       = vocab_size

    # ── Phase transitions ────────────────────────────────────────────────

    def set_phase(self, phase: int) -> None:
        """
        Switch between training phases.
          phase=1 → backbone frozen   (fast, stable early training)
          phase=2 → backbone unfrozen (fine-tune for higher BLEU)
        """
        if phase == 1:
            self.visual_encoder.freeze_backbone()
        elif phase == 2:
            self.visual_encoder.unfreeze_backbone()
        else:
            raise ValueError(f"phase must be 1 or 2, got {phase}")

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        frames: torch.Tensor,
        tgt: torch.Tensor,
        heatmaps: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.

        Args:
            frames:    (B, T, 3, H, W)
            tgt:       (B, S)  target token IDs (full sequence incl. BOS/EOS)
            heatmaps:  (B, T, H, W) or None
            frame_mask:(B, T)  bool, True = real frame — converted to padding mask
            tgt_key_padding_mask: (B, S) bool, True = PAD token

        Returns:
            logits: (B, S, vocab_size)
        """
        # Convert frame_mask (True=real) to padding mask (True=ignore)
        enc_padding_mask = ~frame_mask if frame_mask is not None else None

        visual_feats  = self.visual_encoder(frames, heatmaps)          # (B, T, d_model)
        memory        = self.temporal_encoder(visual_feats, enc_padding_mask)  # (B, T, d_model)
        logits        = self.text_decoder(
            tgt, memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=enc_padding_mask,
        )
        return logits                                                   # (B, S, vocab_size)

    # ── Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def translate(
        self,
        frames: torch.Tensor,
        tokenizer,
        heatmaps: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
        max_len: int = MAX_TEXT_LEN,
        beam_size: int = BEAM_SIZE,
    ):
        """
        Translate a batch of video clips to English strings.

        Args:
            frames:    (B, T, 3, H, W)
            tokenizer: SignTokenizer instance (loaded)
            heatmaps:  (B, T, H, W) or None
            frame_mask:(B, T) bool or None
            beam_size: 1=greedy, >1=beam search

        Returns:
            List[str] of length B
        """
        self.eval()
        enc_padding_mask = ~frame_mask if frame_mask is not None else None

        visual_feats = self.visual_encoder(frames, heatmaps)
        memory       = self.temporal_encoder(visual_feats, enc_padding_mask)
        token_ids    = self.text_decoder.generate(
            memory, enc_padding_mask, max_len=max_len, beam_size=beam_size
        )                                        # (B, max_len)

        return [
            tokenizer.decode(ids.tolist(), skip_special=True)
            for ids in token_ids
        ]

    def count_parameters(self) -> dict:
        """Return parameter counts per sub-module."""
        def n_params(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "visual_encoder":   n_params(self.visual_encoder),
            "temporal_encoder": n_params(self.temporal_encoder),
            "text_decoder":     n_params(self.text_decoder),
            "total":            n_params(self),
        }
