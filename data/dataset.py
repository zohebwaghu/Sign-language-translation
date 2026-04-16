"""
PyTorch Dataset for How2Sign sign language translation.

Each item returns a dict with:
  frames     : (MAX_FRAMES, 3, IMG_SIZE, IMG_SIZE)  float32  ImageNet-normalised
  landmarks  : (MAX_FRAMES, 75, 2)                  float32  normalised [0,1]
  heatmaps   : (MAX_FRAMES, IMG_SIZE, IMG_SIZE)      float32  Gaussian attention maps
  token_ids  : (MAX_TEXT_LEN,)                       int64    padded BPE token IDs
  frame_mask : (MAX_FRAMES,)                         bool     True = real frame
  text_mask  : (MAX_TEXT_LEN,)                       bool     True = real token
  text       : str                                   raw translation (for eval)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import (
    MAX_FRAMES, IMG_SIZE, MAX_TEXT_LEN, N_LANDMARKS,
    PAD_ID, BOS_ID, EOS_ID, TOKENIZER_MODEL,
)
from data.download import extract_frames
from data.landmarks import extract_landmarks, create_landmark_heatmap

logger = logging.getLogger(__name__)


class How2SignDataset(Dataset):
    """
    Map-style PyTorch Dataset wrapping How2Sign.

    For streaming use, convert the HuggingFace IterableDataset to a list
    first (or use How2SignStreamDataset below for on-the-fly loading).
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer,
        max_frames: int = MAX_FRAMES,
        img_size: int = IMG_SIZE,
        max_text_len: int = MAX_TEXT_LEN,
        extract_lm: bool = True,
    ):
        """
        Args:
            records:      List of dicts each containing 'video' bytes field
                          and 'translation' / 'sentence' text field.
            tokenizer:    Loaded SignTokenizer instance.
            max_frames:   Temporal clip length (padding/truncation target).
            img_size:     Spatial resolution.
            max_text_len: Maximum decoder sequence length.
            extract_lm:   Whether to run MediaPipe (can disable for speed tests).
        """
        self.records = records
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.img_size = img_size
        self.max_text_len = max_text_len
        self.extract_lm = extract_lm

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]

        # ── Extract text ──────────────────────────────────────────────────
        text: str = record.get("translation", record.get("sentence", ""))

        # ── Extract frames ────────────────────────────────────────────────
        video_bytes: bytes = record.get("video", b"")
        if isinstance(video_bytes, dict):
            # HuggingFace sometimes wraps as {"bytes": ..., "path": ...}
            video_bytes = video_bytes.get("bytes", b"")

        frames, frame_mask = extract_frames(
            video_bytes,
            max_frames=self.max_frames,
            img_size=self.img_size,
        )
        # frames: (MAX_FRAMES, 3, H, W)

        # ── Extract landmarks + heatmaps ──────────────────────────────────
        landmarks = torch.zeros(self.max_frames, N_LANDMARKS, 2, dtype=torch.float32)
        heatmaps  = torch.zeros(self.max_frames, self.img_size, self.img_size, dtype=torch.float32)

        if self.extract_lm and frame_mask.any():
            # Convert normalised tensor back to uint8 for MediaPipe
            import torchvision.transforms.functional as TF
            from training.config import IMAGENET_MEAN, IMAGENET_STD

            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

            for i in range(self.max_frames):
                if not frame_mask[i]:
                    break
                # Denormalize and convert to uint8 numpy
                frame_t = frames[i] * std + mean        # (3, H, W)
                frame_np = (frame_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

                lm = extract_landmarks(frame_np)        # (75, 2)
                hm = create_landmark_heatmap(lm, self.img_size)  # (H, W)

                landmarks[i] = torch.from_numpy(lm)
                heatmaps[i]  = torch.from_numpy(hm)

        # ── Tokenize text ─────────────────────────────────────────────────
        token_ids, text_mask = self.tokenizer.encode_batch(
            [text], max_len=self.max_text_len, pad=True
        )
        token_ids = token_ids[0]   # (MAX_TEXT_LEN,)
        text_mask = text_mask[0]   # (MAX_TEXT_LEN,)

        return {
            "frames":     frames,       # (T, 3, H, W)
            "landmarks":  landmarks,    # (T, 75, 2)
            "heatmaps":   heatmaps,     # (T, H, W)
            "token_ids":  token_ids,    # (MAX_TEXT_LEN,)
            "frame_mask": frame_mask,   # (T,)  bool
            "text_mask":  text_mask,    # (MAX_TEXT_LEN,)  bool
            "text":       text,         # raw string (eval)
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Default DataLoader collate — stacks tensors, collects raw text strings.
    All tensors are already padded to fixed lengths so no dynamic padding needed.
    """
    keys = [k for k in batch[0] if k != "text"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["text"] = [b["text"] for b in batch]
    return out
