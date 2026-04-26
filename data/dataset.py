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
import random
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import (
    MAX_FRAMES, IMG_SIZE, MAX_TEXT_LEN, N_LANDMARKS,
    PAD_ID, BOS_ID, EOS_ID, TOKENIZER_MODEL,
)
from data.download import extract_frames
from data.landmarks import extract_landmarks, create_landmark_heatmap

logger = logging.getLogger(__name__)


def _process_record(
    record: Dict[str, Any],
    tokenizer,
    max_frames: int,
    img_size: int,
    max_text_len: int,
    extract_lm: bool,
) -> Dict[str, Any]:
    """Shared processing logic for both map-style and streaming datasets."""
    text: str = record.get("translation", record.get("sentence", ""))

    video_bytes: bytes = record.get("video", b"")
    if isinstance(video_bytes, dict):
        video_bytes = video_bytes.get("bytes", b"")

    frames, frame_mask = extract_frames(
        video_bytes,
        max_frames=max_frames,
        img_size=img_size,
    )

    landmarks = torch.zeros(max_frames, N_LANDMARKS, 2, dtype=torch.float32)
    heatmaps  = torch.zeros(max_frames, img_size, img_size, dtype=torch.float32)

    if extract_lm and frame_mask.any():
        from training.config import IMAGENET_MEAN, IMAGENET_STD
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

        for i in range(max_frames):
            if not frame_mask[i]:
                break
            frame_t  = frames[i] * std + mean
            frame_np = (frame_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            lm = extract_landmarks(frame_np)
            hm = create_landmark_heatmap(lm, img_size)
            landmarks[i] = torch.from_numpy(lm)
            heatmaps[i]  = torch.from_numpy(hm)

    token_ids, text_mask = tokenizer.encode_batch(
        [text], max_len=max_text_len, pad=True
    )
    token_ids = token_ids[0]
    text_mask = text_mask[0]

    return {
        "frames":     frames,
        "landmarks":  landmarks,
        "heatmaps":   heatmaps,
        "token_ids":  token_ids,
        "frame_mask": frame_mask,
        "text_mask":  text_mask,
        "text":       text,
    }


class How2SignDataset(Dataset):
    """
    Map-style dataset -- requires all records materialised in memory.
    Use for validation (small) or overfit tests.
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
        self.records = records
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.img_size = img_size
        self.max_text_len = max_text_len
        self.extract_lm = extract_lm

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return _process_record(
            self.records[idx],
            self.tokenizer,
            self.max_frames,
            self.img_size,
            self.max_text_len,
            self.extract_lm,
        )


class How2SignStreamingDataset(IterableDataset):
    """
    IterableDataset for full-scale training on How2Sign.

    Streams records lazily from a HuggingFace streaming dataset so the full
    ~75 GB of video bytes never needs to live in RAM simultaneously.
    Uses a fixed-size shuffle buffer for per-epoch randomisation.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        max_frames: int = MAX_FRAMES,
        img_size: int = IMG_SIZE,
        max_text_len: int = MAX_TEXT_LEN,
        extract_lm: bool = True,
        shuffle_buffer: int = 500,
    ):
        self.hf_dataset   = hf_dataset
        self.tokenizer    = tokenizer
        self.max_frames   = max_frames
        self.img_size     = img_size
        self.max_text_len = max_text_len
        self.extract_lm   = extract_lm
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        buffer: List[Dict[str, Any]] = []

        for record in self.hf_dataset:
            buffer.append(record)
            if len(buffer) >= self.shuffle_buffer:
                random.shuffle(buffer)
                yield from self._flush(buffer)
                buffer = []

        # Flush remaining partial buffer at end of epoch.
        if buffer:
            random.shuffle(buffer)
            yield from self._flush(buffer)

    def _flush(self, records: List[Dict[str, Any]]):
        for rec in records:
            try:
                yield _process_record(
                    rec,
                    self.tokenizer,
                    self.max_frames,
                    self.img_size,
                    self.max_text_len,
                    self.extract_lm,
                )
            except Exception as e:
                logger.warning(f"Skipping bad record: {e}")


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Default DataLoader collate -- stacks tensors, collects raw text strings.
    All tensors are already padded to fixed lengths so no dynamic padding needed.
    """
    keys = [k for k in batch[0] if k != "text"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["text"] = [b["text"] for b in batch]
    return out
