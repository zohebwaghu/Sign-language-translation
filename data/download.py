"""
HuggingFace How2Sign data loading utilities.

Uses streaming mode to avoid downloading the full dataset before training.
Caches preprocessed frames + landmarks to disk to avoid re-processing.
"""

import io
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_dataset_split(split: str = "train", streaming: bool = True):
    """
    Load a How2Sign split from HuggingFace.

    Args:
        split:     One of 'train', 'val', 'test'.
        streaming: If True, use streaming mode (no full download required).

    Returns:
        HuggingFace Dataset / IterableDataset object.
    """
    from datasets import load_dataset
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from training.config import DATASET_NAME, DATASET_SPLITS

    hf_split = DATASET_SPLITS.get(split, split)
    logger.info(f"Loading How2Sign split='{hf_split}' streaming={streaming}")

    dataset = load_dataset(
        DATASET_NAME,
        split=hf_split,
        streaming=streaming,
    )

    # Disable automatic video decoding on all Video columns — we decode with PyAV
    # ourselves in extract_frames(). Without this, datasets>=4.x requires torchcodec.
    try:
        from datasets import Video as HFVideo
        _no_decode = HFVideo(decode=False)
        for col in list(dataset.features.keys()):
            if hasattr(dataset.features[col], "decode"):
                dataset = dataset.cast_column(col, _no_decode)
    except Exception:
        pass

    return dataset


def extract_frames(
    video_bytes: bytes,
    max_frames: int = 64,
    img_size: int = 256,
) -> tuple:
    """
    Decode a video from raw bytes and extract uniformly-sampled frames.

    Args:
        video_bytes: Raw video bytes (MP4 / WebM / etc.).
        max_frames:  Maximum number of frames to return.
        img_size:    Target spatial resolution (square).

    Returns:
        frames: (T, 3, img_size, img_size) float32 tensor, ImageNet normalised.
        mask:   (max_frames,) bool tensor — True for real frames, False for padding.
    """
    import av
    import torchvision.transforms.functional as TF
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from training.config import IMAGENET_MEAN, IMAGENET_STD

    frames_raw = []
    try:
        container = av.open(io.BytesIO(video_bytes))
        stream = container.streams.video[0]
        total_frames = stream.frames or 0

        # Compute uniform sample indices
        if total_frames > max_frames and total_frames > 0:
            indices = set(
                int(i * total_frames / max_frames) for i in range(max_frames)
            )
        else:
            indices = None  # take every frame

        for i, frame in enumerate(container.decode(video=0)):
            if indices is not None and i not in indices:
                continue
            img = frame.to_ndarray(format="rgb24")
            img = _resize(img, img_size)
            frames_raw.append(img)
            if len(frames_raw) >= max_frames:
                break
        container.close()
    except Exception as e:
        logger.warning(f"Frame extraction failed: {e} — returning zeros")

    T = len(frames_raw)

    # Build padded tensor
    frames_tensor = torch.zeros(max_frames, 3, img_size, img_size, dtype=torch.float32)
    mask = torch.zeros(max_frames, dtype=torch.bool)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    for i, img in enumerate(frames_raw):
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        t = (t - mean) / std
        frames_tensor[i] = t
        mask[i] = True

    return frames_tensor, mask


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize (H, W, 3) uint8 array to (size, size, 3) using INTER_LINEAR."""
    import cv2
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
