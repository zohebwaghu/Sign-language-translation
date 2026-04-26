"""
HuggingFace How2Sign data loading utilities.

Uses streaming mode to avoid downloading the full dataset before training.
Caches preprocessed frames + landmarks to disk to avoid re-processing.
"""

# av must load its FFmpeg DLLs before torch on Windows to avoid a segfault
# when datasets streams Video-feature data.
import av  # noqa: F401

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

    Returns an iterable whose items have normalised keys:
      - "video"       : raw MP4 bytes (decoded by extract_frames via PyAV)
      - "translation" : English sentence string

    Video auto-decode is disabled (cast to decode=False) so we avoid the
    torchcodec/FFmpeg DLL dependency -- PyAV handles decoding.
    """
    from datasets import load_dataset, Video
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from training.config import DATASET_NAME, DATASET_SPLITS

    hf_split = DATASET_SPLITS.get(split, split)
    logger.info(f"Loading How2Sign split='{hf_split}' streaming={streaming}")

    dataset = load_dataset(DATASET_NAME, split=hf_split, streaming=streaming)
    # Keep video as raw bytes -- avoids torchcodec / FFmpeg system dependency.
    dataset = dataset.cast_column("mp4", Video(decode=False))

    # Wrap with a lazy normaliser so callers get {"video": bytes, "translation": str}.
    # Use a generator wrapper instead of dataset.map() to avoid a segfault in
    # the streaming+remove_columns code path of the datasets library.
    return _NormalisedDataset(dataset, streaming=streaming)


class _NormalisedDataset:
    """Thin wrapper that normalises How2Sign field names on-the-fly."""

    def __init__(self, raw_dataset, streaming: bool):
        self._ds = raw_dataset
        self._streaming = streaming

    # Support take() so tokenizer training (ds.take(N)) works unchanged.
    def take(self, n: int):
        return _TakeWrapper(self._ds.take(n))

    def __iter__(self):
        for ex in self._ds:
            yield _normalise_ex(ex)

    def __len__(self):
        return len(self._ds)


class _TakeWrapper:
    """Wraps an IterableDataset.take() result and normalises on iteration."""
    def __init__(self, raw):
        self._raw = raw
    def __iter__(self):
        for ex in self._raw:
            yield _normalise_ex(ex)


def _normalise_ex(ex: dict) -> dict:
    video_bytes = (ex.get("mp4") or {}).get("bytes") or b""
    sentence = (ex.get("json") or {}).get("SENTENCE", "")
    return {"video": video_bytes, "translation": sentence}


def extract_frames(
    video_bytes: bytes,
    max_frames: int = 64,
    img_size: int = 256,
    use_cache: bool = True,
) -> tuple:
    """
    Decode a video from raw bytes and extract uniformly-sampled frames.

    Caches result to DATA_CACHE_DIR/frames/<sha1>.pt on first call;
    subsequent calls for the same video load from disk (~1 ms vs ~500 ms).

    Args:
        video_bytes: Raw video bytes (MP4 / WebM / etc.).
        max_frames:  Maximum number of frames to return.
        img_size:    Target spatial resolution (square).
        use_cache:   Write/read .pt cache files (disable for one-off calls).

    Returns:
        frames: (T, 3, img_size, img_size) float32 tensor, ImageNet normalised.
        mask:   (max_frames,) bool tensor -- True for real frames, False for padding.
    """
    import hashlib
    import av
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from training.config import IMAGENET_MEAN, IMAGENET_STD, DATA_CACHE_DIR

    # Cache lookup
    if use_cache and video_bytes:
        cache_dir = DATA_CACHE_DIR / "frames"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(video_bytes).hexdigest()
        cache_path = cache_dir / f"{key}_{max_frames}_{img_size}.pt"
        if cache_path.exists():
            try:
                return torch.load(cache_path, weights_only=True)
            except Exception:
                cache_path.unlink(missing_ok=True)  # corrupt cache -- rebuild
    else:
        cache_path = None

    frames_raw = []
    try:
        container = av.open(io.BytesIO(video_bytes))
        stream = container.streams.video[0]

        # Prefer stream.frames; fall back to duration * fps estimate.
        # stream.frames is often 0 for streaming/remuxed MP4s.
        total_frames = stream.frames or 0
        if total_frames == 0 and stream.duration and stream.time_base:
            fps = float(stream.guessed_rate or stream.average_rate or 30)
            total_frames = int(float(stream.duration * stream.time_base) * fps)

        if total_frames > max_frames and total_frames > 0:
            indices = set(
                int(i * total_frames / max_frames) for i in range(max_frames)
            )
        else:
            indices = None  # video shorter than max_frames -- keep all

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
        logger.warning(f"Frame extraction failed: {e} -- returning zeros")

    T = len(frames_raw)

    # Vectorised build (avoid per-frame Python loop)
    frames_tensor = torch.zeros(max_frames, 3, img_size, img_size, dtype=torch.float32)
    mask = torch.zeros(max_frames, dtype=torch.bool)

    if T > 0:
        import numpy as np
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        arr = np.stack(frames_raw)                               # (T, H, W, 3)
        t   = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)  # (T, 3, H, W)
        frames_tensor[:T] = (t - mean) / std
        mask[:T] = True

    result = (frames_tensor, mask)

    if cache_path is not None:
        try:
            torch.save(result, cache_path)
        except Exception as e:
            logger.warning(f"Frame cache write failed: {e}")

    return result


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize (H, W, 3) uint8 array to (size, size, 3) using INTER_LINEAR."""
    import cv2
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
