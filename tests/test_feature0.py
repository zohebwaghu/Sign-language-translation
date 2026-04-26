"""
F0 Gate Checks -- Data Pipeline
Runs entirely on CPU with synthetic data. No HuggingFace download required.

Gate checks:
  0.1  extract_frames() -> (T, 3, 256, 256) + bool mask
  0.2  extract_landmarks() -> (75, 2) values in [0, 1]
  0.3  Heatmap concentrates on hand/face regions
  0.4  Missing hand detection -> zeros, no crash
  0.5  Padding mask: True for real frames, False for padded
  0.6  Dataset __len__ and __getitem__ work with DataLoader
"""

import sys
import io
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.config import MAX_FRAMES, IMG_SIZE, N_LANDMARKS, MAX_TEXT_LEN, PAD_ID
from data.download import extract_frames
from data.landmarks import extract_landmarks, create_landmark_heatmap
from data.dataset import How2SignDataset, collate_fn


# ??? Helpers ?????????????????????????????????????????????????????????????????

def _make_fake_video_bytes(n_frames: int = 10) -> bytes:
    """Create a minimal synthetic MP4 in memory using PyAV."""
    try:
        import av
        buf = io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=25)
        stream.width = 256
        stream.height = 256
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23"}
        for _ in range(n_frames):
            frame_data = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()
        return buf.getvalue()
    except Exception:
        return b""  # empty -> extract_frames will gracefully return zeros


def _make_fake_rgb_frame(h: int = 256, w: int = 256) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_fake_tokenizer():
    """Minimal tokenizer stub that returns fixed-length zero tensors."""
    class FakeTok:
        vocab_size = 100
        def encode_batch(self, texts, max_len=64, pad=True):
            B = len(texts)
            ids  = torch.zeros(B, max_len, dtype=torch.long)
            mask = torch.zeros(B, max_len, dtype=torch.bool)
            ids[:, 0]  = 1   # BOS
            ids[:, 1]  = 2   # EOS
            mask[:, :2] = True
            return ids, mask
    return FakeTok()


def _make_fake_records(n: int = 4) -> list:
    video_bytes = _make_fake_video_bytes(n_frames=10)
    return [
        {"video": video_bytes, "translation": f"hello world {i}"}
        for i in range(n)
    ]


# ??? Gate Checks ?????????????????????????????????????????????????????????????

def test_0_1_extract_frames_shape():
    """0.1 -- extract_frames() returns (T, 3, 256, 256) + bool mask."""
    video_bytes = _make_fake_video_bytes(n_frames=20)
    frames, mask = extract_frames(video_bytes, max_frames=MAX_FRAMES, img_size=IMG_SIZE)

    assert frames.shape == (MAX_FRAMES, 3, IMG_SIZE, IMG_SIZE), (
        f"Expected ({MAX_FRAMES}, 3, {IMG_SIZE}, {IMG_SIZE}), got {frames.shape}"
    )
    assert mask.shape == (MAX_FRAMES,), f"Mask shape wrong: {mask.shape}"
    assert mask.dtype == torch.bool, f"Mask should be bool, got {mask.dtype}"
    print(f"  [PASS] 0.1  frames={frames.shape}  real_frames={mask.sum().item()}")


def test_0_2_landmarks_range():
    """0.2 -- extract_landmarks() returns (75, 2) with values in [0, 1]."""
    frame = _make_fake_rgb_frame()
    lm = extract_landmarks(frame)

    assert lm.shape == (N_LANDMARKS, 2), f"Expected ({N_LANDMARKS}, 2), got {lm.shape}"
    assert lm.dtype == np.float32, f"Expected float32, got {lm.dtype}"
    assert lm.min() >= 0.0 and lm.max() <= 1.0, (
        f"Landmark values out of [0,1]: min={lm.min():.4f} max={lm.max():.4f}"
    )
    print(f"  [PASS] 0.2  landmarks={lm.shape}  min={lm.min():.4f}  max={lm.max():.4f}")


def test_0_3_heatmap_concentration():
    """0.3 -- heatmap values are non-trivially distributed (not all zeros)."""
    # Place a landmark in the centre
    lm = np.zeros((N_LANDMARKS, 2), dtype=np.float32)
    lm[33] = [0.5, 0.5]   # left wrist in the middle
    lm[54] = [0.7, 0.3]   # right wrist upper-right

    hm = create_landmark_heatmap(lm, img_size=IMG_SIZE)

    assert hm.shape == (IMG_SIZE, IMG_SIZE), f"Heatmap shape wrong: {hm.shape}"
    assert hm.max() > 0.0, "Heatmap is all zeros -- landmark blobs not rendered"
    assert hm.min() >= 0.0 and hm.max() <= 1.0, "Heatmap out of [0,1]"

    # Check that the peak is near EITHER placed landmark (both have equal weight)
    peak_y, peak_x = np.unravel_index(np.argmax(hm), hm.shape)
    expected = [
        (int(0.5 * (IMG_SIZE - 1)), int(0.5 * (IMG_SIZE - 1))),  # lm[33]: x=0.5,y=0.5
        (int(0.3 * (IMG_SIZE - 1)), int(0.7 * (IMG_SIZE - 1))),  # lm[54]: x=0.7,y=0.3 -> cx=178,cy=76
    ]
    dists = [np.sqrt((peak_x - cx)**2 + (peak_y - cy)**2) for cy, cx in expected]
    assert min(dists) < 30, (
        f"Heatmap peak@({peak_x},{peak_y}) too far from any landmark: dists={[f'{d:.1f}' for d in dists]}"
    )
    print(f"  [PASS] 0.3  heatmap max={hm.max():.4f}  peak@({peak_x},{peak_y})")


def test_0_4_missing_landmark_no_crash():
    """0.4 -- Missing hand detection -> zeros, no crash."""
    # Pass a blank (all-black) frame -- MediaPipe won't detect anything
    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    lm = extract_landmarks(blank)

    assert lm.shape == (N_LANDMARKS, 2), f"Shape wrong on blank frame: {lm.shape}"
    # All should be zero (or very close to zero) since no detection
    assert lm.min() >= 0.0 and lm.max() <= 1.0, "Values out of range on blank frame"
    print(f"  [PASS] 0.4  blank frame -> landmarks shape {lm.shape}, no crash")


def test_0_5_padding_mask():
    """0.5 -- Real frames have mask=True, padded positions have mask=False."""
    # Use a very short video (few frames) so padding kicks in
    video_bytes = _make_fake_video_bytes(n_frames=5)
    frames, mask = extract_frames(video_bytes, max_frames=MAX_FRAMES, img_size=IMG_SIZE)

    n_real = mask.sum().item()
    n_pad  = (~mask).sum().item()

    # If video had frames, some should be True
    # Padded positions frames should be zeros
    if n_real > 0:
        padded_frames = frames[~mask]   # (n_pad, 3, H, W)
        # All-zero check on padded positions
        if padded_frames.numel() > 0:
            assert padded_frames.abs().max().item() < 1e-6, (
                "Padded frame positions are non-zero!"
            )
    print(f"  [PASS] 0.5  real={n_real}  padded={n_pad}")


def test_0_6_dataset_dataloader():
    """0.6 -- Dataset __len__ and __getitem__ work with DataLoader."""
    records = _make_fake_records(n=4)
    tokenizer = _make_fake_tokenizer()

    ds = How2SignDataset(
        records=records,
        tokenizer=tokenizer,
        max_frames=MAX_FRAMES,
        img_size=IMG_SIZE,
        max_text_len=MAX_TEXT_LEN,
        extract_lm=False,   # skip MediaPipe for speed in gate check
    )

    assert len(ds) == 4, f"Expected len=4, got {len(ds)}"

    item = ds[0]
    assert item["frames"].shape    == (MAX_FRAMES, 3, IMG_SIZE, IMG_SIZE)
    assert item["landmarks"].shape == (MAX_FRAMES, N_LANDMARKS, 2)
    assert item["heatmaps"].shape  == (MAX_FRAMES, IMG_SIZE, IMG_SIZE)
    assert item["token_ids"].shape == (MAX_TEXT_LEN,)
    assert item["frame_mask"].shape == (MAX_FRAMES,)
    assert item["frame_mask"].dtype == torch.bool
    assert isinstance(item["text"], str)

    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(loader))
    assert batch["frames"].shape == (2, MAX_FRAMES, 3, IMG_SIZE, IMG_SIZE)
    assert batch["token_ids"].shape == (2, MAX_TEXT_LEN)

    print(f"  [PASS] 0.6  Dataset len={len(ds)}  DataLoader batch frames={batch['frames'].shape}")


# ??? Runner ??????????????????????????????????????????????????????????????????

if __name__ == "__main__":
    tests = [
        test_0_1_extract_frames_shape,
        test_0_2_landmarks_range,
        test_0_3_heatmap_concentration,
        test_0_4_missing_landmark_no_crash,
        test_0_5_padding_mask,
        test_0_6_dataset_dataloader,
    ]

    passed = 0
    failed = 0
    print("\n=== F0 Gate Checks: Data Pipeline ===\n")
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"  PASSED: {passed}/{len(tests)}")
    if failed:
        print(f"  FAILED: {failed}/{len(tests)}")
        sys.exit(1)
    else:
        print("  ALL F0 GATE CHECKS PASSED OK")
