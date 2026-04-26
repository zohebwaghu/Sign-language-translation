"""
MediaPipe landmark extraction and Gaussian heatmap generation.

Produces:
  - landmarks : (75, 2)  normalised [0,1] (x, y) coordinates
                          33 pose + 21 left-hand + 21 right-hand keypoints
  - heatmap   : (H, W)   float32 attention prior with Gaussian blobs
                          concentrated on hand and face regions
"""

import urllib.request
import numpy as np
import cv2
from pathlib import Path

try:
    import mediapipe as mp
    # mp.solutions was removed in MediaPipe 0.10 — check it actually exists
    _MP_AVAILABLE = hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic")
except ImportError:
    _MP_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import N_LANDMARKS, HEATMAP_SIGMA, IMG_SIZE

# ??? Model download ???????????????????????????????????????????????????????????

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
_MODEL_CACHE = Path(__file__).parent / "cache" / "holistic_landmarker.task"


def _ensure_model() -> str:
    if not _MODEL_CACHE.exists():
        _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe holistic model -> {_MODEL_CACHE} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_CACHE)
    return str(_MODEL_CACHE)


# ??? MediaPipe setup ??????????????????????????????????????????????????????????

def _build_holistic():
    if not _MP_AVAILABLE:
        return None
    try:
        model_path = _ensure_model()
        options = mp.tasks.vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        return mp.tasks.vision.HolisticLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Warning: MediaPipe HolisticLandmarker init failed: {e}. Using zero landmarks.")
        return None


_UNINITIALIZED = object()
_holistic = _UNINITIALIZED  # lazy-init -- MediaPipe is heavy to load on import


def _get_holistic():
    global _holistic
    if _holistic is _UNINITIALIZED:
        _holistic = _build_holistic()  # may return None on failure; that's fine
    return _holistic


# ??? Public API ???????????????????????????????????????????????????????????????

def extract_landmarks(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Extract 75 body keypoints from a single RGB frame.

    Args:
        frame_rgb: (H, W, 3) uint8 numpy array in RGB colour order.

    Returns:
        landmarks: (75, 2) float32 array with values in [0, 1].
                   Missing detections are zero-filled.
    """
    landmarks = np.zeros((N_LANDMARKS, 2), dtype=np.float32)

    holistic = _get_holistic()
    if holistic is None:
        return landmarks

    try:
        # Ensure uint8 contiguous array for MediaPipe
        frame = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = holistic.detect(mp_image)
    except Exception:
        return landmarks

    # pose_landmarks: List[NormalizedLandmark] (33 items) or empty
    if results.pose_landmarks:
        lms = results.pose_landmarks
        # Handle both flat list and nested list (API versions differ)
        if isinstance(lms[0], list):
            lms = lms[0]
        for i, lm in enumerate(lms[:33]):
            landmarks[i] = [lm.x, lm.y]

    # left_hand_landmarks: List[NormalizedLandmark] (21 items) or empty
    if results.left_hand_landmarks:
        lms = results.left_hand_landmarks
        if isinstance(lms[0], list):
            lms = lms[0]
        for i, lm in enumerate(lms[:21]):
            landmarks[33 + i] = [lm.x, lm.y]

    # right_hand_landmarks: List[NormalizedLandmark] (21 items) or empty
    if results.right_hand_landmarks:
        lms = results.right_hand_landmarks
        if isinstance(lms[0], list):
            lms = lms[0]
        for i, lm in enumerate(lms[:21]):
            landmarks[54 + i] = [lm.x, lm.y]

    return np.clip(landmarks, 0.0, 1.0)


def create_landmark_heatmap(
    landmarks: np.ndarray,
    img_size: int = IMG_SIZE,
    sigma: float = HEATMAP_SIGMA,
) -> np.ndarray:
    """
    Render a 2-D Gaussian attention prior from landmark coordinates.

    Only hand and face-adjacent landmarks are included (indices 0-32 pose
    plus both hands), weighted by spatial importance:
      - Wrist / finger tips: weight 2.0
      - Other hand points:   weight 1.5
      - Face / pose:         weight 1.0

    Args:
        landmarks: (75, 2) float32 normalised coordinates.
        img_size:  Output spatial resolution (square).
        sigma:     Gaussian standard deviation in pixels.

    Returns:
        heatmap: (img_size, img_size) float32, values in [0, 1].
    """
    heatmap = np.zeros((img_size, img_size), dtype=np.float32)

    hand_wrist_tips = {33, 54}
    left_tips  = {33 + t for t in [4, 8, 12, 16, 20]}
    right_tips = {54 + t for t in [4, 8, 12, 16, 20]}

    for idx in range(N_LANDMARKS):
        x_norm, y_norm = landmarks[idx]
        if x_norm == 0.0 and y_norm == 0.0:
            continue

        cx = int(x_norm * (img_size - 1))
        cy = int(y_norm * (img_size - 1))

        if idx in hand_wrist_tips or idx in left_tips or idx in right_tips:
            weight = 2.0
        elif 33 <= idx < 75:
            weight = 1.5
        else:
            weight = 1.0

        _add_gaussian(heatmap, cx, cy, sigma, weight)

    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val

    return heatmap


# ??? Internal helpers ?????????????????????????????????????????????????????????

def _add_gaussian(
    heatmap: np.ndarray,
    cx: int,
    cy: int,
    sigma: float,
    weight: float = 1.0,
) -> None:
    """Add a 2-D Gaussian blob centred at (cx, cy) to heatmap in-place."""
    H, W = heatmap.shape
    radius = int(3 * sigma)

    x0 = max(0, cx - radius)
    x1 = min(W, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(H, cy + radius + 1)

    xs = np.arange(x0, x1) - cx
    ys = np.arange(y0, y1) - cy
    xx, yy = np.meshgrid(xs, ys)
    blob = weight * np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], blob.astype(np.float32))
