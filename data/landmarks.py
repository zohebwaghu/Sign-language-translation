"""
MediaPipe landmark extraction and Gaussian heatmap generation.

Produces:
  - landmarks : (75, 2)  normalised [0,1] (x, y) coordinates
                          33 pose + 21 left-hand + 21 right-hand keypoints
  - heatmap   : (H, W)   float32 attention prior with Gaussian blobs
                          concentrated on hand and face regions
"""

import numpy as np
import cv2

try:
    import mediapipe as mp
    # mp.solutions was removed in MediaPipe 0.10 — check it actually exists
    _MP_AVAILABLE = hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic")
except ImportError:
    _MP_AVAILABLE = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import N_LANDMARKS, HEATMAP_SIGMA, IMG_SIZE

# ─── MediaPipe setup ──────────────────────────────────────────────────────────

def _build_holistic():
    if not _MP_AVAILABLE:
        return None
    mp_holistic = mp.solutions.holistic
    return mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

_holistic = None  # lazy-init — MediaPipe is heavy to load on import


def _get_holistic():
    global _holistic
    if _holistic is None:
        _holistic = _build_holistic()
    return _holistic


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_landmarks(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Extract 75 body keypoints from a single RGB frame.

    Args:
        frame_rgb: (H, W, 3) uint8 numpy array in RGB colour order.

    Returns:
        landmarks: (75, 2) float32 array with values in [0, 1].
                   Missing detections (hand not visible, etc.) are zero-filled.
    """
    landmarks = np.zeros((N_LANDMARKS, 2), dtype=np.float32)

    holistic = _get_holistic()
    if holistic is None:
        return landmarks  # mediapipe not installed → return zeros

    results = holistic.process(frame_rgb)

    # 33 pose landmarks (indices 0–32)
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[i] = [lm.x, lm.y]  # already normalised [0,1]

    # 21 left-hand landmarks (indices 33–53)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            landmarks[33 + i] = [lm.x, lm.y]

    # 21 right-hand landmarks (indices 54–74)
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            landmarks[54 + i] = [lm.x, lm.y]

    # Clamp to valid range
    landmarks = np.clip(landmarks, 0.0, 1.0)
    return landmarks


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

    # Define which landmark indices to render and their weights
    hand_wrist_tips = {33, 54}  # left/right wrist
    # Finger tips: indices 4,8,12,16,20 within each hand block
    left_tips  = {33 + t for t in [4, 8, 12, 16, 20]}
    right_tips = {54 + t for t in [4, 8, 12, 16, 20]}

    for idx in range(N_LANDMARKS):
        x_norm, y_norm = landmarks[idx]
        if x_norm == 0.0 and y_norm == 0.0:
            continue  # missing detection — skip

        cx = int(x_norm * (img_size - 1))
        cy = int(y_norm * (img_size - 1))

        # Weight selection
        if idx in hand_wrist_tips or idx in left_tips or idx in right_tips:
            weight = 2.0
        elif 33 <= idx < 75:  # rest of hand points
            weight = 1.5
        else:
            weight = 1.0  # pose / face

        _add_gaussian(heatmap, cx, cy, sigma, weight)

    # Normalise to [0, 1]
    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val

    return heatmap


# ─── Internal helpers ─────────────────────────────────────────────────────────

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
