"""
F1 Gate Checks -- Visual Encoder
All tests run on CPU with dummy tensors.

Gate checks:
  1.1  Input (B, T, 3, 256, 256) -> output (B, T, 512)
  1.2  Spatial attention weights are in [0, 1]
  1.3  Works with landmark_heatmaps=None (no crash, uses full features)
  1.4  ResNet backbone frozen (.requires_grad = False)
  1.5  Attention output differs from no-attention output (attention is doing something)
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.config import D_MODEL, MAX_FRAMES, IMG_SIZE
from models.visual_encoder import VisualEncoder, SpatialAttention


B, T = 2, 4   # small batch for CPU speed


def _dummy_frames() -> torch.Tensor:
    return torch.randn(B, T, 3, IMG_SIZE, IMG_SIZE)


def _dummy_heatmaps() -> torch.Tensor:
    h = torch.rand(B, T, IMG_SIZE, IMG_SIZE)
    return h


def test_1_1_output_shape():
    """1.1 -- Input (B, T, 3, 256, 256) -> output (B, T, D_MODEL)."""
    model = VisualEncoder(d_model=D_MODEL, freeze_backbone=True)
    model.eval()

    with torch.no_grad():
        out = model(_dummy_frames(), _dummy_heatmaps())

    assert out.shape == (B, T, D_MODEL), (
        f"Expected ({B}, {T}, {D_MODEL}), got {out.shape}"
    )
    print(f"  [PASS] 1.1  output={out.shape}")


def test_1_2_attention_weights_range():
    """1.2 -- SpatialAttention weights are in [0, 1]."""
    attn_module = SpatialAttention(in_channels=512)
    attn_module.eval()

    feat = torch.randn(B * T, 512, 8, 8)
    hm   = torch.rand(B * T, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        # Manually extract attention weights
        import torch.nn.functional as F
        hm_resized = hm.unsqueeze(1)
        hm_resized = F.interpolate(hm_resized, size=(8, 8), mode="bilinear", align_corners=False)
        x = torch.cat([feat, hm_resized], dim=1)
        x = F.relu(attn_module.bn1(attn_module.conv1(x)))
        weights = torch.sigmoid(attn_module.conv2(x))  # (N, 1, 8, 8)

    assert weights.min().item() >= 0.0, f"Attention weight below 0: {weights.min()}"
    assert weights.max().item() <= 1.0, f"Attention weight above 1: {weights.max()}"
    print(f"  [PASS] 1.2  attention weights min={weights.min():.4f}  max={weights.max():.4f}")


def test_1_3_no_heatmap_no_crash():
    """1.3 -- Passing heatmaps=None works without crash."""
    model = VisualEncoder(d_model=D_MODEL, freeze_backbone=True)
    model.eval()

    with torch.no_grad():
        out = model(_dummy_frames(), heatmaps=None)

    assert out.shape == (B, T, D_MODEL), f"Shape wrong: {out.shape}"
    print(f"  [PASS] 1.3  no heatmap -> output={out.shape}")


def test_1_4_backbone_frozen():
    """1.4 -- ResNet backbone is frozen (no requires_grad)."""
    model = VisualEncoder(d_model=D_MODEL, freeze_backbone=True)

    for name, param in model.backbone.named_parameters():
        assert not param.requires_grad, (
            f"Backbone param '{name}' should be frozen but requires_grad=True"
        )

    # Sanity: unfreeze and check
    model.unfreeze_backbone()
    for name, param in model.backbone.named_parameters():
        assert param.requires_grad, f"After unfreeze, '{name}' should have requires_grad=True"

    print("  [PASS] 1.4  backbone frozen (and unfreeze works)")


def test_1_5_attention_changes_output():
    """1.5 -- Output with heatmap differs from output without (attention is active)."""
    model = VisualEncoder(d_model=D_MODEL, freeze_backbone=True)
    model.eval()

    frames = _dummy_frames()

    with torch.no_grad():
        out_no_hm = model(frames, heatmaps=None)
        out_hm    = model(frames, heatmaps=_dummy_heatmaps())

    diff = (out_hm - out_no_hm).abs().mean().item()
    assert diff > 1e-6, (
        f"Attention had no effect -- outputs identical (diff={diff:.2e})"
    )
    print(f"  [PASS] 1.5  attention changes output (mean diff={diff:.4f})")


if __name__ == "__main__":
    tests = [
        test_1_1_output_shape,
        test_1_2_attention_weights_range,
        test_1_3_no_heatmap_no_crash,
        test_1_4_backbone_frozen,
        test_1_5_attention_changes_output,
    ]

    passed = failed = 0
    print("\n=== F1 Gate Checks: Visual Encoder ===\n")
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
        print("  ALL F1 GATE CHECKS PASSED OK")
