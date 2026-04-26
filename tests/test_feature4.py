"""
F4 Gate Checks -- End-to-End Model + Training
All tests run on CPU with tiny dummy data (no GPU, no real dataset).

Gate checks:
  4.1  Full forward pass: (B, T, 3, 256, 256) -> (B, S, V) logits
  4.2  Loss decreases over a few steps (overfit on tiny batch)
  4.3  No NaN/Inf in loss or gradients
  4.4  translate() produces a non-empty string
  4.5  Checkpoint saves and loads correctly
  4.6  Phase 2 unfreeze propagates requires_grad to backbone
  4.7  Gradient accumulation: effective batch tracked correctly
"""

import sys, os, tempfile
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.config import D_MODEL, PAD_ID, BOS_ID, EOS_ID, LABEL_SMOOTHING, MAX_TEXT_LEN
from models.translator import SignLanguageTranslator

# Tiny dimensions to keep tests fast on CPU
B, T, H, W = 2, 8, 64, 64   # smaller than real (256?256) for speed
S = 10
VOCAB = 50


def _dummy_batch(device="cpu"):
    frames     = torch.randn(B, T, 3, H, W, device=device)
    heatmaps   = torch.rand(B, T, H, W, device=device)
    frame_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    token_ids  = torch.randint(4, VOCAB, (B, S), device=device)
    token_ids[:, 0] = BOS_ID
    token_ids[:, -1] = EOS_ID
    return frames, heatmaps, frame_mask, token_ids


def _make_model(freeze=True) -> SignLanguageTranslator:
    # Override IMG_SIZE config via monkey-patch isn't needed -- model accepts any H,W
    return SignLanguageTranslator(vocab_size=VOCAB, d_model=D_MODEL, freeze_backbone=freeze)


# ??? Gate Checks ?????????????????????????????????????????????????????????????

def test_4_1_full_forward_pass():
    """4.1 -- Full forward pass produces correct output shape."""
    model = _make_model()
    model.eval()
    frames, heatmaps, frame_mask, token_ids = _dummy_batch()

    tgt_input = token_ids[:, :-1]

    with torch.no_grad():
        logits = model(frames, tgt_input, heatmaps=heatmaps, frame_mask=frame_mask)

    expected = (B, S - 1, VOCAB)
    assert logits.shape == expected, f"Expected {expected}, got {logits.shape}"
    print(f"  [PASS] 4.1  logits={logits.shape}")


def test_4_2_loss_decreases():
    """4.2 -- Loss decreases when overfitting on a single batch (5 steps)."""
    model = _make_model(freeze=True)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING)

    frames, heatmaps, frame_mask, token_ids = _dummy_batch()
    tgt_input  = token_ids[:, :-1]
    tgt_target = token_ids[:, 1:]

    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        logits = model(frames, tgt_input, heatmaps=heatmaps, frame_mask=frame_mask)
        loss = criterion(logits.reshape(-1, VOCAB), tgt_target.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    print(f"  [PASS] 4.2  loss {losses[0]:.4f} -> {losses[-1]:.4f}")


def test_4_3_no_nan_inf():
    """4.3 -- No NaN/Inf in loss or gradients."""
    model = _make_model()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    frames, heatmaps, frame_mask, token_ids = _dummy_batch()
    tgt_input  = token_ids[:, :-1]
    tgt_target = token_ids[:, 1:]

    logits = model(frames, tgt_input, heatmaps=heatmaps, frame_mask=frame_mask)
    loss = criterion(logits.reshape(-1, VOCAB), tgt_target.reshape(-1))
    loss.backward()

    assert not torch.isnan(loss), f"Loss is NaN"
    assert not torch.isinf(loss), f"Loss is Inf"

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    print(f"  [PASS] 4.3  loss={loss.item():.4f}  no NaN/Inf in gradients")


def test_4_4_translate_output():
    """4.4 -- translate() returns a list of strings (non-None)."""
    model = _make_model()
    frames, heatmaps, frame_mask, _ = _dummy_batch()

    class FakeTok:
        def decode(self, ids, skip_special=True):
            return " ".join(str(i) for i in ids if i not in {PAD_ID, BOS_ID, EOS_ID})

    result = model.translate(frames, FakeTok(), heatmaps=heatmaps, frame_mask=frame_mask, beam_size=1)

    assert isinstance(result, list) and len(result) == B
    assert all(isinstance(s, str) for s in result)
    print(f"  [PASS] 4.4  translate() returned {B} strings: {result[0][:30]!r}...")


def test_4_5_checkpoint_save_load():
    """4.5 -- Checkpoint saves and loads without error; weights are identical."""
    model = _make_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        torch.save({
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": 3.14,
            "best_bleu": 0.0,
        }, ckpt_path)

        # Load into a fresh model
        model2 = _make_model()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model2.load_state_dict(ckpt["model_state_dict"])

        # Verify weights are identical
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Weights differ after load: {n1}"

    print(f"  [PASS] 4.5  checkpoint save/load verified")


def test_4_6_phase2_unfreeze():
    """4.6 -- Phase 2 unfreezes backbone (requires_grad propagates)."""
    model = _make_model(freeze=True)

    # Verify frozen
    for p in model.visual_encoder.backbone.parameters():
        assert not p.requires_grad, "Backbone should be frozen in phase 1"

    # Switch to phase 2
    model.set_phase(2)

    frozen = [p for p in model.visual_encoder.backbone.parameters() if not p.requires_grad]
    assert len(frozen) == 0, f"{len(frozen)} backbone params still frozen after phase 2"
    print("  [PASS] 4.6  phase 1 frozen -> phase 2 fully unfrozen")


def test_4_7_gradient_accumulation():
    """4.7 -- Gradient accumulation: loss /= accum_steps; optimizer steps only every N steps."""
    model = _make_model()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    frames, heatmaps, frame_mask, token_ids = _dummy_batch()
    tgt_input  = token_ids[:, :-1]
    tgt_target = token_ids[:, 1:]

    ACCUM = 4
    optimizer.zero_grad()
    step_count = 0

    for i in range(ACCUM):
        logits = model(frames, tgt_input, heatmaps=heatmaps, frame_mask=frame_mask)
        loss = criterion(logits.reshape(-1, VOCAB), tgt_target.reshape(-1)) / ACCUM
        loss.backward()
        step_count += 1

    # Gradients should be accumulated (non-zero) before optimizer step
    has_grad = any(p.grad is not None and p.grad.abs().max() > 0
                   for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients accumulated"

    optimizer.step()
    optimizer.zero_grad()

    # After zero_grad, all grads should be None or zero
    still_has_grad = any(p.grad is not None and p.grad.abs().max() > 1e-12
                         for p in model.parameters() if p.requires_grad)
    assert not still_has_grad, "Gradients not cleared after zero_grad"

    print(f"  [PASS] 4.7  gradient accumulation over {ACCUM} steps, then step+zero_grad OK")


# ??? Runner ??????????????????????????????????????????????????????????????????

if __name__ == "__main__":
    tests = [
        test_4_1_full_forward_pass,
        test_4_2_loss_decreases,
        test_4_3_no_nan_inf,
        test_4_4_translate_output,
        test_4_5_checkpoint_save_load,
        test_4_6_phase2_unfreeze,
        test_4_7_gradient_accumulation,
    ]

    passed = failed = 0
    print("\n=== F4 Gate Checks: Full Model + Training Loop ===\n")
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"  PASSED: {passed}/{len(tests)}")
    if failed:
        print(f"  FAILED: {failed}/{len(tests)}")
        sys.exit(1)
    else:
        print("  ALL F4 GATE CHECKS PASSED OK")
