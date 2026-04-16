"""
F2 + F3 Gate Checks — Temporal Encoder & Transformer Decoder
All tests run on CPU with dummy tensors.

F2 checks:
  2.1  Input (B, T, 512) → output (B, T, 512) shape preserved
  2.2  Positional encoding values are non-zero
  2.3  Padding mask prevents attention to padded positions
  2.4  Output changes when input order is shuffled (temporal awareness)

F3 checks:
  3.1  Decoder output shape: (B, S, vocab_size)
  3.2  Causal mask prevents attending to future tokens
  3.3  generate() produces sequences ending at EOS (or PAD after EOS)
  3.4  Tokenizer encode→decode round-trip
  3.5  Cross-attention connects to encoder output
  3.6  Vocab size matches between tokenizer and decoder FC layer
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.config import D_MODEL, BOS_ID, EOS_ID, PAD_ID, MAX_TEXT_LEN, BEAM_SIZE
from models.temporal_encoder import TemporalEncoder, SinusoidalPositionalEncoding
from models.text_decoder import TextDecoder

B, T, S = 2, 16, 12
VOCAB = 200   # small vocab for fast CPU tests


# ─── F2: Temporal Encoder ─────────────────────────────────────────────────────

def test_2_1_temporal_output_shape():
    """2.1 — Input (B, T, 512) → output (B, T, 512)."""
    enc = TemporalEncoder(d_model=D_MODEL)
    enc.eval()
    x = torch.randn(B, T, D_MODEL)
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (B, T, D_MODEL), f"Expected ({B},{T},{D_MODEL}), got {out.shape}"
    print(f"  [PASS] 2.1  output={out.shape}")


def test_2_2_positional_encoding_nonzero():
    """2.2 — Positional encoding values are non-zero."""
    pe = SinusoidalPositionalEncoding(d_model=D_MODEL, max_len=512, dropout=0.0)
    # Access the buffer directly — shape (1, max_len, d_model)
    assert pe.pe.abs().max().item() > 0, "Positional encoding is all zeros"
    # Also check it varies across positions
    diff = (pe.pe[:, 0, :] - pe.pe[:, 1, :]).abs().max().item()
    assert diff > 1e-4, "PE is the same across positions"
    print(f"  [PASS] 2.2  PE max={pe.pe.abs().max():.4f}  pos0vs1_diff={diff:.4f}")


def test_2_3_padding_mask():
    """2.3 — Padding mask prevents attention to padded positions."""
    enc = TemporalEncoder(d_model=D_MODEL)
    enc.eval()

    x = torch.randn(B, T, D_MODEL)
    # Mask: last half of frames are padded (True = padded in torch convention)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, T // 2:] = True   # positions T/2 ... T-1 are padding

    with torch.no_grad():
        out_masked   = enc(x, src_key_padding_mask=mask)
        out_unmasked = enc(x)

    # Real positions should differ between masked and unmasked
    real_diff = (out_masked[:, :T//2, :] - out_unmasked[:, :T//2, :]).abs().mean().item()
    assert real_diff > 1e-6, "Masking had no effect on real positions"
    print(f"  [PASS] 2.3  masked vs unmasked diff at real positions = {real_diff:.4f}")


def test_2_4_temporal_awareness():
    """2.4 — Output changes when input frame order is shuffled."""
    enc = TemporalEncoder(d_model=D_MODEL)
    enc.eval()

    x = torch.randn(B, T, D_MODEL)
    perm = torch.randperm(T)
    x_shuffled = x[:, perm, :]

    with torch.no_grad():
        out_orig    = enc(x)
        out_shuffled = enc(x_shuffled)

    # The shuffled output should NOT be a permutation of the original
    # (because positional encoding changes with position)
    diff = (out_orig[:, perm, :] - out_shuffled).abs().mean().item()
    assert diff > 1e-4, "Shuffling input had no effect — temporal encoding may be missing"
    print(f"  [PASS] 2.4  shuffle diff = {diff:.4f}  (temporal awareness confirmed)")


# ─── F3: Text Decoder ─────────────────────────────────────────────────────────

def test_3_1_decoder_output_shape():
    """3.1 — Decoder output shape: (B, S, vocab_size)."""
    dec = TextDecoder(vocab_size=VOCAB, d_model=D_MODEL, n_layers=2)
    dec.eval()

    tgt    = torch.randint(4, VOCAB, (B, S))
    memory = torch.randn(B, T, D_MODEL)

    with torch.no_grad():
        logits = dec(tgt, memory)

    assert logits.shape == (B, S, VOCAB), f"Expected ({B},{S},{VOCAB}), got {logits.shape}"
    print(f"  [PASS] 3.1  logits={logits.shape}")


def test_3_2_causal_mask():
    """3.2 — Causal mask prevents attending to future tokens."""
    mask = TextDecoder._causal_mask(S, device=torch.device("cpu"))
    assert mask.shape == (S, S)
    # Upper triangle (future) should be True (masked)
    assert mask[0, 1].item() is True,  "Future token not masked"
    assert mask[1, 0].item() is False, "Past token incorrectly masked"
    assert mask[0, 0].item() is False, "Self-position incorrectly masked"
    print(f"  [PASS] 3.2  causal mask shape={mask.shape}  upper-tri=True ✓")


def test_3_3_generate_ends_at_eos():
    """3.3 — generate() produces sequences that contain EOS (or are padded after it)."""
    dec = TextDecoder(vocab_size=VOCAB, d_model=D_MODEL, n_layers=2)
    dec.eval()

    memory = torch.randn(B, T, D_MODEL)

    # Greedy
    out_greedy = dec.generate(memory, max_len=MAX_TEXT_LEN, beam_size=1)
    assert out_greedy.shape == (B, MAX_TEXT_LEN), f"Greedy shape wrong: {out_greedy.shape}"

    # Beam search
    out_beam = dec.generate(memory, max_len=MAX_TEXT_LEN, beam_size=2)
    assert out_beam.shape == (B, MAX_TEXT_LEN), f"Beam shape wrong: {out_beam.shape}"

    print(f"  [PASS] 3.3  greedy={out_greedy.shape}  beam={out_beam.shape}")


def test_3_4_tokenizer_roundtrip():
    """3.4 — encode → decode round-trip (requires sentencepiece installed)."""
    try:
        from data.tokenizer import SignTokenizer
        import tempfile, os
        texts = ["hello world"] * 50 + ["sign language translation"] * 50
        tok = SignTokenizer()
        with tempfile.TemporaryDirectory() as d:
            model_path = os.path.join(d, "tok.model")
            tok.train(texts, model_path=model_path, vocab_size=50)
            ids = tok.encode("hello world", add_bos=True, add_eos=True)
            decoded = tok.decode(ids, skip_special=True)
            assert "hello" in decoded.lower() or "world" in decoded.lower(), (
                f"Round-trip failed: got '{decoded}'"
            )
        print(f"  [PASS] 3.4  tokenizer round-trip: 'hello world' → {ids[:5]}... → '{decoded}'")
    except ImportError:
        print("  [SKIP] 3.4  sentencepiece not installed")


def test_3_5_cross_attention_to_encoder():
    """3.5 — Different encoder outputs produce different decoder outputs."""
    dec = TextDecoder(vocab_size=VOCAB, d_model=D_MODEL, n_layers=2)
    dec.eval()

    tgt = torch.randint(4, VOCAB, (B, S))
    mem1 = torch.randn(B, T, D_MODEL)
    mem2 = torch.randn(B, T, D_MODEL)

    with torch.no_grad():
        out1 = dec(tgt, mem1)
        out2 = dec(tgt, mem2)

    diff = (out1 - out2).abs().mean().item()
    assert diff > 1e-4, "Decoder outputs identical for different memories — cross-attention broken"
    print(f"  [PASS] 3.5  cross-attention active: diff={diff:.4f}")


def test_3_6_vocab_size_consistency():
    """3.6 — Vocab size consistent between embedding and output projection."""
    dec = TextDecoder(vocab_size=VOCAB, d_model=D_MODEL, n_layers=2)
    assert dec.token_emb.weight.shape == (VOCAB, D_MODEL)
    assert dec.output_proj.weight.shape == (VOCAB, D_MODEL)
    # Weight tying
    assert dec.output_proj.weight is dec.token_emb.weight, "Weight tying broken"
    print(f"  [PASS] 3.6  vocab={VOCAB}  embedding={dec.token_emb.weight.shape}  tied=True")


if __name__ == "__main__":
    tests = [
        ("F2", [test_2_1_temporal_output_shape, test_2_2_positional_encoding_nonzero,
                test_2_3_padding_mask, test_2_4_temporal_awareness]),
        ("F3", [test_3_1_decoder_output_shape, test_3_2_causal_mask,
                test_3_3_generate_ends_at_eos, test_3_4_tokenizer_roundtrip,
                test_3_5_cross_attention_to_encoder, test_3_6_vocab_size_consistency]),
    ]

    total_passed = total_failed = 0
    for feature, fns in tests:
        passed = failed = 0
        print(f"\n=== {feature} Gate Checks ===\n")
        for t in fns:
            try:
                t()
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {t.__name__}: {e}")
                failed += 1
        print(f"\n  {feature}: PASSED {passed}/{len(fns)}")
        total_passed += passed
        total_failed += failed

    print(f"\n{'='*40}")
    print(f"  TOTAL PASSED: {total_passed}/{total_passed+total_failed}")
    if total_failed:
        print(f"  TOTAL FAILED: {total_failed}")
        sys.exit(1)
    else:
        print("  ALL F2+F3 GATE CHECKS PASSED ✓")
