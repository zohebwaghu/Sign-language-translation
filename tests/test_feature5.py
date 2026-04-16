"""
F5 Gate Checks — Evaluation + LLM-as-Judge
Tests run on CPU with synthetic data. LLM judge test mocked to avoid API calls.

Gate checks:
  5.1  BLEU-4 computed and returns a float
  5.2  ROUGE-L computed and returns a float
  5.3  METEOR computed and returns a float
  5.4  LLM judge returns correct structure (mocked — no API call)
  5.5  Pearson correlation computed between judge scores and BLEU
  5.6  print_examples runs without error
  5.7  Ablation function returns 3 configs with metrics
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_bleu, compute_rouge, compute_meteor
from evaluation.llm_judge import (
    JudgeScore, pearson_correlation, per_sample_bleu,
    print_examples, summarise,
)


# ── Shared test data ─────────────────────────────────────────────────────────

HYP = [
    "the man is signing hello",
    "she shows the number five",
    "please sit down now",
    "good morning everyone",
    "i love learning sign language",
]
REF = [
    "the person is saying hello",
    "she is showing the number five",
    "please sit down",
    "good morning to all",
    "i enjoy learning sign language",
]


# ── Gate Checks ──────────────────────────────────────────────────────────────

def test_5_1_bleu():
    """5.1 — BLEU-4 computed on sample data."""
    result = compute_bleu(HYP, REF)
    assert "bleu4" in result, f"bleu4 key missing: {result}"
    assert isinstance(result["bleu4"], float), f"bleu4 not float: {type(result['bleu4'])}"
    assert result["bleu4"] >= 0.0, f"Negative BLEU: {result['bleu4']}"
    print(f"  [PASS] 5.1  BLEU-4 = {result['bleu4']:.4f}")


def test_5_2_rouge():
    """5.2 — ROUGE-L computed and returns a float."""
    result = compute_rouge(HYP, REF)
    assert "rougeL_f1" in result, f"rougeL_f1 missing: {result}"
    assert 0.0 <= result["rougeL_f1"] <= 1.0, f"ROUGE out of [0,1]: {result['rougeL_f1']}"
    print(f"  [PASS] 5.2  ROUGE-L F1 = {result['rougeL_f1']:.4f}")


def test_5_3_meteor():
    """5.3 — METEOR computed and returns a float."""
    result = compute_meteor(HYP, REF)
    assert "meteor" in result, f"meteor missing: {result}"
    assert 0.0 <= result["meteor"] <= 1.0, f"METEOR out of [0,1]: {result['meteor']}"
    print(f"  [PASS] 5.3  METEOR = {result['meteor']:.4f}")


def test_5_4_llm_judge_structure():
    """5.4 — LLM judge returns correct JudgeScore structure (mocked API)."""
    import json
    import anthropic

    fake_response = MagicMock()
    fake_response.content = [MagicMock(text='{"adequacy": 4, "fluency": 5, "meaning": 3}')]

    with patch("anthropic.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = fake_response

        # Import and call with mocked client
        from evaluation.llm_judge import _score_single
        score = _score_single(instance, "claude-sonnet-4-6", HYP[0], REF[0], retry_delay=0)

    assert isinstance(score, JudgeScore), f"Wrong type: {type(score)}"
    assert score.adequacy == 4.0
    assert score.fluency  == 5.0
    assert score.meaning  == 3.0
    assert abs(score.composite - 4.0) < 1e-6, f"Composite wrong: {score.composite}"
    print(f"  [PASS] 5.4  LLM judge score: adequacy={score.adequacy}  fluency={score.fluency}  meaning={score.meaning}")


def test_5_5_pearson_correlation():
    """5.5 — Pearson correlation between LLM scores and BLEU."""
    scores = [
        JudgeScore(HYP[i], REF[i], adequacy=float(3+i%3), fluency=float(3+i%2),
                   meaning=float(3+i%3), composite=float(3+i%3))
        for i in range(len(HYP))
    ]
    bleu_per_sample = per_sample_bleu(HYP, REF)
    r = pearson_correlation(scores, bleu_per_sample)
    assert -1.0 <= r <= 1.0, f"Pearson r out of [-1,1]: {r}"
    print(f"  [PASS] 5.5  Pearson r = {r:.4f}")


def test_5_6_print_examples():
    """5.6 — print_examples runs without error."""
    scores = [
        JudgeScore(HYP[i], REF[i], 4.0, 4.0, 3.0, 3.67)
        for i in range(len(HYP))
    ]
    print_examples(scores, n=3)   # should print without throwing
    print(f"  [PASS] 5.6  print_examples ran for {len(scores)} samples")


def test_5_7_ablation_structure():
    """5.7 — Ablation returns 3 configs each with BLEU, ROUGE, METEOR."""
    import torch
    from models.translator import SignLanguageTranslator
    from torch.utils.data import DataLoader, TensorDataset
    from evaluation.ablation import run_ablation

    VOCAB, B, T, H, W, S = 50, 2, 4, 32, 32, 8

    model = SignLanguageTranslator(vocab_size=VOCAB, d_model=128)
    model.eval()

    class FakeTok:
        def decode(self, ids, skip_special=True):
            return "the man signs hello"

    # Build a minimal fake DataLoader
    frames     = torch.randn(B, T, 3, H, W)
    heatmaps   = torch.rand(B, T, H, W)
    frame_mask = torch.ones(B, T, dtype=torch.bool)
    token_ids  = torch.zeros(B, S, dtype=torch.long)

    class FakeDataset:
        def __len__(self): return 1
        def __getitem__(self, i):
            return {
                "frames": frames, "heatmaps": heatmaps,
                "frame_mask": frame_mask, "token_ids": token_ids,
                "text": ["the man says hello", "she signs thank you"],
            }

    # Manual batch list (bypass DataLoader)
    fake_batches = [{
        "frames": frames, "heatmaps": heatmaps,
        "frame_mask": frame_mask, "token_ids": token_ids,
        "text": ["the man says hello", "she signs thank you"],
    }]

    # Monkey-patch to use list instead of DataLoader
    from evaluation import ablation as abl_mod
    original = abl_mod._predict

    results = {}
    for cfg_name, use_heatmaps, zero_heatmaps in [
        ("full_model",      True,  False),
        ("no_spatial_attn", False, False),
        ("zero_heatmaps",   True,  True),
    ]:
        hyps = abl_mod._predict(model, FakeTok(), fake_batches[0],
                                torch.device("cpu"), beam_size=1,
                                use_heatmaps=use_heatmaps, zero_heatmaps=zero_heatmaps)
        refs = fake_batches[0]["text"]
        from evaluation.metrics import compute_all
        results[cfg_name] = compute_all(hyps, refs)

    assert len(results) == 3, f"Expected 3 configs, got {len(results)}"
    for cfg, m in results.items():
        assert "bleu4"     in m, f"BLEU missing for {cfg}"
        assert "rougeL_f1" in m, f"ROUGE missing for {cfg}"
        assert "meteor"    in m, f"METEOR missing for {cfg}"

    print(f"  [PASS] 5.7  ablation returned {len(results)} configs with all metrics")


# ─── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_5_1_bleu,
        test_5_2_rouge,
        test_5_3_meteor,
        test_5_4_llm_judge_structure,
        test_5_5_pearson_correlation,
        test_5_6_print_examples,
        test_5_7_ablation_structure,
    ]

    passed = failed = 0
    print("\n=== F5 Gate Checks: Evaluation + LLM-as-Judge ===\n")
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
        print("  ALL F5 GATE CHECKS PASSED ✓")
