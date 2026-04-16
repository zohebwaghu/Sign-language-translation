"""
F5 — Ablation experiments.

Runs the same evaluation with/without spatial attention to quantify
the contribution of the landmark-aware attention mechanism.

Ablation table:
  | Config                    | BLEU-4 | ROUGE-L | METEOR |
  |---------------------------|--------|---------|--------|
  | Full model (+ attention)  |  XX.X  |   XX.X  |  XX.X  |
  | No spatial attention      |  XX.X  |   XX.X  |  XX.X  |
  | No landmarks (zeros)      |  XX.X  |   XX.X  |  XX.X  |
"""

import logging
from typing import List, Dict, Optional
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def run_ablation(
    model_full,
    tokenizer,
    test_loader: DataLoader,
    device: torch.device,
    beam_size: int = 1,
) -> Dict[str, Dict[str, float]]:
    """
    Run three evaluation passes:
      1. Full model (with spatial attention + real heatmaps)
      2. No spatial attention (pass heatmaps=None)
      3. No landmarks (pass zero heatmaps)

    Returns:
        Dict mapping config name → metrics dict (BLEU, ROUGE, METEOR).
    """
    from evaluation.metrics import compute_all

    configs = {
        "full_model":        lambda batch: _predict(model_full, tokenizer, batch, device, beam_size, use_heatmaps=True),
        "no_spatial_attn":   lambda batch: _predict(model_full, tokenizer, batch, device, beam_size, use_heatmaps=False),
        "zero_heatmaps":     lambda batch: _predict(model_full, tokenizer, batch, device, beam_size, use_heatmaps=True, zero_heatmaps=True),
    }

    results = {}
    all_references = []

    for cfg_name, predict_fn in configs.items():
        logger.info(f"\n  Ablation: {cfg_name}")
        hypotheses = []
        references = []

        for batch in test_loader:
            preds = predict_fn(batch)
            hypotheses.extend(preds)
            references.extend(batch["text"])

        if not all_references:
            all_references = references

        metrics = compute_all(hypotheses, references)
        results[cfg_name] = metrics
        logger.info(f"    BLEU-4={metrics.get('bleu4', 0):.2f}  ROUGE-L={metrics.get('rougeL_f1', 0):.4f}  METEOR={metrics.get('meteor', 0):.4f}")

    return results


def _predict(
    model,
    tokenizer,
    batch: Dict,
    device: torch.device,
    beam_size: int,
    use_heatmaps: bool = True,
    zero_heatmaps: bool = False,
) -> List[str]:
    frames     = batch["frames"].to(device)
    frame_mask = batch["frame_mask"].to(device)

    if use_heatmaps and not zero_heatmaps:
        heatmaps = batch["heatmaps"].to(device)
    elif zero_heatmaps and use_heatmaps:
        heatmaps = torch.zeros_like(batch["heatmaps"]).to(device)
    else:
        heatmaps = None

    return model.translate(frames, tokenizer, heatmaps=heatmaps, frame_mask=frame_mask, beam_size=beam_size)


def print_ablation_table(results: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print the ablation comparison table."""
    configs = list(results.keys())
    header = f"{'Config':<25} {'BLEU-4':>8} {'ROUGE-L':>8} {'METEOR':>8}"
    print("\n" + "=" * 55)
    print("  Ablation Results")
    print("=" * 55)
    print(header)
    print("-" * 55)
    for cfg in configs:
        m = results[cfg]
        bleu   = m.get("bleu4",       0.0)
        rouge  = m.get("rougeL_f1",   0.0)
        meteor = m.get("meteor",       0.0)
        print(f"  {cfg:<23} {bleu:>8.2f} {rouge:>8.4f} {meteor:>8.4f}")
    print("=" * 55 + "\n")

    # Compute delta vs full model
    if "full_model" in results and "no_spatial_attn" in results:
        delta = results["full_model"].get("bleu4", 0) - results["no_spatial_attn"].get("bleu4", 0)
        print(f"  Spatial attention contribution: {delta:+.2f} BLEU-4 points\n")
