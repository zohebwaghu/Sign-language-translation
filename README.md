# Sign Language Translation — ASL Video to English Text

**Team:** Kalyan Mamidi · Venkaat Ramireddy Seelam · Shreyas Durairajalu · Zoheb Waghu  
**Dataset:** [How2Sign](https://how2sign.github.io/) (`bdanko/how2sign-rgb-front-clips` on HuggingFace)  
**Task:** Continuous Sign Language Translation — ASL video → English sentence  
**Target:** BLEU-4 > 10 on How2Sign test set

---

## Architecture

```
Video (T frames, 256×256×3)
    │
    ▼
[F0] Frame Extraction + MediaPipe Landmarks (75 keypoints/frame)
    │
    ▼
[F1] Visual Encoder  ←  ResNet-18 + Landmark-Aware SpatialAttention
    │  (B, T, 3, 256, 256) → (B, T, 512)
    ▼
[F2] Temporal Transformer Encoder (4 layers, 8 heads, sinusoidal PE)
    │  (B, T, 512) → (B, T, 512) contextualised
    ▼
[F3] Transformer Decoder (4 layers, cross-attention, beam search k=4)
    │  autoregressive: generates English BPE tokens
    ▼
[F5] English Sentence → BLEU-4 / ROUGE-L / METEOR / Claude LLM-as-Judge
```

**Novelty:** Landmark-aware spatial attention fuses MediaPipe hand/face keypoints as a Gaussian attention prior over CNN feature maps — learned to focus on signing regions without extra supervision.

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run all gate checks (CPU, no GPU needed)
```bash
python tests/test_feature0.py   # F0: data pipeline   (6 checks)
python tests/test_feature1.py   # F1: visual encoder  (5 checks)
python tests/test_feature3.py   # F2+F3: encoder/decoder (10 checks)
python tests/test_feature4.py   # F4: training loop   (7 checks)
python tests/test_feature5.py   # F5: evaluation      (7 checks)
```

All 35 gate checks pass on CPU with synthetic data.

### Train the model
```bash
# Sanity check: overfit on 100 clips first
python training/train.py --overfit-test --no-wandb

# Full training (2 phases, 30 epochs total)
python training/train.py

# Resume from checkpoint
python training/train.py --resume checkpoints/checkpoint_best.pt
```

### Evaluate a trained model
```python
from models.translator import SignLanguageTranslator
from data.tokenizer import SignTokenizer
from evaluation.metrics import compute_all, print_results
from evaluation.llm_judge import judge_batch, summarise, print_examples

# Load model + tokenizer
tokenizer = SignTokenizer()
tokenizer.load("data/tokenizer.model")
model = SignLanguageTranslator(vocab_size=tokenizer.vocab_size)
# model.load_state_dict(torch.load("checkpoints/checkpoint_best.pt")["model_state_dict"])

# Run metrics
metrics = compute_all(hypotheses, references)
print_results(metrics, n_samples=len(hypotheses))

# Claude LLM-as-Judge (requires ANTHROPIC_API_KEY)
scores = judge_batch(hypotheses, references, max_samples=200)
print_examples(scores, n=20)
```

---

## Project Structure

```
sign-language-translation/
├── data/
│   ├── download.py       # HuggingFace streaming loader, frame extraction
│   ├── landmarks.py      # MediaPipe 75-keypoint extraction + Gaussian heatmaps
│   ├── dataset.py        # How2SignDataset + collate_fn
│   └── tokenizer.py      # SentencePiece BPE tokenizer (~8K subwords)
├── models/
│   ├── visual_encoder.py # ResNet-18 + SpatialAttention (F1)
│   ├── temporal_encoder.py # Transformer Encoder + sinusoidal PE (F2)
│   ├── text_decoder.py   # Transformer Decoder + beam search (F3)
│   └── translator.py     # SignLanguageTranslator — full model (F4)
├── training/
│   ├── config.py         # All hyperparameters
│   └── train.py          # Training loop: fp16, WandB, grad accum, 2-phase
├── evaluation/
│   ├── metrics.py        # BLEU-4, ROUGE-L, METEOR
│   ├── llm_judge.py      # Claude-as-judge: adequacy/fluency/meaning scoring
│   └── ablation.py       # Spatial attention ablation table
├── tests/                # Gate-checked test suite (35 checks total)
├── notebooks/            # Data exploration
├── checkpoints/          # Saved model weights (gitignored)
└── requirements.txt
```

---

## Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| d_model | 512 | Standard transformer dimension |
| Heads | 8 | 64 dim/head |
| Encoder layers | 4 | Sufficient for temporal modeling |
| Decoder layers | 4 | Matches encoder depth |
| dim_ff | 2048 | 4× d_model (standard) |
| Effective batch | 32 | 4 micro × 8 accumulation steps |
| Max frames | 64 | Covers most clips; increase to 128 if memory allows |
| Max text len | 64 | Covers 95% of sentences |
| Vocab size | 8K | SentencePiece BPE subwords |
| Phase 1 LR | 3e-4 | Backbone frozen, 15 epochs |
| Phase 2 LR | 3e-5 | Backbone unfrozen, 15 epochs |

---

## SOTA Comparison

| Model | Venue | BLEU-4 | Key Difference |
|---|---|---|---|
| SLTIV (Tarrés 2023) | — | 8.03 | Seq2Seq baseline |
| YouTube-ASL (Uthus 2023) | NeurIPS | 12.39 | Pretrained I3D + T5 |
| Uni-Sign (Li 2025) | ICLR | 14.90 | Massive unified pretraining |
| SSVP-SLT (Rust 2024) | ACL | **15.50** | SignHiera + T5 + MAE pretrain |
| **Ours (Target)** | — | **>10** | From-scratch decoder + landmark attention |

---

## Team Responsibilities

| Member | Modules | 
|---|---|
| **Kalyan** | F0 — Data pipeline, frame extraction, MediaPipe landmarks |
| **Venkaat** | F1 — Visual encoder, spatial attention, ResNet backbone |
| **Shreyas** | F2 + F3 — Temporal encoder, decoder, tokenizer |
| **Zoheb** | F4 + F5 — Training loop, evaluation, LLM-as-Judge, ablations |
