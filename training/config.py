"""
Centralised hyperparameters for Sign Language Translation.
All training/model code imports from here — never hardcode values elsewhere.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
DATA_CACHE_DIR = ROOT / "data" / "cache"
TOKENIZER_MODEL = ROOT / "data" / "tokenizer.model"

# ─── Dataset ──────────────────────────────────────────────────────────────────

DATASET_NAME = "bdanko/how2sign-rgb-front-clips"
DATASET_SPLITS = {"train": "train", "val": "val", "test": "test"}

# ─── Video / Frame ────────────────────────────────────────────────────────────

IMG_SIZE = 256                  # resize all frames to IMG_SIZE × IMG_SIZE
MAX_FRAMES = 64                 # temporal clip length (increase to 128 if GPU allows)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Landmarks ────────────────────────────────────────────────────────────────

N_LANDMARKS = 75                # 33 pose + 21 left hand + 21 right hand
HEATMAP_SIGMA = 8               # Gaussian blob spread in pixels

# ─── Tokenizer ────────────────────────────────────────────────────────────────

VOCAB_SIZE = 8_000              # SentencePiece BPE subwords
PAD_ID   = 0
BOS_ID   = 1
EOS_ID   = 2
UNK_ID   = 3
MAX_TEXT_LEN = 64               # max decoder sequence length

# ─── Model Architecture ───────────────────────────────────────────────────────

D_MODEL       = 512
N_HEADS       = 8               # D_MODEL / N_HEADS = 64
DIM_FEEDFORWARD = 2048          # 4 × D_MODEL
N_ENC_LAYERS  = 4
N_DEC_LAYERS  = 4
DROPOUT       = 0.1

# ─── Training ─────────────────────────────────────────────────────────────────

MICRO_BATCH_SIZE     = 4        # per GPU
GRAD_ACCUM_STEPS     = 8        # effective batch = MICRO_BATCH_SIZE × GRAD_ACCUM_STEPS = 32
GRAD_CLIP_NORM       = 1.0
LABEL_SMOOTHING      = 0.1
WEIGHT_DECAY         = 0.01
NUM_WORKERS          = 4

# Phase 1 — backbone frozen
PHASE1_EPOCHS        = 15
PHASE1_LR            = 3e-4

# Phase 2 — backbone unfrozen
PHASE2_EPOCHS        = 15
PHASE2_LR            = 3e-5

TOTAL_EPOCHS         = PHASE1_EPOCHS + PHASE2_EPOCHS

# Mixed precision
USE_AMP              = True     # fp16 automatic mixed precision

# Checkpointing
CHECKPOINT_EVERY     = 5        # save every N epochs

# ─── Inference ────────────────────────────────────────────────────────────────

BEAM_SIZE            = 4        # beam search width (1 = greedy)

# ─── Evaluation ───────────────────────────────────────────────────────────────

LLM_JUDGE_MODEL      = "claude-sonnet-4-6"
LLM_JUDGE_SAMPLES    = 200      # how many test samples to score
LLM_JUDGE_BATCH_SIZE = 10       # samples per API call

# ─── WandB ────────────────────────────────────────────────────────────────────

WANDB_PROJECT = "sign-language-translation"
WANDB_ENTITY  = None            # set to your wandb username / team
