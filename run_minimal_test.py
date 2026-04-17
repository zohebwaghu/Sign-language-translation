"""
Minimal end-to-end smoke test for Sign Language Translation.
Uses 100% synthetic data — no HuggingFace download, no GPU required.
Expected runtime: ~1-3 min on CPU.
"""

import sys
import io
import logging
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Patch config BEFORE importing any project modules ────────────────────────
import training.config as cfg

cfg.MAX_FRAMES       = 8
cfg.IMG_SIZE         = 64
cfg.D_MODEL          = 128
cfg.N_HEADS          = 2
cfg.N_ENC_LAYERS     = 2
cfg.N_DEC_LAYERS     = 2
cfg.DIM_FEEDFORWARD  = 256
cfg.VOCAB_SIZE       = 100
cfg.MAX_TEXT_LEN     = 16
cfg.MICRO_BATCH_SIZE = 2
cfg.GRAD_ACCUM_STEPS = 1
cfg.NUM_WORKERS      = 0
cfg.USE_AMP          = False   # no AMP on CPU
cfg.BEAM_SIZE        = 1       # greedy for speed

TOKENIZER_PATH = ROOT / "data" / "tokenizer_minimal.model"
cfg.TOKENIZER_MODEL = TOKENIZER_PATH

# ── Now safe to import project modules ───────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.download import extract_frames
from data.dataset import How2SignDataset, collate_fn
from data.tokenizer import SignTokenizer
from models.translator import SignLanguageTranslator
from training.config import MAX_FRAMES, IMG_SIZE, MAX_TEXT_LEN, PAD_ID


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _make_video_bytes(n_frames: int = 12) -> bytes:
    try:
        import av
        buf = io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=25)
        stream.width  = IMG_SIZE
        stream.height = IMG_SIZE
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23"}
        for _ in range(n_frames):
            data = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(data, format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Video encoding failed ({e}), using empty bytes (zeros)")
        return b""


TEXTS = [
    "the cat sat on the mat",
    "she is signing hello to you",
    "please turn left at the corner",
    "good morning how are you today",
    "i would like some water please",
    "thank you very much for helping",
    "where is the nearest bathroom",
    "my name is alice nice to meet you",
    "the weather is nice today outside",
    "can you help me find the store",
    "i am learning sign language now",
    "see you later goodbye have a nice day",
    "what time is it please tell me",
    "i need to go home soon",
    "the quick brown fox jumps over",
    "nice to meet you welcome here",
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cfg.USE_AMP = True   # re-enable AMP on GPU
        logger.info(f"Device: {device}  ({torch.cuda.get_device_name(0)})")
    else:
        logger.info(f"Device: {device}")
    logger.info(
        f"Config: MAX_FRAMES={MAX_FRAMES}, IMG_SIZE={IMG_SIZE}, "
        f"D_MODEL={cfg.D_MODEL}, N_HEADS={cfg.N_HEADS}, "
        f"N_ENC_LAYERS={cfg.N_ENC_LAYERS}, N_DEC_LAYERS={cfg.N_DEC_LAYERS}"
    )

    # ── 1. Tokenizer ──────────────────────────────────────────────────────
    logger.info("\n[1/4] Training tokenizer on synthetic corpus...")
    tokenizer = SignTokenizer()
    tokenizer.train(TEXTS * 20, model_path=TOKENIZER_PATH, vocab_size=100)
    logger.info(f"      vocab_size={tokenizer.vocab_size}")

    # ── 2. Dataset ────────────────────────────────────────────────────────
    logger.info("\n[2/4] Building synthetic dataset...")
    video_bytes = _make_video_bytes(n_frames=12)
    records = [
        {"video": video_bytes, "translation": TEXTS[i % len(TEXTS)]}
        for i in range(16)
    ]
    train_records, val_records = records[:12], records[12:]

    train_ds = How2SignDataset(
        train_records, tokenizer,
        max_frames=MAX_FRAMES, img_size=IMG_SIZE,
        max_text_len=MAX_TEXT_LEN, extract_lm=False,
    )
    val_ds = How2SignDataset(
        val_records, tokenizer,
        max_frames=MAX_FRAMES, img_size=IMG_SIZE,
        max_text_len=MAX_TEXT_LEN, extract_lm=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=2, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    logger.info(f"      train={len(train_ds)} samples, val={len(val_ds)} samples")

    # ── 3. Model ──────────────────────────────────────────────────────────
    logger.info("\n[3/4] Building model...")
    model = SignLanguageTranslator(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.D_MODEL,
        freeze_backbone=True,
    )
    params = model.count_parameters()
    logger.info(f"      Parameters: {params}")
    model.to(device)

    # ── 4. Training + inference ───────────────────────────────────────────
    logger.info("\n[4/4] Running 3 training steps...")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4, weight_decay=0.01,
    )

    model.train()
    for step, batch in enumerate(train_loader):
        if step >= 3:
            break
        frames     = batch["frames"].to(device)
        heatmaps   = batch["heatmaps"].to(device)
        token_ids  = batch["token_ids"].to(device)
        frame_mask = batch["frame_mask"].to(device)

        tgt_input    = token_ids[:, :-1]
        tgt_target   = token_ids[:, 1:]
        tgt_pad_mask = (tgt_input == PAD_ID)

        logits = model(
            frames, tgt_input,
            heatmaps=heatmaps,
            frame_mask=frame_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.info(f"      step {step+1}/3  loss={loss.item():.4f}")

    # ── Inference ─────────────────────────────────────────────────────────
    logger.info("\n      Running greedy inference on val batch...")
    model.eval()
    val_batch  = next(iter(val_loader))
    frames     = val_batch["frames"].to(device)
    heatmaps   = val_batch["heatmaps"].to(device)
    frame_mask = val_batch["frame_mask"].to(device)

    preds = model.translate(
        frames, tokenizer,
        heatmaps=heatmaps, frame_mask=frame_mask, beam_size=1,
    )
    for i, (pred, ref) in enumerate(zip(preds, val_batch["text"])):
        logger.info(f"      [{i}] ref : {ref!r}")
        logger.info(f"           pred: {pred!r}")

    elapsed = time.time() - t0
    logger.info(f"\n=== Smoke test PASSED in {elapsed:.1f}s — full pipeline works! ===")

    # Cleanup temp tokenizer files
    for suffix in (".model", ".vocab"):
        p = TOKENIZER_PATH.with_suffix(suffix)
        if p.exists():
            p.unlink()


if __name__ == "__main__":
    main()
