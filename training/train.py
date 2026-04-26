"""
F4 -- Training loop for SignLanguageTranslator.

Features:
  - CrossEntropyLoss with ignore_index=PAD + label_smoothing=0.1
  - AdamW optimizer (lr=3e-4, weight_decay=0.01)
  - Gradient accumulation (effective batch = MICRO_BATCH ? GRAD_ACCUM_STEPS = 32)
  - Gradient clipping (max_norm=1.0)
  - CosineAnnealingLR scheduler
  - Mixed precision (fp16 AMP) via torch.cuda.amp
  - WandB logging (loss, BLEU, LR, GPU memory)
  - Two-phase training: Phase 1 frozen backbone (15 ep), Phase 2 unfrozen (15 ep)
  - Checkpoint every 5 epochs + best model
"""

# av must be imported before torch on Windows to establish correct FFmpeg DLL
# load order; reversing this causes a segfault when datasets streams video.
import av  # noqa: F401

import os
import logging
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import (
    D_MODEL, MICRO_BATCH_SIZE, GRAD_ACCUM_STEPS, GRAD_CLIP_NORM,
    LABEL_SMOOTHING, WEIGHT_DECAY, PHASE1_EPOCHS, PHASE1_LR,
    PHASE2_EPOCHS, PHASE2_LR, TOTAL_EPOCHS, CHECKPOINT_EVERY,
    CHECKPOINT_DIR, WANDB_PROJECT, WANDB_ENTITY, USE_AMP,
    PAD_ID, BEAM_SIZE,
)

logger = logging.getLogger(__name__)


# ??? BLEU utility (inline, no sacrebleu dependency at import time) ??????????

def _quick_bleu(hypotheses: List[str], references: List[str]) -> float:
    """Fast corpus BLEU-4 using sacrebleu (returns 0 if not available)."""
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU(effective_order=True)
        result = bleu.corpus_score(hypotheses, [references])
        return result.score
    except Exception:
        return 0.0


# ??? Trainer ????????????????????????????????????????????????????????????????

class Trainer:
    """
    Encapsulates the full training lifecycle for SignLanguageTranslator.

    Usage:
        trainer = Trainer(model, tokenizer, train_loader, val_loader)
        trainer.train()
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        use_wandb: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # ?? WandB ?????????????????????????????????????????????????????????
        self.use_wandb = use_wandb and _try_wandb_init()

        # ?? Scaler for AMP ????????????????????????????????????????????????
        self.scaler = GradScaler(enabled=USE_AMP and self.device.type == "cuda")

        # ?? Checkpointing ?????????????????????????????????????????????????
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.best_bleu = 0.0
        self.global_step = 0

    # ?? Public entry point ????????????????????????????????????????????????

    def train(self):
        """Run full two-phase training."""
        logger.info("=== Phase 1: backbone frozen ===")
        self.model.set_phase(1)
        self._run_phase(
            phase=1,
            n_epochs=PHASE1_EPOCHS,
            lr=PHASE1_LR,
        )

        logger.info("=== Phase 2: backbone unfrozen ===")
        self.model.set_phase(2)
        self._run_phase(
            phase=2,
            n_epochs=PHASE2_EPOCHS,
            lr=PHASE2_LR,
            start_epoch=PHASE1_EPOCHS,
        )

    # ?? Phase runner ?????????????????????????????????????????????????????

    def _run_phase(self, phase: int, n_epochs: int, lr: float, start_epoch: int = 0):
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.1)
        criterion = nn.CrossEntropyLoss(
            ignore_index=PAD_ID,
            label_smoothing=LABEL_SMOOTHING,
        )

        for epoch in range(start_epoch, start_epoch + n_epochs):
            train_loss = self._train_epoch(epoch, optimizer, criterion)
            scheduler.step()

            val_loss, bleu = 0.0, 0.0
            if self.val_loader is not None:
                val_loss, bleu = self._val_epoch(epoch)

            logger.info(
                f"Epoch {epoch+1}/{TOTAL_EPOCHS}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  BLEU={bleu:.2f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

            if self.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/bleu4": bleu,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/phase": phase,
                }, step=self.global_step)

            if (epoch + 1) % CHECKPOINT_EVERY == 0:
                self._save_checkpoint(epoch, optimizer, train_loss, tag=f"epoch{epoch+1}")

            if bleu > self.best_bleu:
                self.best_bleu = bleu
                self._save_checkpoint(epoch, optimizer, train_loss, tag="best")
                logger.info(f"  New best BLEU: {bleu:.2f}")

    # ?? Single train epoch ????????????????????????????????????????????????

    def _train_epoch(self, epoch: int, optimizer, criterion) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0
        optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            frames     = batch["frames"].to(self.device)       # (B, T, 3, H, W)
            heatmaps   = batch["heatmaps"].to(self.device)     # (B, T, H, W)
            token_ids  = batch["token_ids"].to(self.device)    # (B, S)
            frame_mask = batch["frame_mask"].to(self.device)   # (B, T)

            # Teacher forcing: input = tgt[:, :-1], target = tgt[:, 1:]
            tgt_input  = token_ids[:, :-1]    # (B, S-1)
            tgt_target = token_ids[:, 1:]     # (B, S-1)

            tgt_pad_mask = (tgt_input == PAD_ID)   # (B, S-1) True = PAD

            with autocast(enabled=USE_AMP and self.device.type == "cuda"):
                logits = self.model(
                    frames, tgt_input,
                    heatmaps=heatmaps,
                    frame_mask=frame_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                )                                              # (B, S-1, V)

                # Flatten for cross-entropy
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_target.reshape(-1),
                )
                loss = loss / GRAD_ACCUM_STEPS

            self.scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            n_batches  += 1

            if step % 100 == 0:
                logger.info(f"  Epoch {epoch+1}  step {step}  loss={loss.item()*GRAD_ACCUM_STEPS:.4f}")

        return total_loss / max(n_batches, 1)

    # ?? Validation epoch ??????????????????????????????????????????????????

    def _val_epoch(self, epoch: int):
        self.model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.0)
        total_loss = 0.0
        n_batches  = 0
        hypotheses, references = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                frames     = batch["frames"].to(self.device)
                heatmaps   = batch["heatmaps"].to(self.device)
                token_ids  = batch["token_ids"].to(self.device)
                frame_mask = batch["frame_mask"].to(self.device)

                tgt_input  = token_ids[:, :-1]
                tgt_target = token_ids[:, 1:]
                tgt_pad_mask = (tgt_input == PAD_ID)

                with autocast(enabled=USE_AMP and self.device.type == "cuda"):
                    logits = self.model(
                        frames, tgt_input,
                        heatmaps=heatmaps,
                        frame_mask=frame_mask,
                        tgt_key_padding_mask=tgt_pad_mask,
                    )
                    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_target.reshape(-1))

                total_loss += loss.item()
                n_batches  += 1

                # Collect predictions for BLEU
                preds = self.model.translate(
                    frames, self.tokenizer,
                    heatmaps=heatmaps, frame_mask=frame_mask, beam_size=1
                )
                hypotheses.extend(preds)
                references.extend(batch["text"])

        val_loss = total_loss / max(n_batches, 1)
        bleu = _quick_bleu(hypotheses, references)
        return val_loss, bleu

    # ?? Checkpoint ???????????????????????????????????????????????????????

    def _save_checkpoint(self, epoch: int, optimizer, loss: float, tag: str = ""):
        path = CHECKPOINT_DIR / f"checkpoint_{tag}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "best_bleu": self.best_bleu,
        }, path)
        logger.info(f"  Saved checkpoint -> {path}")

    def load_checkpoint(self, path: str | Path, optimizer=None):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_bleu = ckpt.get("best_bleu", 0.0)
        logger.info(f"Loaded checkpoint from {path}  (epoch={ckpt['epoch']+1})")
        return ckpt["epoch"]


# ??? WandB helper ????????????????????????????????????????????????????????????

def _try_wandb_init() -> bool:
    """Initialise WandB. Returns False if unavailable or no API key."""
    try:
        import wandb
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config={
                "d_model": D_MODEL,
                "batch_size": MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS,
                "phase1_epochs": PHASE1_EPOCHS,
                "phase2_epochs": PHASE2_EPOCHS,
                "phase1_lr": PHASE1_LR,
                "phase2_lr": PHASE2_LR,
            },
        )
        return True
    except Exception as e:
        logger.warning(f"WandB not available: {e}")
        return False


# ??? CLI entry point ?????????????????????????????????????????????????????????

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train Sign Language Translator")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--overfit-test", action="store_true",
                        help="Quick overfit test on 100 samples (sanity check)")
    parser.add_argument("--no-landmarks", action="store_true",
                        help="Skip MediaPipe landmark extraction (much faster; "
                             "spatial attention uses zero heatmaps)")
    args = parser.parse_args()
    # Landmark extraction runs MediaPipe on every frame at load time -- too slow
    # for on-the-fly streaming.  Default to off; enable with a preprocess step.
    extract_lm = not args.no_landmarks

    from data.tokenizer import SignTokenizer
    from data.download import load_dataset_split
    from data.dataset import How2SignDataset, How2SignStreamingDataset, collate_fn
    from models.translator import SignLanguageTranslator

    logger.info("Loading tokenizer...")
    tokenizer = SignTokenizer()
    tok_model_path = CHECKPOINT_DIR.parent / "data" / "tokenizer.model"
    if not tok_model_path.exists():
        logger.info("Training tokenizer on 3,000 streaming samples...")
        ds_tok = load_dataset_split("train", streaming=True)
        texts = [ex["translation"] for ex in ds_tok.take(3_000)]
        tokenizer.train(texts)
    else:
        from training.config import TOKENIZER_MODEL
        tokenizer.load(TOKENIZER_MODEL)

    logger.info("Building model...")
    model = SignLanguageTranslator(vocab_size=tokenizer.vocab_size)
    params = model.count_parameters()
    logger.info(f"  Parameters: {params}")

    logger.info("Loading dataset...")
    if args.overfit_test:
        # Stream just the first N clips -- no full dataset download needed.
        train_records = list(load_dataset_split("train", streaming=True).take(100))
        val_records   = list(load_dataset_split("val",   streaming=True).take(20))
        train_ds = How2SignDataset(train_records, tokenizer, extract_lm=extract_lm)
        val_ds   = How2SignDataset(val_records,   tokenizer, extract_lm=extract_lm)
    else:
        # Full training: stream lazily to avoid loading 75 GB of video bytes into RAM.
        # Validation split (~1.7 K clips) is small enough to materialise.
        train_hf = load_dataset_split("train", streaming=True)
        logger.info("Materialising validation split (streaming)...")
        val_records = list(load_dataset_split("val", streaming=True))
        train_ds = How2SignStreamingDataset(train_hf, tokenizer, extract_lm=extract_lm)
        val_ds   = How2SignDataset(val_records, tokenizer, extract_lm=extract_lm)

    # num_workers=0: Windows spawns new processes that would re-init DLLs in
    # wrong order (av after torch). Inline loading is simpler and safe.
    # IterableDataset does not support DataLoader shuffle=True; buffer shuffle
    # is handled inside How2SignStreamingDataset.
    from training.config import MICRO_BATCH_SIZE
    train_loader = DataLoader(
        train_ds, batch_size=MICRO_BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=MICRO_BATCH_SIZE * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    trainer = Trainer(model, tokenizer, train_loader, val_loader, use_wandb=not args.no_wandb)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()
