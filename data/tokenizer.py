"""
SentencePiece BPE tokenizer for sign language translation.

Replaces the word-level tokenizer from the PRD with a BPE tokenizer
(~8K subwords) to avoid the 16K+ sparse vocabulary problem.

Special token IDs (must match training/config.py):
  PAD = 0  BOS = 1  EOS = 2  UNK = 3
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

try:
    import sentencepiece as spm
    _SP_AVAILABLE = True
except ImportError:
    _SP_AVAILABLE = False
    logger.warning("sentencepiece not installed — tokenizer will raise on train/encode")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID, UNK_ID, TOKENIZER_MODEL


class SignTokenizer:
    """
    Thin wrapper around SentencePiece with fixed special token layout.

    Special token IDs are enforced at training time via the
    `--control_symbols` flag so they are always 0-3 regardless of
    the subword vocabulary content.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self._sp: Optional["spm.SentencePieceProcessor"] = None
        if model_path is not None:
            self.load(model_path)

    # ─── Build / Save / Load ─────────────────────────────────────────────────

    def train(
        self,
        texts: List[str],
        model_path: Union[str, Path] = TOKENIZER_MODEL,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        """
        Train a BPE tokenizer on a list of sentences.

        Args:
            texts:      Training sentences (translation side of the dataset).
            model_path: Where to save the .model file.
            vocab_size: Subword vocabulary size.
        """
        if not _SP_AVAILABLE:
            raise ImportError("sentencepiece is required — pip install sentencepiece")

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Write corpus to a temp file
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(texts))
            corpus_file = f.name

        model_prefix = str(model_path).removesuffix(".model")

        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            pad_id=PAD_ID,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            unk_id=UNK_ID,
            pad_piece="<pad>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            unk_piece="<unk>",
            character_coverage=0.9999,
            num_threads=os.cpu_count(),
            input_sentence_size=5_000_000,
            shuffle_input_sentence=True,
        )

        os.unlink(corpus_file)
        self.load(model_path)
        logger.info(f"Tokenizer trained → {model_path}  (vocab={self.vocab_size})")

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The .model file is already on disk after train(); this copies it.
        import shutil
        src = Path(str(TOKENIZER_MODEL))
        if src != path:
            shutil.copy(src, path)

    def load(self, path: Union[str, Path]) -> None:
        if not _SP_AVAILABLE:
            raise ImportError("sentencepiece is required — pip install sentencepiece")
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(path))
        logger.info(f"Tokenizer loaded from {path}  (vocab={self.vocab_size})")

    # ─── Encoding / Decoding ─────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_len: Optional[int] = None,
    ) -> List[int]:
        """
        Encode a string to token IDs.

        Returns:
            List[int] including BOS/EOS if requested, truncated to max_len.
        """
        self._check_loaded()
        ids: List[int] = self._sp.encode(text, out_type=int)
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to a string.

        Args:
            ids:           Token ID list (may include BOS/EOS/PAD).
            skip_special:  Strip PAD, BOS, EOS tokens before decoding.
        """
        self._check_loaded()
        special = {PAD_ID, BOS_ID, EOS_ID}
        if skip_special:
            ids = [i for i in ids if i not in special]
        return self._sp.decode(ids)

    def encode_batch(
        self,
        texts: List[str],
        max_len: int = 64,
        pad: bool = True,
    ):
        """
        Encode a list of strings, returning a padded LongTensor.

        Returns:
            token_ids: (B, max_len) torch.LongTensor
            mask:      (B, max_len) bool tensor — True for real tokens
        """
        import torch
        encoded = [self.encode(t, add_bos=True, add_eos=True, max_len=max_len) for t in texts]
        if not pad:
            return encoded

        lengths = [len(e) for e in encoded]
        T = max_len  # always pad to fixed length so collate_fn can stack tensors

        token_ids = torch.full((len(texts), T), PAD_ID, dtype=torch.long)
        mask = torch.zeros(len(texts), T, dtype=torch.bool)

        for i, (ids, length) in enumerate(zip(encoded, lengths)):
            L = min(length, T)
            token_ids[i, :L] = torch.tensor(ids[:L], dtype=torch.long)
            mask[i, :L] = True

        return token_ids, mask

    # ─── Properties ──────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        self._check_loaded()
        return self._sp.get_piece_size()

    def _check_loaded(self) -> None:
        if self._sp is None:
            raise RuntimeError(
                "Tokenizer not loaded. Call .train() or .load(path) first."
            )
