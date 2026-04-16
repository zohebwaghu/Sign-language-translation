"""
F5 — Automatic evaluation metrics for sign language translation.

Metrics:
  BLEU-4   — corpus-level n-gram precision (sacrebleu, smoothing method 1)
  ROUGE-L  — longest common subsequence recall (rouge-score)
  METEOR   — synonym-aware precision/recall (nltk)
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def compute_bleu(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute corpus BLEU-4 with smoothing method 1.

    Args:
        hypotheses: List of model predictions.
        references: List of ground-truth translations (same length).

    Returns:
        dict with keys: bleu4, bleu1, bleu2, bleu3, bp, ratio
    """
    from sacrebleu.metrics import BLEU

    bleu = BLEU(smooth_method="exp", smooth_value=1.0, effective_order=False)
    result = bleu.corpus_score(hypotheses, [references])

    return {
        "bleu4":  result.score,
        "bleu1":  result.precisions[0],
        "bleu2":  result.precisions[1],
        "bleu3":  result.precisions[2],
        "bp":     result.bp,
        "ratio":  result.sys_len / max(result.ref_len, 1),
    }


def compute_rouge(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-L (F1 over LCS) for each pair, then average.

    Returns:
        dict with keys: rougeL_f1, rougeL_precision, rougeL_recall
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    f1s, precs, recalls = [], [], []
    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)["rougeL"]
        f1s.append(scores.fmeasure)
        precs.append(scores.precision)
        recalls.append(scores.recall)

    return {
        "rougeL_f1":        sum(f1s)    / max(len(f1s), 1),
        "rougeL_precision": sum(precs)  / max(len(precs), 1),
        "rougeL_recall":    sum(recalls)/ max(len(recalls), 1),
    }


def compute_meteor(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute METEOR (synonym-aware precision/recall) using NLTK.

    Requires: nltk.download('wordnet') and nltk.download('omw-1.4')

    Returns:
        dict with key: meteor
    """
    import nltk

    # Auto-download required resources if missing
    for resource in ["wordnet", "omw-1.4", "punkt_tab"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        ref_tokens = nltk.word_tokenize(ref.lower())
        score = nltk.translate.meteor_score.single_meteor_score(ref_tokens, hyp_tokens)
        scores.append(score)

    return {"meteor": sum(scores) / max(len(scores), 1)}


def compute_all(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute BLEU-4, ROUGE-L, and METEOR in one call.

    Returns:
        Merged dict of all metric values.
    """
    results = {}
    try:
        results.update(compute_bleu(hypotheses, references))
    except Exception as e:
        logger.warning(f"BLEU failed: {e}")

    try:
        results.update(compute_rouge(hypotheses, references))
    except Exception as e:
        logger.warning(f"ROUGE-L failed: {e}")

    try:
        results.update(compute_meteor(hypotheses, references))
    except Exception as e:
        logger.warning(f"METEOR failed: {e}")

    return results


def print_results(metrics: Dict[str, float], n_samples: int = 0) -> None:
    """Pretty-print a metrics dict."""
    print("\n" + "=" * 50)
    print(f"  Evaluation Results  (n={n_samples})")
    print("=" * 50)
    order = ["bleu4", "bleu1", "bleu2", "bleu3", "rougeL_f1", "meteor", "bp"]
    for key in order:
        if key in metrics:
            print(f"  {key:<20}: {metrics[key]:.4f}")
    for key in sorted(metrics):
        if key not in order:
            print(f"  {key:<20}: {metrics[key]:.4f}")
    print("=" * 50 + "\n")
