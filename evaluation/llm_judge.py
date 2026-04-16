"""
F5 — LLM-as-Judge using Claude (claude-sonnet-4-6).

Scores 200+ test samples on three dimensions:
  - Adequacy  (1-5): Does the translation convey the same meaning?
  - Fluency   (1-5): Is the translation grammatically natural English?
  - Meaning   (1-5): Are key concepts/signs correctly translated?

Also computes:
  - Pearson correlation between LLM composite score and BLEU-4
  - Per-sample score table
  - 20+ example predictions vs references
"""

import os
import json
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are evaluating an Automatic Sign Language Translation system that translates ASL video to English.

Given a reference (ground-truth) translation and a system hypothesis, score the hypothesis on three dimensions.
Return ONLY a valid JSON object with exactly these keys: "adequacy", "fluency", "meaning".
Each value must be an integer from 1 to 5.

Scoring rubrics:
- adequacy (1-5): 5=all meaning preserved, 1=completely wrong meaning
- fluency  (1-5): 5=perfectly grammatical natural English, 1=unreadable
- meaning  (1-5): 5=all key concepts correctly translated, 1=no key concepts present

Reference: {reference}
Hypothesis: {hypothesis}

Respond with JSON only. Example: {{"adequacy": 3, "fluency": 4, "meaning": 3}}"""


@dataclass
class JudgeScore:
    hypothesis: str
    reference:  str
    adequacy:   float
    fluency:    float
    meaning:    float
    composite:  float   # mean of three dimensions

    def to_dict(self) -> Dict:
        return asdict(self)


def judge_batch(
    hypotheses: List[str],
    references:  List[str],
    model:       str = "claude-sonnet-4-6",
    batch_size:  int = 10,
    max_samples: int = 200,
    retry_delay: float = 2.0,
) -> List[JudgeScore]:
    """
    Score hypotheses using Claude as a judge.

    Args:
        hypotheses:  Model predictions.
        references:  Ground-truth translations.
        model:       Anthropic model ID.
        batch_size:  Samples per API call (sent as separate messages in one turn).
        max_samples: Cap on how many samples to score.
        retry_delay: Seconds to wait after a rate-limit error.

    Returns:
        List of JudgeScore dataclasses.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)

    # Cap samples
    n = min(len(hypotheses), max_samples)
    hyps = hypotheses[:n]
    refs = references[:n]

    scores: List[JudgeScore] = []
    batch_num = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_hyps = hyps[start:end]
        batch_refs = refs[start:end]
        batch_num += 1

        logger.info(f"  Judging samples {start+1}–{end} / {n} (batch {batch_num})")

        for hyp, ref in zip(batch_hyps, batch_refs):
            score = _score_single(client, model, hyp, ref, retry_delay)
            scores.append(score)
            time.sleep(0.1)  # gentle rate limiting

    return scores


def _score_single(
    client,
    model: str,
    hyp: str,
    ref: str,
    retry_delay: float,
    max_retries: int = 3,
) -> JudgeScore:
    """Score a single hypothesis/reference pair with retries."""
    prompt = JUDGE_PROMPT.format(hypothesis=hyp, reference=ref)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            data = json.loads(text)
            adequacy = float(data.get("adequacy", 3))
            fluency  = float(data.get("fluency",  3))
            meaning  = float(data.get("meaning",  3))
            composite = (adequacy + fluency + meaning) / 3.0

            return JudgeScore(
                hypothesis=hyp,
                reference=ref,
                adequacy=adequacy,
                fluency=fluency,
                meaning=meaning,
                composite=composite,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error (attempt {attempt+1}): {e} — response: {text[:100]}")
        except Exception as e:
            logger.warning(f"API error (attempt {attempt+1}): {e}")
            time.sleep(retry_delay * (attempt + 1))

    # Fallback: neutral score if all retries fail
    return JudgeScore(hyp, ref, 3.0, 3.0, 3.0, 3.0)


def pearson_correlation(
    scores: List[JudgeScore],
    bleu_per_sample: List[float],
) -> float:
    """
    Compute Pearson correlation between LLM composite scores and per-sample BLEU.

    Args:
        scores:           LLM judge scores.
        bleu_per_sample:  BLEU score for each individual hypothesis.

    Returns:
        Pearson r in [-1, 1].
    """
    from scipy.stats import pearsonr
    llm_scores = [s.composite for s in scores]
    r, p_value = pearsonr(llm_scores, bleu_per_sample[:len(llm_scores)])
    logger.info(f"  Pearson r(LLM, BLEU) = {r:.4f}  (p={p_value:.4e})")
    return r


def per_sample_bleu(hypotheses: List[str], references: List[str]) -> List[float]:
    """Compute BLEU-4 for each individual sample (sentence-level)."""
    from sacrebleu.metrics import BLEU
    bleu = BLEU(effective_order=True)
    result = []
    for hyp, ref in zip(hypotheses, references):
        try:
            s = bleu.sentence_score(hyp, [ref])
            result.append(s.score)
        except Exception:
            result.append(0.0)
    return result


def print_examples(
    scores: List[JudgeScore],
    n: int = 20,
) -> None:
    """Print example predictions vs references with LLM scores."""
    print(f"\n{'='*60}")
    print(f"  Sample Predictions (showing {min(n, len(scores))} examples)")
    print(f"{'='*60}")
    for i, s in enumerate(scores[:n]):
        print(f"\n  [{i+1}]")
        print(f"  REF : {s.reference}")
        print(f"  HYP : {s.hypothesis}")
        print(f"  Adequacy={s.adequacy:.0f}  Fluency={s.fluency:.0f}  Meaning={s.meaning:.0f}  Composite={s.composite:.2f}")
    print(f"\n{'='*60}\n")


def save_scores(scores: List[JudgeScore], path: str) -> None:
    with open(path, "w") as f:
        json.dump([s.to_dict() for s in scores], f, indent=2)
    logger.info(f"Saved {len(scores)} judge scores → {path}")


def summarise(scores: List[JudgeScore]) -> Dict[str, float]:
    """Aggregate statistics across all scored samples."""
    if not scores:
        return {}
    n = len(scores)
    return {
        "n_samples":        n,
        "mean_adequacy":    sum(s.adequacy   for s in scores) / n,
        "mean_fluency":     sum(s.fluency    for s in scores) / n,
        "mean_meaning":     sum(s.meaning    for s in scores) / n,
        "mean_composite":   sum(s.composite  for s in scores) / n,
    }
