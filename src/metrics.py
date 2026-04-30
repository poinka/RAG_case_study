import collections
import math
import re
import string
from typing import Iterable, List, Dict, Any

import numpy as np


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    if s is None:
        return ""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_answer(prediction: str, ground_truth: str) -> float:
    pred = normalize_answer(prediction)
    gold = normalize_answer(ground_truth)
    if not gold:
        return 0.0
    return float(gold in pred)


def evaluate_qa_predictions(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Evaluate rows with fields prediction and answer."""
    ems, f1s, contains = [], [], []
    for row in rows:
        pred = row.get("prediction", "")
        gold = row.get("answer", "")
        ems.append(exact_match_score(pred, gold))
        f1s.append(f1_score(pred, gold))
        contains.append(contains_answer(pred, gold))

    return {
        "exact_match": float(np.mean(ems)) if ems else math.nan,
        "token_f1": float(np.mean(f1s)) if f1s else math.nan,
        "contains_answer": float(np.mean(contains)) if contains else math.nan,
        "n": len(ems),
    }


def hit_at_k(retrieved_doc_ids: List[str], gold_doc_ids: List[str], k: int) -> float:
    if not gold_doc_ids:
        return 0.0
    retrieved = set(retrieved_doc_ids[:k])
    gold = set(gold_doc_ids)
    return float(len(retrieved & gold) > 0)


def support_recall_at_k(retrieved_doc_ids: List[str], gold_doc_ids: List[str], k: int) -> float:
    """Fraction of gold supporting docs covered in top-k."""
    if not gold_doc_ids:
        return 0.0
    retrieved = set(retrieved_doc_ids[:k])
    gold = set(gold_doc_ids)
    return len(retrieved & gold) / len(gold)


def safe_mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(values)) if values else math.nan

def numeric_summary(values, prefix: str) -> Dict[str, float]:
    """
    Robust summary for numeric metrics such as perplexity and latency.

    For PPL, mean can be dominated by outliers, so median/p90/p95 are important.
    """
    arr = np.array(
        [
            float(v)
            for v in values
            if v is not None
            and not (isinstance(v, float) and math.isnan(v))
            and not (isinstance(v, float) and math.isinf(v))
        ],
        dtype=float,
    )

    if arr.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_p90": math.nan,
            f"{prefix}_p95": math.nan,
            f"{prefix}_max": math.nan,
        }

    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
        f"{prefix}_p95": float(np.quantile(arr, 0.95)),
        f"{prefix}_max": float(np.max(arr)),
    }