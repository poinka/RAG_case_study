import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd


def extract_context(example: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    """
    Return list of (title, sentences) from a HotpotQA example.

    Hugging Face format is usually:
    context = {"title": [...], "sentences": [[...], ...]}
    """
    context = example.get("context", {})
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        return [(str(t), list(s)) for t, s in zip(titles, sentences)]

    return []


def extract_supporting_facts(example: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    Return mapping: title -> list of supporting sentence ids.

    Hugging Face format is usually:
    supporting_facts = {"title": [...], "sent_id": [...]}
    """
    sf = example.get("supporting_facts", {})
    result = {}

    if isinstance(sf, dict):
        titles = sf.get("title", [])
        sent_ids = sf.get("sent_id", [])
        for title, sid in zip(titles, sent_ids):
            result.setdefault(str(title), []).append(int(sid))
        return result
    
    return result


def build_tables(dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert HotpotQA examples into:
    - questions table
    - paragraph-level corpus table

    Each paragraph is one retrieval document.
    """
    question_rows = []
    corpus_rows = []

    for example_idx, ex in enumerate(dataset):
        sample_id = str(ex.get("id", ex.get("_id", example_idx)))
        question = str(ex.get("question", ""))
        answer = str(ex.get("answer", ""))
        q_type = str(ex.get("type", ""))
        level = str(ex.get("level", ""))

        support = extract_supporting_facts(ex)
        support_titles = set(support.keys())

        question_rows.append({
            "sample_id": sample_id,
            "question": question,
            "answer": answer,
            "type": q_type,
            "level": level,
            "support_titles": json.dumps(sorted(list(support_titles)), ensure_ascii=False),
        })

        for paragraph_idx, (title, sentences) in enumerate(extract_context(ex)):
            text = " ".join(str(s) for s in sentences)
            support_sentence_ids = support.get(title, [])
            doc_id = f"{sample_id}::{paragraph_idx}::{title}"

            corpus_rows.append({
                "doc_id": doc_id,
                "sample_id": sample_id,
                "title": title,
                "paragraph_idx": paragraph_idx,
                "text": text,
                "sentences_json": json.dumps(sentences, ensure_ascii=False),
                "support_sentence_ids": json.dumps(support_sentence_ids),
                "is_supporting_doc": bool(title in support_titles),
            })

    questions_df = pd.DataFrame(question_rows)
    corpus_df = pd.DataFrame(corpus_rows)

    support_doc_map = (
        corpus_df[corpus_df["is_supporting_doc"]]
        .groupby("sample_id")["doc_id"]
        .apply(list)
        .to_dict()
    )
    questions_df["gold_doc_ids"] = questions_df["sample_id"].map(
        lambda sid: json.dumps(support_doc_map.get(sid, []), ensure_ascii=False)
    )

    return questions_df, corpus_df


def load_gold_doc_ids(value) -> list[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
