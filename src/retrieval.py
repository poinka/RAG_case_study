import json
import pickle
import re
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", str(text).lower())


def build_dense_index(
    corpus_df: pd.DataFrame,
    model_name: str,
    index_path: Path,
    embeddings_path: Path,
    doc_ids_path: Path,
    batch_size: int = 64,
    normalize_embeddings: bool = True,
):
    model = SentenceTransformer(model_name)
    texts = corpus_df["text"].fillna("").tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) if normalize_embeddings else faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    np.save(embeddings_path, embeddings)
    doc_ids_path.write_text(json.dumps(corpus_df["doc_id"].tolist(), ensure_ascii=False), encoding="utf-8")

    return index, embeddings


def build_bm25_index(corpus_df: pd.DataFrame, bm25_path: Path):
    tokenized = [simple_tokenize(t) for t in corpus_df["text"].fillna("").tolist()]
    bm25 = BM25Okapi(tokenized)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)
    return bm25


class DenseRetriever:
    def __init__(self, index_path: Path, doc_ids_path: Path, corpus_df: pd.DataFrame, model_name: str):
        self.index = faiss.read_index(str(index_path))
        self.doc_ids = json.loads(doc_ids_path.read_text(encoding="utf-8"))
        self.corpus_df = corpus_df.set_index("doc_id", drop=False)
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, top_k: int = 5, exclude_doc_ids: Optional[set[str]] = None) -> list[dict]:
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        search_k = min(max(top_k * 5, top_k), len(self.doc_ids))
        scores, indices = self.index.search(q_emb, search_k)

        results = []
        exclude_doc_ids = exclude_doc_ids or set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc_id = self.doc_ids[int(idx)]
            if doc_id in exclude_doc_ids:
                continue
            row = self.corpus_df.loc[doc_id].to_dict()
            row["score"] = float(score)
            row["retriever"] = "dense"
            results.append(row)
            if len(results) >= top_k:
                break

        return results


class BM25Retriever:
    def __init__(self, bm25_path: Path, corpus_df: pd.DataFrame):
        with open(bm25_path, "rb") as f:
            payload = pickle.load(f)
        self.bm25 = payload["bm25"]
        self.corpus_df = corpus_df.reset_index(drop=True)

    def retrieve(self, query: str, top_k: int = 5, exclude_doc_ids: Optional[set[str]] = None) -> list[dict]:
        tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1]

        exclude_doc_ids = exclude_doc_ids or set()
        results = []
        for idx in order:
            row = self.corpus_df.iloc[int(idx)].to_dict()
            if row["doc_id"] in exclude_doc_ids:
                continue
            row["score"] = float(scores[int(idx)])
            row["retriever"] = "bm25"
            results.append(row)
            if len(results) >= top_k:
                break
        return results


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60, top_k: int = 20) -> list[dict]:
    scores = {}
    rows = {}

    for results in result_lists:
        for rank, row in enumerate(results, start=1):
            doc_id = row["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            rows[doc_id] = row

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    fused = []
    for doc_id in ranked_ids:
        row = dict(rows[doc_id])
        row["score"] = float(scores[doc_id])
        row["retriever"] = "rrf"
        fused.append(row)

    return fused


class HybridRerankRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        reranker_name: str,
        dense_k: int = 50,
        bm25_k: int = 50,
        rrf_k: int = 60,
        rerank_k: int = 20,
        device: Optional[str] = None,
    ):
        self.dense = dense
        self.bm25 = bm25
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.rrf_k = rrf_k
        self.rerank_k = rerank_k
        self.reranker = CrossEncoder(reranker_name, device=device)

    def retrieve(self, query: str, top_k: int = 5, exclude_doc_ids: Optional[set[str]] = None) -> list[dict]:
        dense_results = self.dense.retrieve(query, top_k=self.dense_k, exclude_doc_ids=exclude_doc_ids)
        bm25_results = self.bm25.retrieve(query, top_k=self.bm25_k, exclude_doc_ids=exclude_doc_ids)

        candidates = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            k=self.rrf_k,
            top_k=self.rerank_k,
        )

        pairs = [(query, row["text"]) for row in candidates]
        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)

        reranked = []
        for row, score in zip(candidates, rerank_scores):
            row = dict(row)
            row["rerank_score"] = float(score)
            row["retriever"] = "hybrid_rerank"
            reranked.append(row)

        reranked.sort(key=lambda r: r["rerank_score"], reverse=True)
        return reranked[:top_k]


def format_context(chunks: list[dict], max_chars_per_chunk: int = 1200) -> str:
    parts = []
    for i, ch in enumerate(chunks, start=1):
        text = str(ch.get("text", ""))[:max_chars_per_chunk]
        title = ch.get("title", "Untitled")
        parts.append(f"[{i}] Title: {title}\n{text}")
    return "\n\n".join(parts)


def retrieved_doc_ids(chunks: list[dict]) -> list[str]:
    return [str(ch["doc_id"]) for ch in chunks]
