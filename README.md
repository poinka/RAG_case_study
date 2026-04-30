# Case Study: Retrieval-Augmented Language Modeling for Factual Consistency

This repository contains a reproducible case study for comparing a parametric-only transformer against retrieval-augmented language modeling on HotpotQA.

## Research question

When does retrieval augmentation improve factual consistency over a standard parametric transformer of comparable size, and what are the latency and memory costs?

## Experimental systems

1. **Parametric-only LM**: the generator answers from the question only.
2. **Vanilla RAG**: dense FAISS retrieval over HotpotQA paragraphs + generator conditioned on top-k chunks.
3. **Advanced RAG**: hybrid BM25 + dense retrieval, Reciprocal Rank Fusion, cross-encoder reranking, and a retrieval-quality gate.
4. **Robustness variants**: fresh evidence, missing evidence, noisy context, and conflicting context.

## Recommended execution order

```bash
pip install -r requirements.txt
jupyter notebook
```

Run notebooks in this order:

1. `notebooks/01_data_preparation.ipynb`
2. `notebooks/02_build_retrieval_index.ipynb`
3. `notebooks/03_parametric_baseline.ipynb`
4. `notebooks/04_vanilla_rag.ipynb`
5. `notebooks/05_advanced_rag.ipynb`
6. `notebooks/06_robustness_latency_memory_analysis.ipynb`
7. `notebooks/07_poster_tables_and_figures.ipynb`

## Default models

- Generator: `Qwen/Qwen2.5-0.5B-Instruct`
- Dense encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Cross-encoder reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Dataset: `hotpotqa/hotpot_qa`, config `distractor`

## Outputs

- Processed data: `data/processed/`
- Retrieval indexes: `data/indexes/`
- Predictions: `results/predictions/`
- Metrics: `results/metrics/`
- Plots: `results/plots/`
- Poster outline: `poster/poster_outline.md`

## Notes

Start with `N_EXAMPLES=500` for a fast dry run. Increase to 1000-3000 examples for final experiments.
For model-generation notebooks, use a smaller `EVAL_N` first, such as 100-200, then scale if compute allows.
