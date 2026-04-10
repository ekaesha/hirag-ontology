# HiRAG-Ontology

Multi-agent pipeline for automatic ontology construction and knowledge graph 
improvement from unstructured text, with hybrid RRF-based retrieval for 
GraphRAG quality enhancement.

Bachelor's Thesis — HSE DSBA, 2025  
Authors: Eva Karimova, Alexey Popov

## Results

| System | Overall |
|--------|---------|
| Naive RAG | 4.85 |
| HiRAG (baseline) | 5.40 |
| HiRAG + Dedup | 4.45 |
| **HiRAG-Ontology (Ours)** | **5.60** |

Evaluated on 10 oncological clinical guidelines from the Russian Ministry 
of Health (Minzdrav dataset, 78 documents total).

## System Architecture

Six specialised agents (A1–A6):
- **A1 Extraction** — LLM-based triplet extraction from text chunks
- **A2 Typing** — ontological class assignment (τ: V → C)
- **A3 Deduplication** — hybrid lexical+semantic entity merging (α=0.6, θ=0.85)
- **A4 Validation** — rule-based consistency checking (Cons(G) = 0.773)
- **A5 Reasoning** — missing relation inference (T_missing = T* − T̂)
- **A6 Update** — PageRank computation and graph persistence

Hybrid retriever: BM25 + BERT embeddings + PageRank → RRF fusion (k=60)

## Installation

```bash
pip install -r requirements.txt
```

Create `.env` file: