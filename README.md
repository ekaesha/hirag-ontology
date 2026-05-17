# HiRAG-Ontology

Multi-agent pipeline for automatic ontology construction and knowledge graph
improvement from unstructured text, with hybrid RRF-based retrieval for
GraphRAG quality enhancement.

**Bachelor's Thesis — HSE DSBA, 2026**
**Author: Eva Karimova**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ekaesha/hirag-ontology/blob/main/hirag_ontology_colab.ipynb)

---

## Individual Contribution (Eva Karimova)

| Component | Description |
|-----------|-------------|
| **Chapter 3: Formal Model** | Complete mathematical framework: G=(V,E), O=(C,R,A), Q(G), RRF formulation |
| **Typing Agent (A2)** | LLMs4OL Task A paradigm, MAP@1 = 0.945 on 200-entity sample |
| **Deduplication Agent (A3)** | Hybrid sim = α·sem + (1-α)·lex, Union-Find, 413 entities merged (15.1%) |
| **Validation Agent (A4)** | 5 OWL 2 axiom checkers, auto-repair, Cons(G) = 0.773 |
| **Reasoning Agent (A5)** | Knowledge gap detection: T_missing = T* − T̂ |
| **Hybrid Retriever** | BM25 + BERT + PageRank → RRF (k=60), Hit@10=0.78, MRR=0.57 |
| **Experimental Evaluation** | Benchmark design, ablation, latency, dedup grid search, RAGAS metrics |

---

## Results

Evaluated on 78 oncological clinical guidelines from the Russian Ministry
of Health (Minzdrav). 50-question domain-specific benchmark.
LLM-as-judge: DeepSeek-Chat · Scale: 0–10.

### Generation Quality (LLM-as-Judge)

| System | Comp | Emp | Div | Overall |
|--------|------|-----|-----|---------|
| Naive RAG | 5.21 | 5.10 | 5.42 | 5.34 |
| HiRAG (baseline) | 6.15 | 6.05 | 6.20 | 6.28 |
| HiRAG + Dedup | 5.00 | 4.95 | 5.35 | 5.10 |
| **HiRAG-Ontology (Ours)** | **6.55** | **6.45** | **6.90** | **6.68** |

**+25.1% over Naive RAG · +6.4% over HiRAG baseline**

### Intrinsic Retrieval Metrics

| System | Hit@5 | Hit@10 | MRR | MAP@10 |
|--------|-------|--------|-----|--------|
| Semantic-only (BGE-M3) | 0.52 | 0.64 | 0.41 | 0.38 |
| BM25 Lexical-only | 0.44 | 0.58 | 0.35 | 0.31 |
| Structural-only (PageRank) | 0.36 | 0.48 | 0.28 | 0.24 |
| **Hybrid RRF (proposed)** | **0.68** | **0.78** | **0.57** | **0.52** |

### RAGAS-Aligned Generation Metrics

| Metric | Score |
|--------|-------|
| Faithfulness | 0.86 |
| Answer Relevance | 0.81 |
| Context Precision | 0.72 |
| Context Recall | 0.67 |

### Ablation Study

| Configuration | Overall | vs Full System |
|---------------|---------|----------------|
| Baseline (semantic only, no dedup) | 5.20 | −26.8% |
| w/o Hybrid Retriever | 5.40 | −23.9% |
| w/o Deduplication | 7.20 | +1.4% |
| **Full System (Ours)** | **7.10** | — |

### System Latency (TTFT)

| Stage | Mean (s) | % of Total |
|-------|----------|------------|
| Hybrid RRF retrieval (E. Karimova) | 0.18 | 2.1% |
| Context formatting | 0.03 | 0.4% |
| HiRAG community summarisation | 1.40 | 16.3% |
| LLM answer generation | 6.90 | 80.2% |
| **End-to-end (TTFT)** | **8.61** | 100% |

---

## Knowledge Graph Statistics

| Metric | Value |
|--------|-------|
| Documents processed | 78 (full Minzdrav corpus) |
| Entities (raw) | 2,727 |
| Entities (after deduplication) | 2,314 |
| Relations | 2,346 |
| Deduplication rate | 15.1% (413 entities merged) |
| Consistency score Cons(G) | 0.773 |
| Quality functional Q(G) | 0.730 |
| Typing accuracy MAP@1 | 0.945 |

---

## System Architecture

Six specialised agents (A1–A6):

| Agent | Author | Function | Formal |
|-------|--------|----------|--------|
| A1 Extraction | A. Popov | LLM triplet extraction | f_θ: C → 2^T |
| A2 Typing | **E. Karimova** | Ontological class assignment | τ: V → C |
| A3 Deduplication | **E. Karimova** | Hybrid lexical+semantic merging | π: V → V_canon |
| A4 Validation | **E. Karimova** | Rule-based consistency checking | Cons(G) = 1 − |violations|/|A| |
| A5 Reasoning | **E. Karimova** | Missing relation inference | T_missing = T* − T̂ |
| A6 Update | A. Popov | PageRank + graph persistence | G_T = (A_m ∘ ... ∘ A_1)(G_0) |

**Hybrid retriever (E. Karimova):** BM25 + BERT embeddings + PageRank → RRF (k=60)

**Quality functional:** Q(G) = λ1·Coverage + λ2·Consistency + λ3·Precision − λ4·Redundancy

**Ontology:** `ontology.json` — 9 classes, 7 predicates, 5 axiom types

---

## Project Structure

```
hirag-ontology/
├── ontology.json                        # Domain ontology O = (C, P, A)
├── pipeline/
│   ├── ontology_loader.py               # Dynamic ontology loading
│   ├── knowledge_graph.py               # Core KG data structure
│   ├── extractor.py                     # A1 — triplet extraction (A. Popov)
│   ├── typing_agent.py                  # A2 — ontological typing (E. Karimova)
│   ├── deduplication.py                 # A3 — hybrid deduplication (E. Karimova)
│   ├── validator.py                     # A4+A5 — validation + reasoning (E. Karimova)
│   └── quality.py                       # Q(G) quality functional
├── retrieval/
│   └── retriever.py                     # Hybrid RRF retriever (E. Karimova)
├── evaluation/
│   ├── ground_truth.json                # 50-question annotations (E. Karimova)
│   ├── judge.py                         # LLM-as-judge
│   ├── run_eval.py                      # Original main experiment runner
│   ├── retrieval_eval.py                # Hit@k, MRR, MAP@k (E. Karimova) ← NEW
│   ├── generation_eval.py               # RAGAS-style metrics (E. Karimova)  ← NEW
│   ├── latency_eval.py                  # TTFT latency benchmark (E. Karimova) ← NEW
│   ├── dedup_ablation.py                # α × θ grid search (E. Karimova)    ← NEW
│   └── run_full_eval.py                 # Master evaluation runner            ← NEW
├── results/                             # Output directory (auto-created)
│   ├── knowledge_graph_final.json       # Built KG
│   ├── retrieval_metrics.json           # Hit@k, MRR results
│   ├── generation_metrics.json          # RAGAS metrics
│   ├── latency_results.json             # TTFT breakdown
│   ├── dedup_ablation.json              # Grid search results
│   └── full_evaluation_report.json      # Consolidated report
├── iterative_pipeline.py                # Iterative improvement loop
├── langchain_integration.py             # LangChain wrapper
├── web_demo.py                          # Browser demo (localhost:5000)
├── graph_explorer.ipynb                 # Jupyter Notebook
├── hirag_ontology_colab.ipynb           # Google Colab notebook
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt
```

Create `.env` in the repo root:

```
DEEPSEEK_API_KEY=your_key_here
```

### Run full pipeline (original)

```bash
python -m evaluation.run_eval
```

### Run ALL evaluations (retrieval + generation + latency + dedup ablation)

```bash
python -m evaluation.run_full_eval
```

This produces the complete set of metrics used in the thesis.

### Run individual evaluation components

```bash
# Intrinsic retrieval metrics (Hit@k, MRR, MAP@k) — no API key needed
python -m evaluation.retrieval_eval

# RAGAS-style generation metrics (requires DEEPSEEK_API_KEY)
python -m evaluation.generation_eval --n 50

# Latency benchmark (20 queries)
python -m evaluation.latency_eval --n 20

# Deduplication hyperparameter grid search — no API key needed
python -m evaluation.dedup_ablation
```

### Run iterative improvement loop

```bash
python iterative_pipeline.py
```

### Web demo

```bash
python web_demo.py
# Open http://localhost:5000
```

### LangChain integration

```python
from langchain_integration import build_langchain_rag_chain
chain = build_langchain_rag_chain("results/knowledge_graph_final.json")
answer = chain.invoke("What is the treatment protocol for ALL?")
```

---

## Evaluation Details

### ground_truth.json

The file `evaluation/ground_truth.json` contains manually annotated ground-truth
entity labels for all 50 benchmark questions. For each question, the field
`relevant_entity_labels` lists the canonical entity labels that must appear
in the top-K retrieved set for the answer to be correct.

Annotated by E. Karimova based on manual inspection of the Minzdrav corpus.
Average 2.3 ground-truth entities per question (range: 1–6).

### Deduplication Hyperparameter Selection

The mixing coefficient α = 0.6 and threshold θ = 0.85 were selected by
grid search over α ∈ {0.4, 0.5, 0.6, 0.7, 0.8} and θ ∈ {0.75, 0.80, 0.85, 0.90, 0.95},
evaluated on a manually annotated sample of 30 entity pairs (15 true duplicates,
15 true non-duplicates). Full results: `results/dedup_ablation.json`.

---

## Dataset

78 clinical guidelines from the Russian Ministry of Health (Minzdrav),
covering oncological conditions in Russian Markdown format.
Corpus size: 108 KB – 566 KB per document.

---

## Models

| Component | Model |
|-----------|-------|
| Extraction, Typing, Generation, Judge | DeepSeek-Chat |
| Embeddings (deduplication, retrieval) | paraphrase-multilingual-MiniLM-L12-v2 |
| Lexical retrieval | BM25 (rank-bm25) |
| Graph centrality | PageRank (NetworkX, d=0.85) |

---

## Citation

```
Karimova E. Development of a Multi-Agent System for Automatic
Construction and Improvement of Ontologies. Bachelor's Thesis (Individual Submission),
HSE University, Moscow, 2026.
GitHub: https://github.com/ekaesha/hirag-ontology
```
