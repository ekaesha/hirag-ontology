# HiRAG-Ontology

Multi-agent pipeline for automatic ontology construction and knowledge graph
improvement from unstructured text, with hybrid RRF-based retrieval for
GraphRAG quality enhancement.

**Bachelor's Thesis — HSE DSBA, 2026**  
Authors: Eva Karimova, Alexey Popov

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ekaesha/hirag-ontology/blob/main/hirag_ontology_colab.ipynb)

---

## Results

Evaluated on 78 oncological clinical guidelines from the Russian Ministry
of Health (Minzdrav dataset). 50-question domain-specific benchmark.
LLM-as-judge: DeepSeek-Chat · Scale: 0–10.

| System | Comp | Emp | Div | Overall |
|--------|------|-----|-----|---------|
| Naive RAG | 5.40 | 4.94 | 5.42 | 5.34 |
| HiRAG (baseline) | 6.36 | 5.74 | 6.36 | 6.28 |
| HiRAG + Dedup | 5.34 | 4.64 | 5.34 | 5.10 |
| **HiRAG-Ontology (Ours)** | **6.82** | **6.00** | **6.90** | **6.68** |

**+25.1% over Naive RAG** · **+6.4% over HiRAG baseline**

### Ablation Study

| Configuration | Overall | vs Full System |
|---------------|---------|----------------|
| Baseline (semantic only, no dedup) | 5.20 | −26.8% |
| w/o Hybrid Retriever | 5.40 | −23.9% |
| w/o Deduplication | 7.20 | +1.4% |
| **Full System (Ours)** | **7.10** | — |

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
| Top entity by degree | ОЛЛ (116 connections) |

---

## System Architecture

Six specialised agents (A1–A6):

| Agent | Function | Formal |
|-------|----------|--------|
| A1 Extraction | LLM-based triplet extraction | fθ: C → 2^T |
| A2 Typing | Ontological class assignment | τ: V → C |
| A3 Deduplication | Hybrid lexical+semantic merging | π: V → V_canon |
| A4 Validation | Rule-based consistency checking | Cons(G) = 1 − \|violations\|/\|A\| |
| A5 Reasoning | Missing relation inference | T_missing = T* − T̂ |
| A6 Update | PageRank + graph persistence | G_T = (A_m ∘ ... ∘ A_1)(G_0) |

**Hybrid retriever:** BM25 + BERT embeddings + PageRank → RRF (k=60)

**Quality functional:** Q(G) = λ1·Coverage + λ2·Consistency + λ3·Precision − λ4·Redundancy

**Ontology:** `ontology.json` — 9 classes, 7 predicates, 5 axiom types

---

## Project Structure

```
hirag_eval/
├── ontology.json                  # Domain ontology O = (C, P, A)
├── pipeline/
│   ├── ontology_loader.py         # Dynamic ontology loading
│   ├── knowledge_graph.py         # Core data structure
│   ├── extractor.py               # A1 — extraction
│   ├── typing_agent.py            # A2 — typing
│   ├── deduplication.py           # A3 — deduplication
│   ├── validator.py               # A4+A5 — validation + reasoning
│   └── quality.py                 # Q(G) functional
├── retrieval/
│   └── retriever.py               # Hybrid RRF retriever
├── evaluation/
│   ├── judge.py                   # LLM-as-judge
│   └── run_eval.py                # Main experiment runner
├── iterative_pipeline.py          # Iterative improvement loop
├── langchain_integration.py       # LangChain wrapper
├── web_demo.py                    # Browser demo (localhost:5000)
├── knowledge_graph_explorer.html  # Interactive graph visualisation
├── graph_explorer.ipynb           # Jupyter Notebook
├── hirag_ontology_colab.ipynb     # Google Colab notebook
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt
```

Create `.env`:
```
DEEPSEEK_API_KEY=your_key_here
```

**Run full pipeline:**
```bash
python -m evaluation.run_eval
```

**Run iterative loop:**
```bash
python iterative_pipeline.py
```

**Web demo:**
```bash
python web_demo.py
# Open http://localhost:5000
```

**Google Colab:** click the badge at the top of this README.

**LangChain:**
```python
from langchain_integration import build_langchain_rag_chain
chain = build_langchain_rag_chain("results/knowledge_graph_final.json")
answer = chain.invoke("What is the treatment protocol for ALL?")
```

---

## Dataset

78 clinical guidelines from the Russian Ministry of Health (Minzdrav),
covering oncological conditions in Russian Markdown format.

---

## Models

| Component | Model |
|-----------|-------|
| Extraction, Typing, Generation, Judge | DeepSeek-Chat |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |

---

## Citation

```
Karimova E., Popov A. Development of a Multi-Agent System for Automatic
Construction and Improvement of Ontologies. Bachelor's Thesis,
HSE University, Moscow, 2026.
```
