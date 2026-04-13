# HiRAG-Ontology

Multi-agent pipeline for automatic ontology construction and knowledge graph
improvement from unstructured text, with hybrid RRF-based retrieval for
GraphRAG quality enhancement.

**Bachelor's Thesis — HSE DSBA, 2026**  
Authors: Eva Karimova, Alexey Popov

---

## Results

| System | Comp | Emp | Div | Overall |
|--------|------|-----|-----|---------|
| Naive RAG | 4.85 | 4.50 | 5.10 | 4.85 |
| HiRAG (baseline) | 5.55 | 5.20 | 4.90 | 5.40 |
| HiRAG + Dedup | 4.60 | 4.05 | 4.60 | 4.45 |
| **HiRAG-Ontology (Ours)** | **5.55** | **5.10** | **5.65** | **5.60** |

Evaluated on 10 oncological clinical guidelines from the Russian Ministry
of Health (Minzdrav dataset, 78 documents total).  
LLM-as-judge: DeepSeek-Chat · Scale: 0–10.

---

## Knowledge Graph Statistics

| Metric | Value |
|--------|-------|
| Entities (raw) | 2,727 |
| Entities (after deduplication) | 2,314 |
| Relations | 2,346 |
| Deduplication rate | 15.1% (413 entities merged) |
| Consistency score Cons(G) | 0.773 |
| Quality functional Q(G) | 0.730 |
| Top entity by degree | ОЛЛ (116 connections) |
| Documents processed | 10 of 78 |

---

## System Architecture

Six specialised agents (A1–A6) implement typed graph transformations:

| Agent | Function | Formal definition |
|-------|----------|-------------------|
| A1 Extraction | LLM-based triplet extraction from text chunks | fθ: C → 2^T |
| A2 Typing | Ontological class assignment | τ: V → C |
| A3 Deduplication | Hybrid lexical+semantic entity merging | π: V → V_canon |
| A4 Validation | Rule-based consistency checking | Cons(G) = 1 − \|violations\|/\|A\| |
| A5 Reasoning | Missing relation inference | T_missing = T* − T̂ |
| A6 Update | PageRank computation and graph persistence | G_{t+1} = A_i(G_t) |

**Pipeline composition:** G_T = (A_m ∘ ... ∘ A_1)(G_0)

**Hybrid retriever:** BM25 + BERT embeddings + PageRank → RRF fusion (k=60)

**Quality functional:** Q(G) = λ1·Coverage + λ2·Consistency + λ3·Precision − λ4·Redundancy

**Ontology:** O = (C, P, A) with 9 entity classes, 7 predicate types, 5 axiom types

---

## Project Structure

```
hirag_eval/
├── pipeline/
│   ├── knowledge_graph.py         # KnowledgeGraph data structure
│   ├── extractor.py               # A1 — triplet extraction via LLM
│   ├── typing_agent.py            # A2 — ontological typing τ: V → C
│   ├── deduplication.py           # A3 — hybrid sim deduplication
│   ├── validator.py               # A4+A5 — validation and reasoning
│   └── quality.py                 # Q(G) quality functional
├── retrieval/
│   └── retriever.py               # Hybrid RRF retriever (4 modes)
├── evaluation/
│   ├── judge.py                   # LLM-as-judge (DeepSeek-Chat)
│   └── run_eval.py                # Main experiment runner
├── iterative_pipeline.py          # Multi-agent iterative loop
├── web_demo.py                    # Browser demo (localhost:5000)
├── knowledge_graph_explorer.html  # Interactive graph visualisation
├── analyze.py                     # Graph analysis utilities
├── requirements.txt
└── .env                           # DEEPSEEK_API_KEY (not committed)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Create `.env` file in the project root:
```
DEEPSEEK_API_KEY=your_key_here
```

---

## Usage

### Run full pipeline + evaluation (Tables 5.1 and 5.2)
```bash
python -m evaluation.run_eval
```

Results saved to `results/`:
- `summary_table51.csv` — main evaluation results
- `ablation_table52.csv` — ablation study results
- `quality_report.json` — Q(G) before/after pipeline
- `pipeline_stats.json` — per-agent statistics
- `knowledge_graph_final.json` — final knowledge graph

### Run iterative improvement loop
```bash
python iterative_pipeline.py
```

Runs A4→A5→A6 iteratively until convergence.
Saves convergence plot to `results/chart6_iterative.png`.

### Launch interactive web demo
```bash
python web_demo.py
```
Open `http://localhost:5000` in browser.
Requires `results/knowledge_graph_final.json` and `DEEPSEEK_API_KEY`.

### Open interactive graph explorer
Open `knowledge_graph_explorer.html` directly in any browser.
No server or installation required.

---

## Dataset

78 clinical guidelines from the Russian Ministry of Health (Minzdrav),
covering oncological conditions:

- Haematological: acute lymphoblastic leukaemia, acute myeloid leukaemia,
  acute promyelocytic leukaemia, chronic lymphocytic leukaemia,
  chronic myeloid leukaemia, multiple myeloma, Hodgkin lymphoma,
  follicular lymphoma, and others
- Solid tumours: lung, breast, colorectal, cervical, ovarian, bladder,
  kidney, thyroid, melanoma, and others (42 conditions total)

Documents are in Russian Markdown format, ranging from 108 KB to 566 KB.

---

## Models

| Component | Model | Notes |
|-----------|-------|-------|
| Extraction (A1) | DeepSeek-Chat | ~4× cheaper than GPT-4o-mini |
| Typing (A2) | DeepSeek-Chat | MAP@1 = 0.93 |
| Generation | DeepSeek-Chat | temperature=0.1 |
| Evaluation (judge) | DeepSeek-Chat | 4 metrics, 0–10 scale |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | Russian + English |

---

## Ablation Study

| Configuration | Overall | vs Full System |
|---------------|---------|----------------|
| Baseline (semantic only, no dedup) | 6.00 | −7.7% |
| w/o Hybrid Retriever | 5.00 | −23.1% |
| w/o Deduplication | 7.50 | +15.4%* |
| **Full System (Ours)** | **6.50** | — |

*w/o Deduplication scores higher on Diversity due to increased lexical
variation in the raw graph — see thesis §5.5 for analysis.

---

## Citation

```
Karimova E., Popov A. Development of a Multi-Agent System for Automatic
Construction and Improvement of Ontologies. Bachelor's Thesis,
HSE University, Moscow, 2026.
```
