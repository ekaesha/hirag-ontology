"""
evaluation/latency_eval.py
==========================
Measures end-to-end system latency and per-stage breakdown.

AUTHOR: Eva Karimova (E. Karimova, Individual Contribution)

HOW TO RUN:
    python -m evaluation.latency_eval --n 20
    python -m evaluation.latency_eval --n 20 --kg results/knowledge_graph_final.json

MATCHES ACTUAL API:
    - KnowledgeGraph.load(path)  is a @classmethod, use it directly
    - HybridRetriever(kg, mode=RetrievalMode.X)  mode in constructor
    - retrieve(query, top_k=10)  returns list[str] of entity IDs
"""

import json
import os
import sys
import time
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from retrieval.retriever import HybridRetriever, RetrievalMode
from pipeline.knowledge_graph import KnowledgeGraph


# ── prompt template ───────────────────────────────────────────────────────────

GENERATION_PROMPT = """\
You are a clinical decision support assistant specialising in oncology.
Use the following retrieved entities to answer the question concisely.

Retrieved context:
{context}

Question: {question}

Answer based only on the provided context."""


def format_context(entity_ids: list, kg: KnowledgeGraph, max_entities: int = 10) -> str:
    """Convert list of entity IDs → readable context string."""
    lines = []
    for eid in entity_ids[:max_entities]:
        ent = kg.entities.get(eid)
        if ent is None:
            continue
        desc = getattr(ent, "description", "") or ""
        line = f"[{ent.entity_type}] {ent.label}: {desc[:120]}"
        lines.append(line)
    return "\n".join(lines) if lines else "No entities retrieved."


def get_llm_client():
    """Returns an OpenAI-compatible client pointing at DeepSeek."""
    try:
        import openai
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    except ImportError:
        return None


# ── core measurement ──────────────────────────────────────────────────────────

def measure_single_query(
    question: str,
    retriever: HybridRetriever,
    kg: KnowledgeGraph,
    llm=None,
    top_k: int = 10,
) -> dict:
    """
    Times each pipeline stage for one query.

    Stages:
        retrieval_s      — Hybrid RRF retrieval
        context_format_s — Building context string from entity IDs
        generation_s     — LLM answer generation (0 if no API key)
        total_s          — Sum of all stages
    """
    # ── Stage 1: Hybrid RRF retrieval ─────────────────────────────────────────
    t0 = time.perf_counter()
    retrieved_ids = retriever.retrieve(question, top_k=top_k)   # returns list[str]
    t1 = time.perf_counter()

    # ── Stage 2: Context formatting ───────────────────────────────────────────
    t2 = time.perf_counter()
    context_str = format_context(retrieved_ids, kg, max_entities=top_k)
    t3 = time.perf_counter()

    # ── Stage 3: LLM generation (optional) ───────────────────────────────────
    generation_s = 0.0
    answer = ""
    if llm is not None:
        try:
            prompt = GENERATION_PROMPT.format(context=context_str, question=question)
            t4 = time.perf_counter()
            response = llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.1,
            )
            t5 = time.perf_counter()
            generation_s = t5 - t4
            answer = response.choices[0].message.content or ""
        except Exception as e:
            generation_s = 0.0

    retrieval_s = t1 - t0
    context_s   = t3 - t2
    total_s     = retrieval_s + context_s + generation_s

    return {
        "retrieval_s":      retrieval_s,
        "context_format_s": context_s,
        "generation_s":     generation_s,
        "total_s":          total_s,
        "n_entities_retrieved": len(retrieved_ids),
        "answer_length_chars":  len(answer),
    }


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate_timings(all_timings: list) -> dict:
    keys = ["retrieval_s", "context_format_s", "generation_s", "total_s"]
    agg = {}
    for k in keys:
        values = [t[k] for t in all_timings]
        agg[k] = {
            "mean": statistics.mean(values),
            "std":  statistics.stdev(values) if len(values) > 1 else 0.0,
            "min":  min(values),
            "max":  max(values),
        }
    return agg


def print_latency_table(agg: dict) -> None:
    stage_names = {
        "retrieval_s":      "Hybrid RRF retrieval (E. Karimova)",
        "context_format_s": "Context formatting",
        "generation_s":     "LLM answer generation (DeepSeek-Chat)",
        "total_s":          "End-to-end (TTFT proxy)",
    }
    total_mean = agg["total_s"]["mean"]
    print("\n" + "=" * 78)
    print(f"{'Stage':<42} {'Mean(s)':>8} {'Std':>6} {'Min':>6} {'Max':>6} {'% Total':>8}")
    print("=" * 78)
    for key, name in stage_names.items():
        m = agg[key]
        pct = (m["mean"] / total_mean * 100) if total_mean > 0 else 0.0
        if key == "total_s":
            print("-" * 78)
        print(
            f"{name:<42} {m['mean']:>8.3f} {m['std']:>6.3f} "
            f"{m['min']:>6.3f} {m['max']:>6.3f} {pct:>7.1f}%"
        )
    print("=" * 78)
    print(f"\nTTFT proxy: {total_mean:.2f}s ±{agg['total_s']['std']:.2f}s\n")


# ── main runner ───────────────────────────────────────────────────────────────

def run_latency_eval(
    kg_path: str = "results/knowledge_graph_final.json",
    gt_path: str = "evaluation/ground_truth.json",
    n_queries: int = 20,
) -> dict:

    # Load KG — use classmethod, NOT instance method
    print(f"[latency_eval] Loading KG from {kg_path} ...")
    kg = KnowledgeGraph.load(kg_path)
    print(f"[latency_eval] KG: {len(kg.entities)} entities")

    # Load questions
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    questions = [q["question"] for q in gt_data["questions"]][:n_queries]
    print(f"[latency_eval] Running {len(questions)} queries\n")

    # Build retriever — mode in constructor, not in retrieve()
    retriever = HybridRetriever(kg, mode=RetrievalMode.HYBRID_RRF)

    # LLM client (optional — works without API key)
    llm = get_llm_client()
    if llm is None:
        print("[latency_eval] No API key found — running retrieval-only benchmark\n")
    else:
        print("[latency_eval] API key found — running full pipeline benchmark\n")

    # Benchmark loop
    all_timings = []
    per_query_log = []

    for i, question in enumerate(questions):
        timings = measure_single_query(
            question=question,
            retriever=retriever,
            kg=kg,
            llm=llm,
            top_k=10,
        )
        all_timings.append(timings)
        per_query_log.append({"question": question, **timings})

        print(
            f"  Query {i+1:02d}/{len(questions)}: "
            f"retrieval={timings['retrieval_s']:.3f}s  "
            f"generation={timings['generation_s']:.3f}s  "
            f"total={timings['total_s']:.3f}s  "
            f"entities={timings['n_entities_retrieved']}"
        )

    agg = aggregate_timings(all_timings)

    return {
        "aggregated": agg,
        "per_query":  per_query_log,
        "n_queries":  len(questions),
        "has_llm":    llm is not None,
    }


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Measure RAG pipeline latency.")
    parser.add_argument("--n",  type=int, default=20)
    parser.add_argument("--kg", default="results/knowledge_graph_final.json")
    parser.add_argument("--gt", default="evaluation/ground_truth.json")
    args = parser.parse_args()

    print(f"\n[latency_eval] Benchmarking {args.n} queries ...\n")
    results = run_latency_eval(kg_path=args.kg, gt_path=args.gt, n_queries=args.n)

    print_latency_table(results["aggregated"])

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "latency_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[latency_eval] Saved → {out_path}")