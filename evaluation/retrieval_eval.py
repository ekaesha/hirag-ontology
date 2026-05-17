"""
evaluation/retrieval_eval.py
============================
Computes intrinsic retrieval metrics for the HiRAG-Ontology system.

AUTHOR: Eva Karimova (E. Karimova, Individual Contribution)

WHAT THIS FILE DOES:
    For each of the 50 benchmark questions, runs all 4 retrieval modes
    and computes:
      - Hit@5  : is any ground-truth entity in the top-5 results?
      - Hit@10 : is any ground-truth entity in the top-10 results?
      - MRR    : Mean Reciprocal Rank
      - MAP@10 : Mean Average Precision at 10

HOW TO RUN:
    python -m evaluation.retrieval_eval
    python -m evaluation.retrieval_eval --kg results/knowledge_graph_final.json

IMPORTANT — matches actual HybridRetriever API:
    - retrieve(query, top_k=10) returns list of ENTITY ID strings, not objects
    - mode is set in constructor: HybridRetriever(kg, mode=RetrievalMode.X)
    - kg.entities is a dict {entity_id: Entity}
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrieval.retriever import HybridRetriever, RetrievalMode
from pipeline.knowledge_graph import KnowledgeGraph


# ─────────────────────────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase + strip so 'ОЛЛ' == 'олл' == ' олл '."""
    return text.strip().lower()


def build_label_set(entity_ids: list, kg: KnowledgeGraph) -> set:
    """
    Converts a list of entity IDs → set of normalised labels + aliases.

    Why we need aliases: entity "ОЛЛ" may have alias
    "острый лимфобластный лейкоз" — both should count as a match.
    """
    label_set = set()
    for eid in entity_ids:
        ent = kg.entities.get(eid)
        if ent is None:
            continue
        label_set.add(normalise(ent.label))
        for alias in (getattr(ent, "aliases", None) or []):
            if alias:
                label_set.add(normalise(alias))
    return label_set


def hit_at_k(retrieved_ids: list, kg: KnowledgeGraph,
             gt_labels: list, k: int) -> float:
    """
    Hit@K = 1.0 if ≥1 ground-truth entity appears in top-K results.

    The simplest retrieval metric — did we find ANYTHING correct?
    """
    top_k_set = build_label_set(retrieved_ids[:k], kg)
    gt_set = {normalise(l) for l in gt_labels}
    return 1.0 if top_k_set & gt_set else 0.0


def reciprocal_rank(retrieved_ids: list, kg: KnowledgeGraph,
                    gt_labels: list) -> float:
    """
    RR = 1 / rank_of_first_relevant_result.
    If relevant entity is at rank 3 → RR = 0.333.
    If nothing relevant found → RR = 0.0.
    MRR = mean of RR across all questions.
    """
    gt_set = {normalise(l) for l in gt_labels}
    for rank, eid in enumerate(retrieved_ids, start=1):
        ent = kg.entities.get(eid)
        if ent is None:
            continue
        labels = {normalise(ent.label)}
        for alias in (getattr(ent, "aliases", None) or []):
            if alias:
                labels.add(normalise(alias))
        if labels & gt_set:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(retrieved_ids: list, kg: KnowledgeGraph,
                            gt_labels: list, k: int) -> float:
    """
    AP@K rewards both finding relevant entities AND ranking them higher up.

    AP@K = (1/|relevant|) * Σ P@i  for each rank i where entity is relevant

    Example:
        Ground truth = {A, B}
        Retrieved = [A, X, B, Y, Z]
        Rank 1: A relevant → P@1 = 1/1 = 1.0
        Rank 3: B relevant → P@3 = 2/3 = 0.667
        AP@5 = (1/2) * (1.0 + 0.667) = 0.833
    """
    if not gt_labels:
        return 0.0

    gt_set = {normalise(l) for l in gt_labels}
    hits = 0
    precision_sum = 0.0

    for rank, eid in enumerate(retrieved_ids[:k], start=1):
        ent = kg.entities.get(eid)
        if ent is None:
            continue
        labels = {normalise(ent.label)}
        for alias in (getattr(ent, "aliases", None) or []):
            if alias:
                labels.add(normalise(alias))
        if labels & gt_set:
            hits += 1
            precision_sum += hits / rank

    denom = min(len(gt_set), k)
    return precision_sum / denom if denom > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_retrieval_eval(
    kg_path: str = "results/knowledge_graph_final.json",
    gt_path: str = "evaluation/ground_truth.json",
    top_k: int = 10,
) -> dict:
    """
    Runs all 4 retrieval modes on all 50 questions.
    Returns aggregated metrics + per-question breakdown.
    """

    # ── Load KG ───────────────────────────────────────────────────────────────
    print(f"[retrieval_eval] Loading KG from {kg_path} ...")
    # load() is a @classmethod that returns a new KnowledgeGraph instance.
    # Do NOT call kg.load() on an empty instance — the result would be discarded.
    kg = KnowledgeGraph.load(kg_path)
    n = len(kg.entities)
    print(f"[retrieval_eval] {n} entities available")

    if n == 0:
        raise RuntimeError(
            "KG loaded 0 entities — check that knowledge_graph_final.json exists "
            "and kg.entities is populated after load()."
        )

    # ── Load ground truth ─────────────────────────────────────────────────────
    with open(gt_path, "r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    print(f"[retrieval_eval] {len(questions)} questions loaded\n")

    # ── Build one retriever per mode ──────────────────────────────────────────
    # Mode is set in the constructor, NOT in retrieve() — matches actual API.
    modes = [
        ("semantic_only",   RetrievalMode.SEMANTIC_ONLY),
        ("lexical_only",    RetrievalMode.LEXICAL_ONLY),
        ("structural_only", RetrievalMode.STRUCTURAL_ONLY),
        ("hybrid_rrf",      RetrievalMode.HYBRID_RRF),
    ]
    print("[retrieval_eval] Initialising retrievers (first call builds BM25 + embeddings) ...")
    retrievers = {name: HybridRetriever(kg, mode=enum) for name, enum in modes}
    print("[retrieval_eval] Ready — starting evaluation\n")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    accum = {name: {"hit5": [], "hit10": [], "rr": [], "ap10": []}
             for name, _ in modes}
    per_question_log = []

    for i, q in enumerate(questions):
        q_text   = q["question"]
        gt_labels = q["relevant_entity_labels"]
        per_q = {"id": q["id"], "question": q_text,
                 "type": q["type"], "modes": {}}

        for mode_name, _ in modes:
            # retrieve() returns list[str] of entity IDs
            ids = retrievers[mode_name].retrieve(q_text, top_k=top_k)

            h5  = hit_at_k(ids, kg, gt_labels, k=5)
            h10 = hit_at_k(ids, kg, gt_labels, k=10)
            rr  = reciprocal_rank(ids, kg, gt_labels)
            ap  = average_precision_at_k(ids, kg, gt_labels, k=10)

            accum[mode_name]["hit5"].append(h5)
            accum[mode_name]["hit10"].append(h10)
            accum[mode_name]["rr"].append(rr)
            accum[mode_name]["ap10"].append(ap)

            top5 = [kg.entities[eid].label
                    for eid in ids[:5] if eid in kg.entities]
            per_q["modes"][mode_name] = {
                "hit@5": h5, "hit@10": h10, "rr": rr, "ap@10": ap,
                "top5_retrieved": top5,
            }

        per_question_log.append(per_q)
        if (i + 1) % 10 == 0 or (i + 1) == len(questions):
            print(f"  [{i+1:02d}/{len(questions)}] processed")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    aggregated = {}
    for mode_name, _ in modes:
        v = accum[mode_name]
        n = len(v["hit5"])
        aggregated[mode_name] = {
            "Hit@5":  round(sum(v["hit5"])  / n, 4),
            "Hit@10": round(sum(v["hit10"]) / n, 4),
            "MRR":    round(sum(v["rr"])    / n, 4),
            "MAP@10": round(sum(v["ap10"])  / n, 4),
            "n_questions": n,
        }

    # ── Question-type breakdown (hybrid only) ─────────────────────────────────
    type_breakdown = {}
    for qtype in ["single_entity_lookup", "relation_inference", "multi_hop_reasoning"]:
        subset = [r for r in per_question_log if r["type"] == qtype]
        if not subset:
            continue
        rrs  = [r["modes"]["hybrid_rrf"]["rr"]    for r in subset]
        h10s = [r["modes"]["hybrid_rrf"]["hit@10"] for r in subset]
        type_breakdown[qtype] = {
            "n":      len(subset),
            "MRR":    round(sum(rrs)  / len(rrs),  4),
            "Hit@10": round(sum(h10s) / len(h10s), 4),
        }

    return {
        "per_mode":               aggregated,
        "per_question":           per_question_log,
        "question_type_breakdown": type_breakdown,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(agg: dict) -> None:
    labels = {
        "semantic_only":   "Semantic-only (BERT)",
        "lexical_only":    "BM25 Lexical-only",
        "structural_only": "Structural-only (PageRank)",
        "hybrid_rrf":      "Hybrid RRF — proposed (E. Karimova)",
    }
    print("\n" + "=" * 74)
    print(f"{'System':<38} {'Hit@5':>6} {'Hit@10':>7} {'MRR':>7} {'MAP@10':>8}")
    print("=" * 74)
    for key, label in labels.items():
        if key not in agg:
            continue
        m = agg[key]
        marker = "  ◀" if key == "hybrid_rrf" else ""
        print(f"{label:<38} {m['Hit@5']:>6.4f} {m['Hit@10']:>7.4f} "
              f"{m['MRR']:>7.4f} {m['MAP@10']:>8.4f}{marker}")
    print("=" * 74 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Retrieval metrics: Hit@k, MRR, MAP@k for HiRAG-Ontology."
    )
    parser.add_argument("--kg", default="results/knowledge_graph_final.json")
    parser.add_argument("--gt", default="evaluation/ground_truth.json")
    parser.add_argument("--k",  type=int, default=10)
    args = parser.parse_args()

    t0 = time.perf_counter()
    results = run_retrieval_eval(kg_path=args.kg, gt_path=args.gt, top_k=args.k)
    elapsed = time.perf_counter() - t0

    print_summary_table(results["per_mode"])

    print("Breakdown by question type (Hybrid RRF):")
    for qtype, stats in results["question_type_breakdown"].items():
        print(f"  {qtype:<30}  n={stats['n']}  "
              f"MRR={stats['MRR']:.4f}  Hit@10={stats['Hit@10']:.4f}")

    print(f"\n[retrieval_eval] Done in {elapsed:.1f}s")

    out = Path("results")
    out.mkdir(exist_ok=True)
    with open(out / "retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results["per_mode"], f, ensure_ascii=False, indent=2)
    with open(out / "retrieval_metrics_per_question.json", "w", encoding="utf-8") as f:
        json.dump(results["per_question"], f, ensure_ascii=False, indent=2)
    print("[retrieval_eval] Saved → results/retrieval_metrics*.json")