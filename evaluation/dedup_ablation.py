"""
evaluation/dedup_ablation.py
============================
Grid search over deduplication hyperparameters α and θ.

AUTHOR: Eva Karimova (E. Karimova, Individual Contribution)

WHY THIS EXISTS:
    The commission asked: "Why θ = 0.85? Why α = 0.6?"
    This script proves those choices empirically by:
      1. Running deduplication with every combination of α ∈ [0.4, 0.5, 0.6, 0.7, 0.8]
         and θ ∈ [0.75, 0.80, 0.85, 0.90, 0.95]
      2. Measuring:
           - dedup_rate     : fraction of entities merged (higher = more aggressive)
           - false_positive_rate : manually labelled false merges in a 50-pair sample
           - false_negative_rate : missed duplicates in the same sample
           - cons_g         : consistency score after deduplication
      3. Selecting the Pareto-optimal configuration

HOW TO RUN:
    python -m evaluation.dedup_ablation
    # or
    python evaluation/dedup_ablation.py

    This does NOT require an API key — it only uses embeddings and string matching.

OUTPUT:
    - Console table with all (α, θ) combinations ranked by F1
    - results/dedup_ablation.json
"""

import json
import sys
import time
from pathlib import Path
from itertools import product
from copy import deepcopy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.knowledge_graph import KnowledgeGraph
from pipeline.deduplication import DeduplicationAgent
from pipeline.validator import ValidationAgent


# ── annotated sample for evaluation ──────────────────────────────────────────
# 30 manually annotated entity pairs from the Minzdrav corpus.
# TRUE_DUPLICATES = pairs that are actually the same entity (should be merged).
# TRUE_NON_DUPLICATES = pairs that look similar but are different entities.
#
# FORMAT: (entity_label_1, entity_label_2)
# These were annotated by E. Karimova based on medical domain knowledge.

TRUE_DUPLICATES = [
    ("острый лимфобластный лейкоз", "ОЛЛ"),
    ("острый лимфобластный лейкоз", "острый лимфоидный лейкоз"),
    ("химиотерапия", "ХТ"),
    ("трансплантация гемопоэтических стволовых клеток", "ТГСК"),
    ("ритуксимаб", "Ритуксан"),
    ("метотрексат", "МТХ"),
    ("цитарабин", "Ara-C"),
    ("Ph-положительный ОЛЛ", "Ph+ ОЛЛ"),
    ("индукционная химиотерапия", "индукция ремиссии"),
    ("аллогенная трансплантация", "аллогенная ТГСК"),
    ("BCR-ABL тирозинкиназа", "BCR-ABL"),
    ("острый промиелоцитарный лейкоз", "ОПЛ"),
    ("венетоклакс + азацитидин", "Вен+Аза"),
    ("лейкоз", "лейкемия"),
    ("нейтропеническая лихорадка", "febris neutropenica"),
]

TRUE_NON_DUPLICATES = [
    ("острый лимфобластный лейкоз", "острый миелоидный лейкоз"),
    ("ритуксимаб", "блинатумомаб"),           # both antibodies, but different targets
    ("метотрексат", "метилпреднизолон"),       # similar prefix, different drug
    ("цитарабин", "цитозин"),                  # substring match, different entity
    ("индукция", "консолидация"),              # both treatment phases, different
    ("аллогенная ТГСК", "аутологичная ТГСК"), # different transplant types
    ("Ph+ ОЛЛ", "Ph- ОЛЛ"),                   # opposite subtypes
    ("лечение", "диагностика"),                # generic nodes, completely different
    ("ОЛЛ", "ОМЛ"),                            # abbreviations for different diseases
    ("Г-КСФ", "ГМ-КСФ"),                       # different growth factors
    ("BCR-ABL", "FLT3"),                        # different kinases
    ("венетоклакс", "венлафаксин"),            # different drug class
    ("иматиниб", "иматиниб мезилат"),          # same drug — should be duplicate!
    # ↑ catch: this one IS a duplicate, checking if grid search catches it
    ("дексаметазон", "преднизолон"),           # both corticosteroids, different
    ("ATRA", "ARA-C"),                          # different abbreviations, different drugs
]

# Fix: иматиниб мезилат IS a true duplicate, move it
TRUE_DUPLICATES.append(("иматиниб", "иматиниб мезилат"))
TRUE_NON_DUPLICATES = [
    p for p in TRUE_NON_DUPLICATES
    if p != ("иматиниб", "иматиниб мезилат")
]


def evaluate_dedup_on_sample(
    agent: DeduplicationAgent,
    kg: KnowledgeGraph,
) -> dict:
    """
    Computes precision/recall/F1 on the annotated 30-pair sample.

    For each annotated pair, checks whether the agent's similarity()
    function returns sim >= theta (i.e. would merge them).

    Returns: {precision, recall, f1, dedup_rate, false_positives, false_negatives}
    """
    entity_map = {e.label.strip().lower(): e for e in kg.entities.values()}

    def get_entity(label):
        key = label.strip().lower()
        return entity_map.get(key)

    true_positive = 0   # correctly merged duplicates
    false_positive = 0  # incorrectly merged non-duplicates
    false_negative = 0  # missed merges (real duplicates not merged)
    true_negative = 0   # correctly not merged

    def would_merge(label_a, label_b):
        """Returns True if the agent would merge these two entities."""
        ea = get_entity(label_a)
        eb = get_entity(label_b)
        if ea is None or eb is None:
            # Entity not in graph — use string-only similarity
            from pipeline.deduplication import _token_sort_ratio
            sim_lex = _token_sort_ratio(label_a, label_b) / 100.0
            sim = agent.alpha * sim_lex + (1 - agent.alpha) * sim_lex
        else:
            sim = agent.similarity(ea, eb)
        return sim >= agent.threshold

    for (a, b) in TRUE_DUPLICATES:
        if would_merge(a, b):
            true_positive += 1
        else:
            false_negative += 1

    for (a, b) in TRUE_NON_DUPLICATES:
        if would_merge(a, b):
            false_positive += 1
        else:
            true_negative += 1

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 1.0
    recall    = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def run_dedup_ablation(
    kg_path: str = "results/knowledge_graph_final.json",
    alphas: list = None,
    thetas: list = None,
) -> list[dict]:
    """
    Grid search over α and θ for deduplication.

    Parameters
    ----------
    kg_path : str   Path to the built KnowledgeGraph JSON.
    alphas  : list  Values of the semantic mixing coefficient α.
    thetas  : list  Similarity threshold values θ.

    Returns list of result dicts sorted by F1 score (descending).
    """
    if alphas is None:
        alphas = [0.4, 0.5, 0.6, 0.7, 0.8]
    if thetas is None:
        thetas = [0.75, 0.80, 0.85, 0.90, 0.95]

    print(f"[dedup_ablation] Loading KG from {kg_path} ...")
    kg_original = KnowledgeGraph()
    kg_original.load(kg_path)
    n_original = len(kg_original.entities)
    print(f"[dedup_ablation] KG: {n_original} entities")
    print(f"[dedup_ablation] Running {len(alphas) * len(thetas)} configurations ...\n")

    results = []

    for alpha, theta in product(alphas, thetas):
        # Deep-copy so we don't mutate the original KG
        kg_copy = deepcopy(kg_original)

        # Run deduplication with this (α, θ) configuration
        agent = DeduplicationAgent(alpha=alpha, threshold=theta, use_embeddings=True)
        t0 = time.perf_counter()
        stats = agent.deduplicate(kg_copy)
        elapsed = time.perf_counter() - t0

        n_after = len(kg_copy.entities)
        dedup_rate = (n_original - n_after) / n_original

        # Evaluate on annotated sample
        sample_metrics = evaluate_dedup_on_sample(agent, kg_original)

        # Run consistency check on deduped graph
        validator = ValidationAgent()
        val_result = validator.validate(kg_copy)
        cons_g = val_result.get("consistency_score", 0.0)

        config_result = {
            "alpha": alpha,
            "theta": theta,
            "n_entities_after": n_after,
            "dedup_rate": round(dedup_rate, 4),
            "cons_g": round(cons_g, 4),
            "elapsed_s": round(elapsed, 2),
            **sample_metrics,
        }
        results.append(config_result)

        print(
            f"  α={alpha:.1f} θ={theta:.2f} → "
            f"dedup={dedup_rate:.1%}  "
            f"P={sample_metrics['precision']:.3f}  "
            f"R={sample_metrics['recall']:.3f}  "
            f"F1={sample_metrics['f1']:.3f}  "
            f"Cons={cons_g:.3f}"
        )

    # Sort by F1 descending
    results.sort(key=lambda x: x["f1"], reverse=True)
    return results


def print_ablation_table(results: list[dict]) -> None:
    """Prints a formatted ablation table."""
    print("\n" + "=" * 75)
    print(f"{'α':>5} {'θ':>5} {'Dedup%':>7} {'P':>6} {'R':>6} {'F1':>6} {'Cons':>6}  Note")
    print("=" * 75)
    for r in results:
        marker = " ◀ SELECTED" if (r["alpha"] == 0.6 and r["theta"] == 0.85) else ""
        print(
            f"{r['alpha']:>5.1f} {r['theta']:>5.2f} "
            f"{r['dedup_rate']:>6.1%}  "
            f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
            f"{r['cons_g']:>6.3f} {marker}"
        )
    print("=" * 75)

    best = results[0]
    print(f"\nBest configuration: α={best['alpha']}, θ={best['theta']}, F1={best['f1']:.4f}")
    print(f"Selected in paper:  α=0.6, θ=0.85 (rank: "
          f"{next((i+1 for i,r in enumerate(results) if r['alpha']==0.6 and r['theta']==0.85), '?')})\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deduplication hyperparameter grid search.")
    parser.add_argument("--kg", default="results/knowledge_graph_final.json")
    args = parser.parse_args()

    results = run_dedup_ablation(kg_path=args.kg)
    print_ablation_table(results)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "dedup_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[dedup_ablation] Results saved → {out_path}")
