"""
iterative_pipeline.py — multi-agent iterative improvement loop.

Реализует формулу из диплома:
  G_T = (A_m ∘ ... ∘ A_1)(G_0)

Запускает A4→A5→A6 итерационно пока Q(G) растёт или max_iter.
Сохраняет историю для построения графика.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.knowledge_graph import KnowledgeGraph
from pipeline.validator import ValidationAgent, ReasoningAgent
from pipeline.quality import compute_quality


def run_iterative_pipeline(
    kg: KnowledgeGraph,
    max_iter: int = 5,
    min_improvement: float = 0.001,
) -> list[dict]:
    """
    Итерационно применяет A4→A5→A6 к графу.
    Останавливается когда Q(G) перестаёт расти.

    Возвращает историю метрик по итерациям.
    """
    validator = ValidationAgent()
    reasoner  = ReasoningAgent()
    history   = []

    print(f"\n{'='*55}")
    print("Iterative Multi-Agent Pipeline")
    print(f"G_T = (A6 ∘ A5 ∘ A4)^T (G_0)")
    print(f"{'='*55}")

    prev_q = 0.0

    for iteration in range(max_iter + 1):
        # A4 — Validation
        val = validator.validate(kg)
        cons = val["consistency_score"]

        # A5 — Reasoning (skip on last iter)
        suggestions = []
        added = 0
        if iteration < max_iter:
            suggestions = reasoner.find_missing_relations(kg)
            added = reasoner.apply_suggestions(
                kg, suggestions, max_apply=min(10, len(suggestions))
            )

        # A6 — Update PageRank
        kg.compute_pagerank()

        # Q(G)
        q_result = compute_quality(kg, val)
        q = q_result["Q"]

        step = {
            "iteration":   iteration,
            "nodes":       kg.stats()["nodes"],
            "edges":       kg.stats()["edges"],
            "violations":  val["total_violations"],
            "consistency": cons,
            "coverage":    q_result["coverage"],
            "precision":   q_result["precision"],
            "redundancy":  q_result["redundancy"],
            "Q":           q,
            "suggestions": len(suggestions),
            "added":       added,
        }
        history.append(step)

        delta = q - prev_q
        print(f"\n[Iter {iteration}]  nodes={step['nodes']}  "
              f"edges={step['edges']}  Cons={cons:.3f}  Q={q:.4f}  "
              f"Δ={delta:+.4f}  added={added}")

        if iteration > 0 and delta < min_improvement:
            print(f"\n  Converged after {iteration} iterations "
                  f"(Δ={delta:.4f} < {min_improvement})")
            break

        prev_q = q

    print(f"\n{'='*55}")
    print(f"Final:  Q(G) = {history[-1]['Q']:.4f}  "
          f"Cons = {history[-1]['consistency']:.3f}")
    print(f"Start:  Q(G) = {history[0]['Q']:.4f}  "
          f"Cons = {history[0]['consistency']:.3f}")
    delta_total = history[-1]['Q'] - history[0]['Q']
    print(f"Total improvement: Δ = {delta_total:+.4f} "
          f"({delta_total/history[0]['Q']*100:+.1f}%)")
    print(f"{'='*55}")

    return history


def plot_history(history: list[dict], save_path: str) -> None:
    """Построить график Q(G) и Cons(G) по итерациям."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    iters = [h["iteration"] for h in history]
    qs    = [h["Q"]           for h in history]
    cons  = [h["consistency"] for h in history]
    edges = [h["edges"]       for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Q(G) and Cons(G)
    ax1.plot(iters, qs,   'o-', color='#534AB7', linewidth=2.2,
             markersize=8, label='Q(G)', zorder=3)
    ax1.plot(iters, cons, 's--', color='#D85A30', linewidth=1.8,
             markersize=7, label='Cons(G)', zorder=3)
    for i, (q, c) in enumerate(zip(qs, cons)):
        ax1.annotate(f'{q:.4f}', (iters[i], q),
                     textcoords='offset points', xytext=(0, 10),
                     ha='center', fontsize=9, color='#534AB7')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Q(G) and Cons(G) per iteration', fontsize=12,
                  fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xticks(iters)
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(alpha=0.25, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: Edge count growth
    ax2.bar(iters, edges, color='#1D9E75', alpha=0.8, edgecolor='none')
    for i, e in enumerate(edges):
        ax2.text(iters[i], e + 0.5, str(e),
                 ha='center', va='bottom', fontsize=10, color='#333')
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Number of relations', fontsize=11)
    ax2.set_title('Relation count growth per iteration', fontsize=12,
                  fontweight='bold')
    ax2.set_xticks(iters)
    ax2.set_ylim(0, max(edges) * 1.12)
    ax2.grid(axis='y', alpha=0.25, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle('Fig. 5.5. Iterative multi-agent pipeline convergence\n'
                 'G_T = (A6 ∘ A5 ∘ A4)^T (G_0)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Plot saved: {save_path}")


def main():
    print("Loading knowledge graph...")
    kg = KnowledgeGraph.load("results/knowledge_graph_dedup.json")

    history = run_iterative_pipeline(kg, max_iter=5, min_improvement=0.0005)

    # Save history
    with open("results/iterative_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\n[Saved] results/iterative_history.json")

    # Plot
    plot_history(history, "results/chart6_iterative.png")

    # Save final graph
    kg.save("results/knowledge_graph_iterative.json")


if __name__ == "__main__":
    main()
