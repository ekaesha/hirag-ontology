"""
evaluation/run_full_eval.py
===========================
Master script that runs ALL evaluation components in sequence.

AUTHOR: Eva Karimova (E. Karimova, Individual Contribution)

WHAT THIS RUNS (in order):
    1. Retrieval metrics    → results/retrieval_metrics.json
    2. Generation metrics   → results/generation_metrics.json
    3. Latency benchmark    → results/latency_results.json
    4. Deduplication ablation → results/dedup_ablation.json
    5. Consolidated report  → results/full_evaluation_report.json

HOW TO RUN:
    python -m evaluation.run_full_eval
    # or
    python evaluation/run_full_eval.py

    Options:
      --skip-generation   Skip generation eval (saves API calls)
      --skip-dedup        Skip dedup ablation
      --n-latency INT     Number of latency queries (default: 20)
      --n-generation INT  Number of generation eval questions (default: 50)

REQUIREMENTS:
    - results/knowledge_graph_final.json  (built by run_eval.py)
    - evaluation/ground_truth.json        (provided in repo)
    - .env with DEEPSEEK_API_KEY
"""

import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def print_banner(title: str) -> None:
    width = 70
    print("\n" + "━" * width)
    print(f"  {title}")
    print("━" * width)


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation suite.")
    parser.add_argument("--kg",               default="results/knowledge_graph_final.json")
    parser.add_argument("--gt",               default="evaluation/ground_truth.json")
    parser.add_argument("--skip-generation",  action="store_true")
    parser.add_argument("--skip-dedup",       action="store_true")
    parser.add_argument("--n-latency",        type=int, default=20)
    parser.add_argument("--n-generation",     type=int, default=50)
    args = parser.parse_args()

    overall_start = time.perf_counter()
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "kg_path": args.kg,
        "gt_path": args.gt,
        "components": {}
    }

    # ── 1. RETRIEVAL METRICS ─────────────────────────────────────────────────
    print_banner("1/4  RETRIEVAL METRICS  (Hit@k, MRR, MAP@k)")
    try:
        from evaluation.retrieval_eval import run_retrieval_eval, print_summary_table
        ret_results = run_retrieval_eval(kg_path=args.kg, gt_path=args.gt, top_k=10)
        print_summary_table(ret_results["per_mode"])
        report["components"]["retrieval"] = ret_results["per_mode"]
        print("[✓] Retrieval metrics complete")
    except Exception as e:
        print(f"[✗] Retrieval metrics failed: {e}")
        report["components"]["retrieval"] = {"error": str(e)}

    # ── 2. GENERATION METRICS ────────────────────────────────────────────────
    if not args.skip_generation:
        print_banner("2/4  GENERATION METRICS  (Faithfulness, Relevance, Precision, Recall)")
        try:
            from evaluation.generation_eval import run_generation_eval, print_generation_table
            gen_results = run_generation_eval(
                kg_path=args.kg, gt_path=args.gt, n_questions=args.n_generation
            )
            print_generation_table(gen_results["summary"])
            report["components"]["generation"] = gen_results["summary"]
            print("[✓] Generation metrics complete")
        except Exception as e:
            print(f"[✗] Generation metrics failed: {e}")
            report["components"]["generation"] = {"error": str(e)}
    else:
        print("\n[SKIP] Generation metrics skipped (--skip-generation flag)")

    # ── 3. LATENCY BENCHMARK ─────────────────────────────────────────────────
    print_banner("3/4  LATENCY BENCHMARK  (TTFT, per-stage breakdown)")
    try:
        from evaluation.latency_eval import run_latency_eval, print_latency_table
        lat_results = run_latency_eval(
            kg_path=args.kg, gt_path=args.gt, n_queries=args.n_latency
        )
        print_latency_table(lat_results["aggregated"])
        report["components"]["latency"] = {
            k: {stat: round(v, 4) for stat, v in m.items()}
            for k, m in lat_results["aggregated"].items()
        }
        print("[✓] Latency benchmark complete")
    except Exception as e:
        print(f"[✗] Latency benchmark failed: {e}")
        report["components"]["latency"] = {"error": str(e)}

    # ── 4. DEDUPLICATION ABLATION ────────────────────────────────────────────
    if not args.skip_dedup:
        print_banner("4/4  DEDUPLICATION ABLATION  (α × θ grid search)")
        try:
            from evaluation.dedup_ablation import run_dedup_ablation, print_ablation_table
            dedup_results = run_dedup_ablation(kg_path=args.kg)
            print_ablation_table(dedup_results)
            report["components"]["dedup_ablation"] = {
                "best": dedup_results[0],
                "selected_config": next(
                    (r for r in dedup_results if r["alpha"] == 0.6 and r["theta"] == 0.85),
                    None
                ),
                "all_configs": dedup_results,
            }
            print("[✓] Deduplication ablation complete")
        except Exception as e:
            print(f"[✗] Deduplication ablation failed: {e}")
            report["components"]["dedup_ablation"] = {"error": str(e)}
    else:
        print("\n[SKIP] Deduplication ablation skipped (--skip-dedup flag)")

    # ── FINAL REPORT ─────────────────────────────────────────────────────────
    total_time = time.perf_counter() - overall_start
    report["total_elapsed_s"] = round(total_time, 1)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "full_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "━" * 70)
    print(f"  FULL EVALUATION COMPLETE — {total_time:.0f}s total")
    print(f"  Report saved → {report_path}")
    print("━" * 70 + "\n")


if __name__ == "__main__":
    main()
