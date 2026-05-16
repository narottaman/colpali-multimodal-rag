"""
run_eval.py — Compare Approach A vs B on eval questions. Logs to W&B.

For each question:
  - Run Approach A retrieval + Gemini generation
  - Run Approach B retrieval + Gemini generation
  - Gemini self-judges answer relevancy (1-5)
  - Tracks: context_hit_rate, answer_relevancy, latency_ms, cost_usd,
            figures_retrieved, tables_retrieved

Usage:
  source ~/envs/rag/bin/activate
  cd /scratch/ngangada/portfolio/colpali-multimodal-rag
  python scripts/run_eval.py --config configs/config.yaml
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("run_eval")


def context_hit(results_a: dict, results_b: list, ground_truth_paper: str, ground_truth_page: int) -> tuple[int, int]:
    """
    Returns (hit_a, hit_b): 1 if ground truth paper+page appears in top results, else 0.
    Lenient: matches if filename contains the arxiv ID.
    """
    def _check_a(ctx):
        all_results = ctx.get("text_results", []) + ctx.get("figure_results", []) + ctx.get("table_results", [])
        for r in all_results:
            fname = r.get("filename", "")
            pnum = r.get("page_num", -1)
            if ground_truth_paper in fname and abs(pnum - ground_truth_page) <= 1:
                return 1
        return 0

    def _check_b(pages):
        for r in pages:
            fname = r.get("filename", "")
            pnum = r.get("page_num", -1)
            if ground_truth_paper in fname and abs(pnum - ground_truth_page) <= 1:
                return 1
        return 0

    return _check_a(results_a), _check_b(results_b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Limit questions for quick test (e.g. --max_questions 10)")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load eval questions
    qa_path = cfg["paths"]["eval_qa_pairs"]
    with open(qa_path) as f:
        qa_pairs = json.load(f)
    if args.max_questions:
        qa_pairs = qa_pairs[:args.max_questions]
    log.info(f"Evaluating {len(qa_pairs)} questions")

    # Init W&B
    import wandb
    run = wandb.init(
        project=cfg["wandb"]["project"],
        name="eval_comparison",
        tags=cfg["wandb"]["base_tags"] + ["eval"],
        config={"num_questions": len(qa_pairs)},
    )

    # Init retrievers + generator
    from src.retrievers.hybrid_retriever import HybridRetriever
    from src.retrievers.colpali_retriever import ColPaliRetriever
    from src.generator import GeminiGenerator

    retriever_a = HybridRetriever(cfg=cfg)
    retriever_b = ColPaliRetriever(cfg=cfg)
    generator = GeminiGenerator(cfg=cfg)

    results_log = []
    metrics_a = {"hit": [], "relevancy": [], "latency": [], "cost": [], "figures": [], "tables": []}
    metrics_b = {"hit": [], "relevancy": [], "latency": [], "cost": [], "figures": [], "tables": []}

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        gt_paper = qa.get("ground_truth_paper", "")
        gt_page = qa.get("ground_truth_page", -1)
        log.info(f"[{i+1}/{len(qa_pairs)}] {question[:70]} …")

        # ── Approach A ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        ctx_a = retriever_a.retrieve(question)
        gen_a = generator.generate_approach_a(question, ctx_a)
        latency_a = (time.perf_counter() - t0) * 1000

        relevancy_a = generator.judge_relevancy(question, gen_a["answer"])
        hit_a, _ = context_hit(ctx_a, [], gt_paper, gt_page)

        metrics_a["hit"].append(hit_a)
        metrics_a["relevancy"].append(relevancy_a)
        metrics_a["latency"].append(latency_a)
        metrics_a["cost"].append(gen_a["cost_usd"])
        metrics_a["figures"].append(1 if ctx_a["has_figures"] else 0)
        metrics_a["tables"].append(1 if ctx_a["has_tables"] else 0)

        # ── Approach B ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        pages_b = retriever_b.retrieve(question)
        gen_b = generator.generate_approach_b(question, pages_b)
        latency_b = (time.perf_counter() - t0) * 1000

        relevancy_b = generator.judge_relevancy(question, gen_b["answer"])
        _, hit_b = context_hit(ctx_a, pages_b, gt_paper, gt_page)

        metrics_b["hit"].append(hit_b)
        metrics_b["relevancy"].append(relevancy_b)
        metrics_b["latency"].append(latency_b)
        metrics_b["cost"].append(gen_b["cost_usd"])
        metrics_b["figures"].append(len(pages_b))
        metrics_b["tables"].append(0)

        # W&B per-question log
        run.log({
            "q/index": i,
            "approach_a/hit": hit_a,
            "approach_a/relevancy": relevancy_a,
            "approach_a/latency_ms": latency_a,
            "approach_a/cost_usd": gen_a["cost_usd"],
            "approach_b/hit": hit_b,
            "approach_b/relevancy": relevancy_b,
            "approach_b/latency_ms": latency_b,
            "approach_b/cost_usd": gen_b["cost_usd"],
        })

        results_log.append({
            "question": question,
            "approach_a": {"answer": gen_a["answer"], "hit": hit_a, "relevancy": relevancy_a,
                           "latency_ms": latency_a, "cost_usd": gen_a["cost_usd"]},
            "approach_b": {"answer": gen_b["answer"], "hit": hit_b, "relevancy": relevancy_b,
                           "latency_ms": latency_b, "cost_usd": gen_b["cost_usd"]},
        })

    # ── Summary metrics ───────────────────────────────────────────────────────
    def avg(lst): return sum(lst) / max(len(lst), 1)

    summary = {
        "approach_a": {
            "context_hit_rate": avg(metrics_a["hit"]),
            "answer_relevancy": avg(metrics_a["relevancy"]),
            "avg_latency_ms": avg(metrics_a["latency"]),
            "total_cost_usd": sum(metrics_a["cost"]),
            "figures_retrieved_rate": avg(metrics_a["figures"]),
            "tables_retrieved_rate": avg(metrics_a["tables"]),
        },
        "approach_b": {
            "context_hit_rate": avg(metrics_b["hit"]),
            "answer_relevancy": avg(metrics_b["relevancy"]),
            "avg_latency_ms": avg(metrics_b["latency"]),
            "total_cost_usd": sum(metrics_b["cost"]),
            "figures_retrieved_rate": avg(metrics_b["figures"]),
            "tables_retrieved_rate": 0.0,
        },
    }

    log.info("\n=== EVAL SUMMARY ===")
    for approach, m in summary.items():
        log.info(f"\n{approach.upper()}:")
        for k, v in m.items():
            log.info(f"  {k}: {v:.4f}")

    # Log summary to W&B
    flat = {}
    for approach, m in summary.items():
        for k, v in m.items():
            flat[f"summary/{approach}/{k}"] = v
    run.log(flat)

    # Save results
    results_dir = Path(cfg["paths"]["eval_results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "eval_results.json", "w") as f:
        json.dump({"summary": summary, "per_question": results_log}, f, indent=2)
    with open(results_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Results saved → {results_dir}")

    run.finish()


if __name__ == "__main__":
    main()