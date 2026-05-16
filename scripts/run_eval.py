"""
run_eval.py — Compare Approach A vs B on eval questions. Logs to W&B.

Requires GPU (submit via sol/run_eval.slurm).
Approach A uses sentence-transformer on CPU (small model, fast).
Approach B uses ColQwen2 on GPU (7B model, needs A100).
Gemini API handles generation — no GPU needed for that step.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
log = logging.getLogger("run_eval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max_questions", type=int, default=None)
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))

    # Sentence-transformer runs on CPU — small model, no need for GPU
    # ColQwen2 runs on GPU — needed for Approach B retrieval
    cfg["models"]["text_embedder_device"] = "cpu"
    cfg["models"]["colpali_device"] = "cuda"

    # Load eval questions
    qa_path = Path(cfg["paths"]["eval_qa_pairs"])
    if not qa_path.exists() or qa_path.stat().st_size == 0:
        log.error(f"qa_pairs.json missing or empty at {qa_path}")
        sys.exit(1)

    with open(qa_path) as f:
        qa_pairs = json.load(f)
    if args.max_questions:
        qa_pairs = qa_pairs[:args.max_questions]
    log.info(f"Evaluating {len(qa_pairs)} questions")

    import wandb
    run = wandb.init(
        project=cfg["wandb"]["project"],
        name="eval_comparison",
        tags=cfg["wandb"]["base_tags"] + ["eval"],
        config={"num_questions": len(qa_pairs)},
    )

    # Init retrievers
    from src.retrievers.hybrid_retriever import HybridRetriever
    from src.retrievers.colpali_retriever import ColPaliRetriever
    from src.generator import GeminiGenerator

    log.info("Loading Approach A retriever (CPU)...")
    retriever_a = HybridRetriever(cfg=cfg)

    log.info("Loading Approach B retriever (GPU)...")
    retriever_b = ColPaliRetriever(cfg=cfg)

    log.info("Loading Gemini generator...")
    generator = GeminiGenerator(cfg=cfg)

    results_log = []
    metrics_a = {"hit": [], "relevancy": [], "latency": [], "cost": [], "figures": [], "tables": []}
    metrics_b = {"hit": [], "relevancy": [], "latency": [], "cost": [], "pages": []}

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        gt_paper = qa.get("ground_truth_paper", "")
        gt_page  = qa.get("ground_truth_page", -1)
        log.info(f"[{i+1}/{len(qa_pairs)}] {question[:70]}")

        # ── Approach A ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        ctx_a  = retriever_a.retrieve(question)
        gen_a  = generator.generate_approach_a(question, ctx_a)
        lat_a  = (time.perf_counter() - t0) * 1000
        rel_a  = generator.judge_relevancy(question, gen_a["answer"])

        # context hit: did the right paper appear?
        hit_a = 0
        for r in ctx_a["text_results"] + ctx_a["figure_results"] + ctx_a["table_results"]:
            if gt_paper in r.get("filename", "") and abs(r.get("page_num", -99) - gt_page) <= 1:
                hit_a = 1
                break

        metrics_a["hit"].append(hit_a)
        metrics_a["relevancy"].append(rel_a)
        metrics_a["latency"].append(lat_a)
        metrics_a["cost"].append(gen_a["cost_usd"])
        metrics_a["figures"].append(1 if ctx_a["has_figures"] else 0)
        metrics_a["tables"].append(1 if ctx_a["has_tables"] else 0)

        # ── Approach B ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        pages_b = retriever_b.retrieve(question)
        gen_b   = generator.generate_approach_b(question, pages_b)
        lat_b   = (time.perf_counter() - t0) * 1000
        rel_b   = generator.judge_relevancy(question, gen_b["answer"])

        hit_b = 0
        for p in pages_b:
            if gt_paper in p.get("filename", "") and abs(p.get("page_num", -99) - gt_page) <= 1:
                hit_b = 1
                break

        metrics_b["hit"].append(hit_b)
        metrics_b["relevancy"].append(rel_b)
        metrics_b["latency"].append(lat_b)
        metrics_b["cost"].append(gen_b["cost_usd"])
        metrics_b["pages"].append(len(pages_b))

        # Per-question W&B log
        run.log({
            "q/index":               i,
            "approach_a/hit":        hit_a,
            "approach_a/relevancy":  rel_a,
            "approach_a/latency_ms": lat_a,
            "approach_a/cost_usd":   gen_a["cost_usd"],
            "approach_b/hit":        hit_b,
            "approach_b/relevancy":  rel_b,
            "approach_b/latency_ms": lat_b,
            "approach_b/cost_usd":   gen_b["cost_usd"],
        })

        results_log.append({
            "question": question,
            "approach_a": {
                "answer": gen_a["answer"], "hit": hit_a,
                "relevancy": rel_a, "latency_ms": lat_a, "cost_usd": gen_a["cost_usd"],
            },
            "approach_b": {
                "answer": gen_b["answer"], "hit": hit_b,
                "relevancy": rel_b, "latency_ms": lat_b, "cost_usd": gen_b["cost_usd"],
            },
        })

        log.info(f"  A: hit={hit_a} rel={rel_a} ${gen_a['cost_usd']:.5f} | "
                 f"B: hit={hit_b} rel={rel_b} ${gen_b['cost_usd']:.5f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    def avg(lst): return round(sum(lst) / max(len(lst), 1), 4)

    summary = {
        "approach_a": {
            "context_hit_rate":      avg(metrics_a["hit"]),
            "answer_relevancy":      avg(metrics_a["relevancy"]),
            "avg_latency_ms":        avg(metrics_a["latency"]),
            "total_cost_usd":        round(sum(metrics_a["cost"]), 5),
            "figures_retrieved_rate": avg(metrics_a["figures"]),
            "tables_retrieved_rate":  avg(metrics_a["tables"]),
        },
        "approach_b": {
            "context_hit_rate":      avg(metrics_b["hit"]),
            "answer_relevancy":      avg(metrics_b["relevancy"]),
            "avg_latency_ms":        avg(metrics_b["latency"]),
            "total_cost_usd":        round(sum(metrics_b["cost"]), 5),
            "avg_pages_retrieved":   avg(metrics_b["pages"]),
        },
    }

    log.info("\n=== EVAL SUMMARY ===")
    for approach, m in summary.items():
        log.info(f"\n{approach}:")
        for k, v in m.items():
            log.info(f"  {k}: {v}")

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