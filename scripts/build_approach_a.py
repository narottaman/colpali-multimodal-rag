"""
build_approach_a.py — Full Approach A pipeline. Run on Sol via SLURM.

Pipeline:
  1. DoclingExtractor  → extract text, tables, figures from all 10 PDFs
  2. QwenVLDescriber   → describe each figure with Qwen2-VL-7B (GPU)
  3. ChromaIndexer     → index all elements into 3 ChromaDB collections
  4. Save manifest JSON
  5. Log all stats to W&B

Usage:
  source ~/envs/rag/bin/activate
  cd /scratch/ngangada/portfolio/colpali-multimodal-rag
  python scripts/build_approach_a.py --config configs/config.yaml
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("build_approach_a")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip_describe", action="store_true",
                        help="Skip Qwen2-VL description (use caption placeholders)")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import wandb
    run = wandb.init(
        project=cfg["wandb"]["project"],
        name="approach_a_build",
        tags=cfg["wandb"]["base_tags"] + ["approach_a", "build"],
        config={"extraction": cfg["extraction"], "approach": "A"},
    )

    t_total = time.perf_counter()

    # ── Step 1: Extract ──────────────────────────────────────────────────────
    log.info("STEP 1/3 — Docling extraction")
    from src.extractors.docling_extractor import DoclingExtractor, load_config
    extractor = DoclingExtractor(cfg=cfg, wandb_run=run)
    elements = extractor.extract_directory(cfg["paths"]["data_raw_pdfs"])
    log.info(f"Extracted {len(elements)} elements total")

    # ── Step 2: Describe figures ─────────────────────────────────────────────
    if not args.skip_describe:
        log.info("STEP 2/3 — Qwen2-VL figure description")
        from src.describers.qwen_vl_describer import QwenVLDescriber
        describer = QwenVLDescriber(cfg=cfg, wandb_run=run)
        elements = describer.describe_figures(elements)
    else:
        log.info("STEP 2/3 — Skipping description (--skip_describe flag set)")

    # ── Step 3: Index ─────────────────────────────────────────────────────────
    log.info("STEP 3/3 — ChromaDB indexing")
    from src.indexers.chroma_indexer import ChromaIndexer
    indexer = ChromaIndexer(cfg=cfg, wandb_run=run)
    counts = indexer.index(elements)

    # ── Save manifest ─────────────────────────────────────────────────────────
    processed_dir = Path(cfg["paths"]["data_processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = processed_dir / "approach_a_manifest.json"

    # Strip image_base64 from manifest to keep file size reasonable
    manifest = []
    for e in elements:
        m = {k: v for k, v in e.items() if k != "image_base64"}
        manifest.append(m)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    log.info(f"Manifest saved → {manifest_path}")

    elapsed = time.perf_counter() - t_total
    run.log({
        "build/total_elements": len(elements),
        "build/total_elapsed_sec": elapsed,
        "build/index_text": counts["text"],
        "build/index_figures": counts["figure"],
        "build/index_tables": counts["table"],
    })
    run.finish()
    log.info(f"Approach A build complete in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()