"""
build_approach_b.py — Full Approach B pipeline. Run on Sol via SLURM.

Pipeline:
  1. PageRenderer   → render all PDF pages as PNG images at 150 DPI
  2. QdrantIndexer  → embed each page with ColQwen2, store in Qdrant
  3. Save page manifest JSON
  4. Log all stats to W&B

Usage:
  source ~/envs/rag/bin/activate
  cd /scratch/ngangada/portfolio/colpali-multimodal-rag
  python scripts/build_approach_b.py --config configs/config.yaml
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
log = logging.getLogger("build_approach_b")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import wandb
    run = wandb.init(
        project=cfg["wandb"]["project"],
        name="approach_b_build",
        tags=cfg["wandb"]["base_tags"] + ["approach_b", "build"],
        config={"extraction": cfg["extraction"], "approach": "B"},
    )

    t_total = time.perf_counter()

    # ── Step 1: Render PDF pages ─────────────────────────────────────────────
    log.info("STEP 1/2 — Rendering PDF pages")
    from src.extractors.page_renderer import PageRenderer
    renderer = PageRenderer(cfg=cfg, wandb_run=run)
    pages_meta = renderer.render_directory(cfg["paths"]["data_raw_pdfs"])
    log.info(f"Rendered {len(pages_meta)} pages total")

    # ── Step 2: Embed and index ──────────────────────────────────────────────
    log.info("STEP 2/2 — ColQwen2 embedding + Qdrant indexing")
    from src.indexers.qdrant_indexer import QdrantIndexer
    indexer = QdrantIndexer(cfg=cfg, wandb_run=run)
    indexed = indexer.index_pages(pages_meta)

    # ── Save page manifest ────────────────────────────────────────────────────
    processed_dir = Path(cfg["paths"]["data_processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = processed_dir / "approach_b_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(pages_meta, f, indent=2)
    log.info(f"Page manifest saved → {manifest_path}")

    elapsed = time.perf_counter() - t_total
    run.log({
        "build/total_pages": len(pages_meta),
        "build/pages_indexed": indexed,
        "build/total_elapsed_sec": elapsed,
    })
    run.finish()
    log.info(f"Approach B build complete in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()