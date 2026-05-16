"""
qdrant_indexer.py — Approach B: Index ColQwen2 patch embeddings into Qdrant.

For each PDF page image:
  - ColQwen2 produces a list of 128-dim patch vectors (~196 patches per page)
  - Each page is stored as one Qdrant point with multi-vector payload
  - Metadata stored: pdf_stem, filename, page_num, image_path

Qdrant runs in local mode (no server) persisting to paths.qdrant_db_dir.
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("qdrant_indexer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class QdrantIndexer:
    """Embed PDF page images with ColQwen2 and store multi-vectors in Qdrant."""

    def __init__(self, cfg: dict, wandb_run=None):
        self.cfg = cfg
        self.wandb_run = wandb_run
        qdrant_cfg = cfg["qdrant"]
        self.collection_name = qdrant_cfg["collection_name"]
        self.vector_size = qdrant_cfg["vector_size"]
        db_dir = cfg["paths"]["qdrant_db_dir"]

        # Load ColQwen2
        model_name = cfg["models"]["colpali"]
        device = cfg["models"].get("colpali_device", "cuda")
        log.info(f"Loading ColQwen2: {model_name} on {device}")
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        import torch

        self.model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name)
        self.device = device

        # Init Qdrant local client
        log.info(f"Opening Qdrant at {db_dir}")
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, MultiVectorConfig, MultiVectorComparator

        Path(db_dir).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=db_dir)

        # Create collection if it doesn't exist
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                ),
            )
            log.info(f"Created Qdrant collection: {self.collection_name}")
        else:
            log.info(f"Using existing Qdrant collection: {self.collection_name}")

    def index_pages(self, pages_meta: list[dict[str, Any]]) -> int:
        """
        Embed and index all page images. Returns number of pages indexed.
        pages_meta: list from PageRenderer.render_directory()
        """
        import torch
        from PIL import Image
        from qdrant_client.models import PointStruct

        log.info(f"Indexing {len(pages_meta)} pages into Qdrant …")
        t0 = time.perf_counter()
        indexed = 0
        points = []

        for i, page in enumerate(pages_meta):
            try:
                pil_img = Image.open(page["image_path"]).convert("RGB")
                batch = self.processor.process_images([pil_img]).to(self.device)

                with torch.no_grad():
                    embeddings = self.model(**batch)  # shape: (1, num_patches, 128)

                # Convert to list of 128-dim vectors (one per patch)
                patch_vectors = embeddings[0].float().cpu().tolist()

                point = PointStruct(
                    id=i,
                    vector=patch_vectors,
                    payload={
                        "pdf_stem": page["pdf_stem"],
                        "filename": page["filename"],
                        "page_num": page["page_num"],
                        "image_path": page["image_path"],
                        "width": page.get("width", 0),
                        "height": page.get("height", 0),
                    },
                )
                points.append(point)
                indexed += 1

                if (i + 1) % 10 == 0:
                    log.info(f"  Embedded {i+1}/{len(pages_meta)} pages …")

            except Exception as exc:
                log.warning(f"Failed to embed page {page.get('image_path')}: {exc}")

        # Upsert all points
        BATCH = 50
        for i in range(0, len(points), BATCH):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i: i + BATCH],
            )

        elapsed = time.perf_counter() - t0
        log.info(f"Indexed {indexed} pages in {elapsed:.1f}s")

        if self.wandb_run:
            self.wandb_run.log({
                "index_b/pages_indexed": indexed,
                "index_b/elapsed_sec": elapsed,
                "index_b/throughput_pages_per_min": indexed / max(elapsed / 60, 0.001),
            })
        return indexed