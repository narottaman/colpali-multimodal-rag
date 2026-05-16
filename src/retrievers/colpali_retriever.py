"""
colpali_retriever.py — Approach B: ColQwen2 late interaction retrieval from Qdrant.

Embeds the query as multi-vector patch embeddings using ColQwen2,
then queries Qdrant with MaxSim scoring to find the most visually relevant pages.
Returns top-k page image paths for Gemini to read directly.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("colpali_retriever")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class ColPaliRetriever:
    """Late interaction retrieval using ColQwen2 embeddings and Qdrant MaxSim."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        qdrant_cfg = cfg["qdrant"]
        self.collection_name = qdrant_cfg["collection_name"]
        self.top_k = qdrant_cfg.get("top_k_pages", 3)
        db_dir = cfg["paths"]["qdrant_db_dir"]
        device = cfg["models"].get("colpali_device", "cpu")  # CPU on laptop

        log.info(f"Loading ColQwen2 for retrieval on {device} …")
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        import torch

        model_name = cfg["models"]["colpali"]
        self.model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            device_map=device,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name)
        self.device = device

        from qdrant_client import QdrantClient
        self.client = QdrantClient(path=db_dir)
        log.info("ColPaliRetriever ready.")

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """
        Embed query and return top-k page metadata dicts.
        Each dict: {pdf_stem, filename, page_num, image_path, score}
        """
        import torch

        batch = self.processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            query_embeddings = self.model(**batch)  # (1, num_query_patches, 128)

        query_vectors = query_embeddings[0].float().cpu().tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vectors,
            limit=self.top_k,
            with_payload=True,
        )

        output = []
        for hit in results.points:
            payload = hit.payload or {}
            output.append({
                "pdf_stem": payload.get("pdf_stem", ""),
                "filename": payload.get("filename", ""),
                "page_num": payload.get("page_num", 0),
                "image_path": payload.get("image_path", ""),
                "score": hit.score,
            })
        return output