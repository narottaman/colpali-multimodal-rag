"""
chroma_indexer.py — Approach A: Index elements using ChromaDB EphemeralClient + FAISS.

WHY NOT PersistentClient:
  Sol's /etc/python/sitecustomize.py loads Mamba's old libsqlite3.so before
  any venv code can intercept it. PersistentClient requires sqlite ≥ 3.23.0.
  EphemeralClient is pure in-memory — zero sqlite dependency.

PERSISTENCE STRATEGY:
  After indexing, we save everything to disk as .npz (embeddings) + .json
  (documents + metadata) using numpy. At retrieval time, HybridRetriever
  reloads these files and uses FAISS for ANN search — same query results,
  no sqlite.

Collections saved to paths.chroma_db_dir/:
  approach_a_text.npz   + approach_a_text_docs.json
  approach_a_figures.npz + approach_a_figures_docs.json
  approach_a_tables.npz  + approach_a_tables_docs.json
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

log = logging.getLogger("chroma_indexer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class ChromaIndexer:
    """
    Index unified element dicts using ChromaDB EphemeralClient for embedding,
    then persist to FAISS-compatible .npz + .json files for retrieval without sqlite.
    """

    def __init__(self, cfg: dict, wandb_run=None):
        self.cfg = cfg
        self.wandb_run = wandb_run
        self.chroma_cfg = cfg["chroma"]
        self.db_dir = Path(cfg["paths"]["chroma_db_dir"])
        self.db_dir.mkdir(parents=True, exist_ok=True)

        model_name = cfg["models"]["text_embedder"]
        device = cfg["models"].get("text_embedder_device", "cpu")
        log.info(f"Loading sentence-transformer: {model_name} on {device}")

        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(model_name, device=device)
        log.info("Embedder ready.")

    def index(self, elements: list[dict[str, Any]]) -> dict[str, int]:
        """
        Embed all elements and save to disk.
        Returns counts: {text, figure, table}.
        """
        text_elems   = [e for e in elements if e["element_type"] == "text"]
        fig_elems    = [e for e in elements if e["element_type"] == "figure"]
        table_elems  = [e for e in elements if e["element_type"] == "table"]

        t0 = time.perf_counter()
        counts = {}
        counts["text"]   = self._embed_and_save(text_elems,  self.chroma_cfg["text_collection"],   include_image=False)
        counts["figure"] = self._embed_and_save(fig_elems,   self.chroma_cfg["figure_collection"], include_image=True)
        counts["table"]  = self._embed_and_save(table_elems, self.chroma_cfg["table_collection"],  include_image=False)

        elapsed = time.perf_counter() - t0
        log.info(f"Indexed — text:{counts['text']} figures:{counts['figure']} tables:{counts['table']} in {elapsed:.1f}s")

        if self.wandb_run:
            self.wandb_run.log({
                "index/text_docs":    counts["text"],
                "index/figure_docs":  counts["figure"],
                "index/table_docs":   counts["table"],
                "index/total_docs":   sum(counts.values()),
                "index/elapsed_sec":  elapsed,
            })
        return counts

    def _embed_and_save(self, elements: list[dict], collection_name: str, include_image: bool) -> int:
        """Embed a list of elements and save embeddings + metadata to disk."""
        if not elements:
            log.info(f"  {collection_name}: 0 elements, skipping")
            return 0

        texts = [e["text"] for e in elements]
        log.info(f"  Embedding {len(texts)} docs for {collection_name} …")

        # Batch embed — sentence_transformers handles batching internally
        embeddings = self.embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine sim = dot product after normalization
        )  # shape: (N, embedding_dim)

        # Build metadata list (strip image_base64 from non-figure collections)
        metadata = []
        for e in elements:
            meta = {
                "id":             e["id"],
                "text":           e["text"],
                "element_type":   e["element_type"],
                "filename":       e["filename"],
                "page_num":       e["page_num"],
                "title":          e["title"],
                "caption":        e["caption"],
                "chunk_method":   e["chunk_method"],
                "chunk_size":     e["chunk_size"],
                "markdown_table": e["markdown_table"],
                "json_rows":      e["json_rows"],
            }
            if include_image:
                meta["image_base64"] = e.get("image_base64", "")
                meta["image_width"]  = e.get("image_width", 0)
                meta["image_height"] = e.get("image_height", 0)
            metadata.append(meta)

        # Save embeddings as .npz
        npz_path = self.db_dir / f"{collection_name}.npz"
        np.savez_compressed(str(npz_path), embeddings=embeddings.astype(np.float32))

        # Save metadata as .json
        json_path = self.db_dir / f"{collection_name}_docs.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)

        log.info(f"  {collection_name}: saved {len(elements)} docs → {npz_path.name} + {json_path.name}")
        return len(elements)