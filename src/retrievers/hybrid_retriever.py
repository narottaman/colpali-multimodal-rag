"""
hybrid_retriever.py — Approach A: FAISS-based search over saved .npz index files.

Loads embeddings + metadata saved by chroma_indexer.py (no sqlite needed).
Uses FAISS for fast cosine similarity search (dot product on normalized vecs).
Searches text, figure, and table collections and merges results.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

log = logging.getLogger("hybrid_retriever")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class HybridRetriever:
    """
    FAISS-based retriever over three collections saved by ChromaIndexer.
    No sqlite. No chromadb dependency at retrieval time.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.chroma_cfg = cfg["chroma"]
        self.db_dir = Path(cfg["paths"]["chroma_db_dir"])
        self.top_k_text    = self.chroma_cfg.get("top_k_text", 3)
        self.top_k_figures = self.chroma_cfg.get("top_k_figures", 2)
        self.top_k_tables  = self.chroma_cfg.get("top_k_tables", 2)

        model_name = cfg["models"]["text_embedder"]
        device     = cfg["models"].get("text_embedder_device", "cpu")
        log.info(f"Loading embedder: {model_name} on {device}")
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(model_name, device=device)

        # Load all three FAISS indexes
        self._indexes: dict[str, Any] = {}
        self._metadata: dict[str, list] = {}
        for col_name in [
            self.chroma_cfg["text_collection"],
            self.chroma_cfg["figure_collection"],
            self.chroma_cfg["table_collection"],
        ]:
            self._load_collection(col_name)

        log.info("HybridRetriever ready.")

    def _load_collection(self, collection_name: str):
        """Load a .npz embedding file + .json metadata file into a FAISS index."""
        import faiss

        npz_path  = self.db_dir / f"{collection_name}.npz"
        json_path = self.db_dir / f"{collection_name}_docs.json"

        if not npz_path.exists() or not json_path.exists():
            log.warning(f"Collection not found on disk: {collection_name} — skipping")
            self._indexes[collection_name]  = None
            self._metadata[collection_name] = []
            return

        embeddings = np.load(str(npz_path))["embeddings"].astype(np.float32)
        with open(json_path) as f:
            metadata = json.load(f)

        # Build FAISS flat index (exact cosine via inner product on normalized vecs)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self._indexes[collection_name]  = index
        self._metadata[collection_name] = metadata
        log.info(f"  Loaded {collection_name}: {index.ntotal} vectors (dim={dim})")

    def retrieve(self, query: str) -> dict[str, Any]:
        """
        Embed query, search all three collections, return merged context.
        """
        query_vec = self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)  # shape (1, dim)

        text_res  = self._search(self.chroma_cfg["text_collection"],   query_vec, self.top_k_text)
        fig_res   = self._search(self.chroma_cfg["figure_collection"],  query_vec, self.top_k_figures)
        table_res = self._search(self.chroma_cfg["table_collection"],   query_vec, self.top_k_tables)

        context_parts = []
        for r in text_res:
            context_parts.append(f"[Text | {r['filename']} p.{r['page_num']}]\n{r['text']}")
        for r in fig_res:
            context_parts.append(f"[Figure | {r['filename']} p.{r['page_num']}] {r.get('caption','')}\n{r['text']}")
        for r in table_res:
            context_parts.append(f"[Table | {r['filename']} p.{r['page_num']}]\n{r.get('markdown_table', r['text'])}")

        return {
            "text_results":   text_res,
            "figure_results": fig_res,
            "table_results":  table_res,
            "context_str":    "\n\n---\n\n".join(context_parts),
            "has_figures":    len(fig_res) > 0,
            "has_tables":     len(table_res) > 0,
        }

    def _search(self, collection_name: str, query_vec: np.ndarray, k: int) -> list[dict]:
        """FAISS inner-product search against one collection."""
        index    = self._indexes.get(collection_name)
        metadata = self._metadata.get(collection_name, [])
        if index is None or index.ntotal == 0:
            return []

        k = min(k, index.ntotal)
        scores, indices = index.search(query_vec, k)  # both shape (1, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS returns -1 for empty slots
            row = dict(metadata[idx])
            row["score"] = float(score)
            results.append(row)
        return results