"""
page_renderer.py — Approach B: Render PDF pages as PNG images for ColQwen2.

Renders every page of every PDF at 150 DPI using PyMuPDF (fitz).
Saves PNGs to: data/processed/pages/{pdf_stem}/page_{N:04d}.png
Returns metadata list consumed by qdrant_indexer.py.
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("page_renderer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class PageRenderer:
    """Render PDF pages as PNG images using PyMuPDF."""

    def __init__(self, cfg: dict, wandb_run=None):
        self.cfg = cfg
        self.wandb_run = wandb_run
        self.dpi = cfg["extraction"].get("page_render_dpi", 150)
        self.pages_dir = Path(cfg["paths"]["pages_dir"])
        self.pages_dir.mkdir(parents=True, exist_ok=True)

    def render_pdf(self, pdf_path: str | Path) -> list[dict[str, Any]]:
        """Render all pages of a single PDF. Returns list of page metadata dicts."""
        import fitz  # PyMuPDF

        pdf_path = Path(pdf_path)
        pdf_stem = pdf_path.stem
        out_dir = self.pages_dir / pdf_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)  # 72 DPI is PDF default
        pages_meta: list[dict[str, Any]] = []
        t0 = time.perf_counter()

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat)
            img_path = out_dir / f"page_{page_num + 1:04d}.png"
            pix.save(str(img_path))
            pages_meta.append({
                "pdf_stem": pdf_stem,
                "filename": pdf_path.name,
                "page_num": page_num + 1,
                "image_path": str(img_path),
                "width": pix.width,
                "height": pix.height,
            })

        doc.close()
        elapsed = time.perf_counter() - t0
        log.info(f"  {pdf_path.name}: {len(pages_meta)} pages rendered in {elapsed:.1f}s")

        if self.wandb_run:
            self.wandb_run.log({
                f"render/{pdf_stem}/pages": len(pages_meta),
                f"render/{pdf_stem}/elapsed_sec": elapsed,
            })
        return pages_meta

    def render_directory(self, pdf_dir: str | Path) -> list[dict[str, Any]]:
        """Render all PDFs in a directory. Returns combined page metadata list."""
        pdf_dir = Path(pdf_dir)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        log.info(f"Rendering {len(pdf_files)} PDFs at {self.dpi} DPI …")
        all_pages: list[dict[str, Any]] = []
        for pdf_path in pdf_files:
            all_pages.extend(self.render_pdf(pdf_path))
        log.info(f"Total pages rendered: {len(all_pages)}")
        if self.wandb_run:
            self.wandb_run.log({
                "render/total_pdfs": len(pdf_files),
                "render/total_pages": len(all_pages),
            })
        return all_pages