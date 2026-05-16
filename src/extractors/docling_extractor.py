"""
docling_extractor.py — Approach A: Structured extraction from research PDFs.

What this does:
  1. Accepts a list of PDF paths (or a directory).
  2. Runs each PDF through Docling's DocumentConverter with table structure
     analysis and picture image generation enabled.
  3. Extracts three element types from each document:
       - TEXT:   paragraph chunks (split by max_chunk_chars with overlap)
       - TABLE:  markdown rendering + JSON rows + caption
       - FIGURE: PIL image (as base64) + caption + page number
  4. Returns a flat list of typed dicts with a unified schema compatible
     with both chroma_indexer.py (Approach A) and future indexers.
  5. Logs per-document and aggregate extraction stats to W&B.

Unified element schema (TypedDict):
  id              str   — "{filename}_{element_type}_{sequential_index}"
  text            str   — main text to embed (paragraph / VLM desc / table summary)
  element_type    str   — "text" | "figure" | "table"
  title           str   — section heading above this element (best effort)
  filename        str   — source PDF filename (no path)
  page_num        int   — 1-indexed page number
  chunk_method    str   — "docling_paragraph" | "docling_table" | "docling_figure"
  chunk_size      int   — len(text) in chars
  caption         str   — figure/table caption if found, else ""
  markdown_table  str   — markdown table string (tables only, else "")
  json_rows       list  — list of row dicts (tables only, else [])
  image_base64    str   — base64-encoded PNG (figures only, else "")
  image_width     int   — pixel width (figures only, else 0)
  image_height    int   — pixel height (figures only, else 0)

Usage (standalone test):
  python -m src.extractors.docling_extractor \
      --pdf /path/to/paper.pdf \
      --config configs/config.yaml \
      --wandb_run test_extraction
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Logging setup — structured, no noise
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("docling_extractor")


# ---------------------------------------------------------------------------
# Lazy imports — only fail at runtime if missing, not at import time
# ---------------------------------------------------------------------------
def _import_docling():
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        return DocumentConverter, PdfFormatOption, PdfPipelineOptions, InputFormat
    except ImportError as e:
        log.error("docling not found. Run: pip install docling")
        raise e


def _import_wandb():
    try:
        import wandb
        return wandb
    except ImportError:
        log.warning("wandb not installed — stats will be logged to console only.")
        return None


def _import_pil():
    try:
        from PIL import Image
        return Image
    except ImportError as e:
        log.error("Pillow not found. Run: pip install pillow")
        raise e


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(config_path: str | Path) -> dict:
    """Load configs/config.yaml and return as nested dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Text chunking helper
# ---------------------------------------------------------------------------
def _chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    Split a long paragraph into overlapping chunks of at most max_chars.
    Uses sentence-aware splitting when possible (split on '. ').
    Falls back to hard char split.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # Try to split at last sentence boundary within window
        split_at = text.rfind(". ", start, end)
        if split_at == -1 or split_at <= start:
            split_at = end  # hard split
        else:
            split_at += 1  # include the period
        chunks.append(text[start:split_at].strip())
        start = split_at - overlap  # overlap for context continuity
        if start < 0:
            start = 0
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# PIL image → base64 PNG string
# ---------------------------------------------------------------------------
def _pil_to_base64(pil_image) -> tuple[str, int, int]:
    """Convert PIL Image to base64 PNG. Returns (b64_str, width, height)."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, pil_image.width, pil_image.height


# ---------------------------------------------------------------------------
# Core extractor class
# ---------------------------------------------------------------------------
class DoclingExtractor:
    """
    Extracts text paragraphs, tables, and figures from a PDF using Docling.

    Args:
        cfg: Parsed config.yaml dict (from load_config).
        wandb_run: An active wandb.Run object, or None for no W&B logging.
    """

    def __init__(self, cfg: dict, wandb_run=None):
        self.cfg = cfg
        self.wandb_run = wandb_run
        self.extraction_cfg = cfg.get("extraction", {})
        self.max_chunk_chars = self.extraction_cfg.get("max_chunk_chars", 1200)
        self.chunk_overlap = self.extraction_cfg.get("chunk_overlap_chars", 150)
        self.min_para_chars = self.extraction_cfg.get("min_paragraph_chars", 80)

        log.info("Initializing Docling DocumentConverter …")
        DocumentConverter, PdfFormatOption, PdfPipelineOptions, InputFormat = _import_docling()

        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.do_table_structure = self.extraction_cfg.get("do_table_structure", True)
        pipeline_opts.generate_picture_images = self.extraction_cfg.get("generate_picture_images", True)

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
            }
        )
        log.info("DocumentConverter ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_pdf(self, pdf_path: str | Path) -> list[dict[str, Any]]:
        """
        Extract all elements from a single PDF.

        Returns:
            List of element dicts matching the unified schema.
        """
        pdf_path = Path(pdf_path)
        filename = pdf_path.name
        log.info(f"Extracting: {filename}")

        t0 = time.perf_counter()
        try:
            result = self.converter.convert(str(pdf_path))
        except Exception as exc:
            log.error(f"Docling conversion failed for {filename}: {exc}")
            return []

        doc = result.document
        elements: list[dict[str, Any]] = []
        counters = {"text": 0, "table": 0, "figure": 0, "skipped": 0}

        # --- Helper: running section title tracker ---
        current_title = ""

        # ------------------------------------------------------------------
        # Iterate over document body items
        # ------------------------------------------------------------------
        for item, level in doc.iterate_items():
            try:
                item_type = type(item).__name__

                # ---- Section headings — update running title ----
                if item_type in ("SectionHeaderItem", "TitleItem", "HeadingItem"):
                    try:
                        current_title = item.text.strip()
                    except Exception:
                        pass
                    continue

                # ---- Text paragraphs ----
                if item_type == "TextItem":
                    raw_text = item.text.strip() if hasattr(item, "text") else ""
                    if len(raw_text) < self.min_para_chars:
                        counters["skipped"] += 1
                        continue
                    page_num = self._get_page_num(item)
                    chunks = _chunk_text(raw_text, self.max_chunk_chars, self.chunk_overlap)
                    for chunk in chunks:
                        elem = self._make_element(
                            text=chunk,
                            element_type="text",
                            title=current_title,
                            filename=filename,
                            page_num=page_num,
                            chunk_method="docling_paragraph",
                        )
                        elements.append(elem)
                        counters["text"] += 1

                # ---- Tables ----
                elif item_type == "TableItem":
                    caption = self._get_caption(item)
                    page_num = self._get_page_num(item)
                    markdown_table = ""
                    json_rows: list[dict] = []
                    try:
                        markdown_table = item.export_to_markdown()
                    except Exception:
                        pass
                    try:
                        df = item.export_to_dataframe()
                        json_rows = json.loads(df.to_json(orient="records"))
                    except Exception:
                        pass

                    # Embed the markdown (summary text) — Qwen VLM not used for tables
                    embed_text = f"Table: {caption}\n{markdown_table}" if caption else markdown_table
                    if not embed_text.strip():
                        counters["skipped"] += 1
                        continue

                    elem = self._make_element(
                        text=embed_text,
                        element_type="table",
                        title=current_title,
                        filename=filename,
                        page_num=page_num,
                        chunk_method="docling_table",
                        caption=caption,
                        markdown_table=markdown_table,
                        json_rows=json_rows,
                    )
                    elements.append(elem)
                    counters["table"] += 1

                # ---- Figures / Pictures ----
                elif item_type == "PictureItem":
                    caption = self._get_caption(item)
                    page_num = self._get_page_num(item)
                    image_base64 = ""
                    image_width = 0
                    image_height = 0

                    try:
                        pil_img = item.get_image(doc)
                        if pil_img is not None:
                            image_base64, image_width, image_height = _pil_to_base64(pil_img)
                    except Exception as img_err:
                        log.debug(f"Could not get image for figure in {filename}: {img_err}")

                    # text field will be filled later by qwen_vl_describer;
                    # store caption as placeholder so element is indexable now.
                    placeholder_text = caption if caption else f"Figure on page {page_num}"

                    elem = self._make_element(
                        text=placeholder_text,
                        element_type="figure",
                        title=current_title,
                        filename=filename,
                        page_num=page_num,
                        chunk_method="docling_figure",
                        caption=caption,
                        image_base64=image_base64,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    elements.append(elem)
                    counters["figure"] += 1

            except Exception as item_err:
                log.warning(f"Skipping element in {filename} due to error: {item_err}")
                counters["skipped"] += 1
                continue

        elapsed = time.perf_counter() - t0
        log.info(
            f"  {filename}: {counters['text']} text, {counters['table']} tables, "
            f"{counters['figure']} figures, {counters['skipped']} skipped — {elapsed:.1f}s"
        )

        # --- W&B per-document stats ---
        if self.wandb_run is not None:
            self.wandb_run.log({
                f"extraction/{filename}/text_chunks": counters["text"],
                f"extraction/{filename}/tables": counters["table"],
                f"extraction/{filename}/figures": counters["figure"],
                f"extraction/{filename}/skipped": counters["skipped"],
                f"extraction/{filename}/elapsed_sec": elapsed,
            })

        return elements

    def extract_directory(self, pdf_dir: str | Path) -> list[dict[str, Any]]:
        """
        Extract all PDFs in a directory. Logs aggregate stats to W&B.

        Returns:
            Combined flat list of elements from all PDFs.
        """
        pdf_dir = Path(pdf_dir)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            log.warning(f"No PDFs found in {pdf_dir}")
            return []

        log.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")
        all_elements: list[dict[str, Any]] = []
        agg = {"text": 0, "table": 0, "figure": 0, "skipped": 0}

        for pdf_path in pdf_files:
            elems = self.extract_pdf(pdf_path)
            all_elements.extend(elems)
            for e in elems:
                agg[e["element_type"]] += 1

        log.info(
            f"Aggregate totals — text: {agg['text']}, tables: {agg['table']}, "
            f"figures: {agg['figure']} — {len(all_elements)} total elements"
        )

        if self.wandb_run is not None:
            self.wandb_run.log({
                "extraction/total_pdfs": len(pdf_files),
                "extraction/total_text_chunks": agg["text"],
                "extraction/total_tables": agg["table"],
                "extraction/total_figures": agg["figure"],
                "extraction/total_elements": len(all_elements),
            })

        return all_elements

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _make_element(
        self,
        text: str,
        element_type: str,
        title: str,
        filename: str,
        page_num: int,
        chunk_method: str,
        caption: str = "",
        markdown_table: str = "",
        json_rows: list = None,
        image_base64: str = "",
        image_width: int = 0,
        image_height: int = 0,
    ) -> dict[str, Any]:
        """Build a unified element dict."""
        return {
            "id": f"{Path(filename).stem}_{element_type}_{uuid.uuid4().hex[:8]}",
            "text": text,
            "element_type": element_type,
            "title": title,
            "filename": filename,
            "page_num": page_num,
            "chunk_method": chunk_method,
            "chunk_size": len(text),
            "caption": caption,
            "markdown_table": markdown_table,
            "json_rows": json_rows or [],
            "image_base64": image_base64,
            "image_width": image_width,
            "image_height": image_height,
        }

    @staticmethod
    def _get_page_num(item) -> int:
        """Extract 1-indexed page number from a Docling item. Returns 0 if unknown."""
        try:
            prov = item.prov
            if prov:
                return prov[0].page_no
        except Exception:
            pass
        return 0

    @staticmethod
    def _get_caption(item) -> str:
        """Extract caption text from a Docling table or figure item."""
        try:
            if hasattr(item, "captions") and item.captions:
                return item.captions[0].text.strip()
        except Exception:
            pass
        return ""


# ---------------------------------------------------------------------------
# CLI entry point — for testing a single PDF before batch processing
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test DoclingExtractor on one or more PDFs."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to a single PDF file to extract.",
    )
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default=None,
        help="Path to directory of PDFs (extracts all). Overrides --pdf.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--wandb_run",
        type=str,
        default=None,
        help="W&B run name. If omitted, W&B is disabled for this test.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: save extracted elements as JSON to this path.",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Optional W&B init
    wandb_run = None
    if args.wandb_run:
        wandb = _import_wandb()
        if wandb:
            wandb_run = wandb.init(
                project=cfg["wandb"]["project"],
                name=args.wandb_run,
                tags=cfg["wandb"]["base_tags"] + ["extraction_test"],
                config={
                    "extraction": cfg["extraction"],
                    "source": args.pdf or args.pdf_dir,
                },
            )

    extractor = DoclingExtractor(cfg=cfg, wandb_run=wandb_run)

    # Run extraction
    if args.pdf_dir:
        elements = extractor.extract_directory(args.pdf_dir)
    elif args.pdf:
        elements = extractor.extract_pdf(args.pdf)
    else:
        parser.error("Provide --pdf or --pdf_dir")

    # Print summary to console
    by_type = {"text": [], "figure": [], "table": []}
    for e in elements:
        by_type.get(e["element_type"], []).append(e)

    print("\n" + "=" * 60)
    print(f"EXTRACTION SUMMARY — {len(elements)} total elements")
    print("=" * 60)
    print(f"  Text chunks : {len(by_type['text'])}")
    print(f"  Tables      : {len(by_type['table'])}")
    print(f"  Figures     : {len(by_type['figure'])}")

    # Print one example of each type
    for etype, label in [("text", "TEXT"), ("table", "TABLE"), ("figure", "FIGURE")]:
        items = by_type[etype]
        if items:
            sample = items[0]
            print(f"\n--- Sample {label} ---")
            print(f"  id       : {sample['id']}")
            print(f"  filename : {sample['filename']}")
            print(f"  page_num : {sample['page_num']}")
            print(f"  title    : {sample['title'][:80]!r}")
            print(f"  text     : {sample['text'][:120]!r} …")
            if etype == "figure":
                print(f"  image_sz : {sample['image_width']}x{sample['image_height']}")
                print(f"  b64_len  : {len(sample['image_base64'])} chars")
            if etype == "table":
                print(f"  rows     : {len(sample['json_rows'])}")
                print(f"  markdown :\n{sample['markdown_table'][:200]}")

    # Optionally save to JSON
    if args.output:
        # Convert any non-serialisable objects (e.g., bytes) gracefully
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(elements, f, indent=2, default=str)
        print(f"\nSaved {len(elements)} elements → {out_path}")

    if wandb_run:
        wandb_run.finish()

    return elements


if __name__ == "__main__":
    main()