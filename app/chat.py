"""
chat.py — Gradio chat UI for colpali-multimodal-rag. Runs on laptop, no GPU.

Usage:
  cp .env.example .env  # fill in GEMINI_API_KEY
  python app/chat.py
  # Open http://localhost:7860
"""

from __future__ import annotations
import base64
import os
import sys
import time
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import yaml
import gradio as gr
from PIL import Image as PILImage

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _b64_to_pil(b64: str) -> PILImage.Image | None:
    try:
        return PILImage.open(BytesIO(base64.b64decode(b64)))
    except Exception:
        return None


# ── Lazy-load retrievers and generator once ───────────────────────────────────
_cfg = None
_retriever_a = None
_retriever_b = None
_generator = None


def get_components():
    global _cfg, _retriever_a, _retriever_b, _generator
    if _cfg is None:
        _cfg = load_config()
        # Force CPU for laptop
        _cfg["models"]["text_embedder_device"] = "cpu"
        _cfg["models"]["colpali_device"] = "cpu"

        from src.retrievers.hybrid_retriever import HybridRetriever
        from src.retrievers.colpali_retriever import ColPaliRetriever
        from src.generator import GeminiGenerator

        _retriever_a = HybridRetriever(_cfg)
        _retriever_b = ColPaliRetriever(_cfg)
        _generator = GeminiGenerator(_cfg)
    return _cfg, _retriever_a, _retriever_b, _generator


def answer_question(question: str, approach: str, history: list):
    """Main chat handler — returns updated history + images."""
    if not question.strip():
        return history, []

    cfg, retriever_a, retriever_b, generator = get_components()
    t0 = time.perf_counter()
    images_out = []

    if approach == "Approach A — Structured (Docling + ChromaDB)":
        ctx = retriever_a.retrieve(question)
        result = generator.generate_approach_a(question, ctx)
        answer = result["answer"]

        # Collect figure images for display
        for fig in ctx.get("figure_results", []):
            b64 = fig.get("image_base64", "")
            if b64:
                pil = _b64_to_pil(b64)
                if pil:
                    images_out.append(pil)

        meta = (
            f"📚 Sources: {len(ctx['text_results'])} text, "
            f"{len(ctx['figure_results'])} figures, "
            f"{len(ctx['table_results'])} tables | "
            f"💰 ${result['cost_usd']:.5f} | "
            f"⏱ {(time.perf_counter()-t0)*1000:.0f}ms"
        )

    else:  # Approach B
        pages = retriever_b.retrieve(question)
        result = generator.generate_approach_b(question, pages)
        answer = result["answer"]

        # Show retrieved page images
        for page in pages:
            img_path = page.get("image_path", "")
            if img_path and Path(img_path).exists():
                try:
                    images_out.append(PILImage.open(img_path))
                except Exception:
                    pass

        meta = (
            f"📄 Pages retrieved: {len(pages)} | "
            f"💰 ${result['cost_usd']:.5f} | "
            f"⏱ {(time.perf_counter()-t0)*1000:.0f}ms"
        )

    full_answer = f"{answer}\n\n*{meta}*"
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": full_answer})
    return history, images_out


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="ColPali Multimodal RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """# 🔬 ColPali Multimodal RAG
        **Compare two retrieval architectures on 10 foundational AI papers.**
        - **Approach A**: Docling extracts structure → Qwen2-VL describes figures → ChromaDB hybrid search
        - **Approach B**: ColQwen2 embeds whole pages as patch vectors → Qdrant MaxSim → Gemini reads images
        """
    )

    with gr.Row():
        approach_selector = gr.Radio(
            choices=[
                "Approach A — Structured (Docling + ChromaDB)",
                "Approach B — Visual (ColQwen2 + Qdrant)",
            ],
            value="Approach A — Structured (Docling + ChromaDB)",
            label="Retrieval Approach",
        )

    chatbot = gr.Chatbot(type="messages", height=500, label="Chat")
    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Ask anything about the 10 AI papers…",
            label="Your question",
            scale=5,
        )
        send_btn = gr.Button("Ask", variant="primary", scale=1)

    image_gallery = gr.Gallery(label="Retrieved figures / pages", columns=3, height=300)
    clear_btn = gr.Button("Clear chat")

    send_btn.click(
        answer_question,
        inputs=[msg_box, approach_selector, chatbot],
        outputs=[chatbot, image_gallery],
    ).then(lambda: "", outputs=msg_box)

    msg_box.submit(
        answer_question,
        inputs=[msg_box, approach_selector, chatbot],
        outputs=[chatbot, image_gallery],
    ).then(lambda: "", outputs=msg_box)

    clear_btn.click(lambda: ([], []), outputs=[chatbot, image_gallery])

    gr.Examples(
        examples=[
            "What is the scaled dot-product attention formula?",
            "How does BERT use masked language modeling?",
            "What makes ResNet skip connections effective?",
            "How does LoRA reduce trainable parameters?",
            "What is the training objective of GANs?",
        ],
        inputs=msg_box,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)