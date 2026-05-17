"""
chat.py — Gradio 6.14 compatible UI for colpali-multimodal-rag.
"""

from __future__ import annotations
import base64
import sys
import time
from io import BytesIO
from pathlib import Path
import os
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import yaml
import gradio as gr
from PIL import Image as PILImage

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"
os.chdir(Path(__file__).parent.parent)

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["models"]["text_embedder_device"] = "cpu"
    return cfg


def b64_to_pil(b64: str):
    try:
        return PILImage.open(BytesIO(base64.b64decode(b64)))
    except Exception:
        return None


class LightweightColPaliRetriever:
    def __init__(self, cfg: dict):
        self.pages_dir = Path(cfg["paths"]["pages_dir"])
        self.top_k = cfg["qdrant"].get("top_k_pages", 3)

        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(cfg["models"]["text_embedder"], device="cpu")

        import json, numpy as np
        chroma_dir = Path(cfg["paths"]["chroma_db_dir"])
        npz    = chroma_dir / "approach_a_text.npz"
        json_f = chroma_dir / "approach_a_text_docs.json"
        if npz.exists() and json_f.exists():
            self.embeddings = np.load(str(npz))["embeddings"].astype("float32")
            self.docs = json.load(open(json_f))
        else:
            self.embeddings = None
            self.docs = []

    def retrieve(self, query: str) -> list[dict]:
        import numpy as np, faiss
        if self.embeddings is None:
            return []
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        idx = faiss.IndexFlatIP(self.embeddings.shape[1])
        idx.add(self.embeddings)
        scores, indices = idx.search(q, min(10, idx.ntotal))
        seen, results = set(), []
        for score, i in zip(scores[0], indices[0]):
            if i < 0:
                continue
            doc = self.docs[i]
            fname, page = doc.get("filename", ""), doc.get("page_num", 1)
            if (fname, page) in seen:
                continue
            seen.add((fname, page))
            stem = Path(fname).stem
            img = self.pages_dir / stem / f"page_{page:04d}.png"
            if img.exists():
                results.append({"filename": fname, "page_num": page,
                                "image_path": str(img), "score": float(score)})
            if len(results) >= self.top_k:
                break
        return results


_cfg = _ret_a = _ret_b = _gen = None
_ready = False
_status = "Click 'Load Models' to start"


def load_components():
    global _cfg, _ret_a, _ret_b, _gen, _ready, _status
    if _ready:
        return _status
    try:
        _status = "Loading config...";          _cfg = load_config()
        _status = "Loading Approach A (FAISS)..."
        from src.retrievers.hybrid_retriever import HybridRetriever
        _ret_a = HybridRetriever(_cfg)
        _status = "Loading Approach B..."
        _ret_b = LightweightColPaliRetriever(_cfg)
        _status = "Connecting to Gemini..."
        from src.generator import GeminiGenerator
        _gen = GeminiGenerator(_cfg)
        _ready = True;  _status = "✅ Ready!"
    except Exception as e:
        _status = f"❌ {e}"
    return _status



def chat(message, history, approach, gallery_images):
    if not message.strip():
        return history, gallery_images

    status = load_components()
    if not _ready:
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": f"⚠️ {status}"})
        return history, gallery_images

    t0 = time.perf_counter()
    images = []

    if "Approach A" in approach:
        ctx    = _ret_a.retrieve(message)
        result = _gen.generate_approach_a(message, ctx)
        for fig in ctx.get("figure_results", []):
            pil = b64_to_pil(fig.get("image_base64", ""))
            if pil:
                images.append(pil)
        elapsed = (time.perf_counter() - t0) * 1000
        meta = (f"📚 {len(ctx['text_results'])} text · "
                f"{len(ctx['figure_results'])} figs · "
                f"{len(ctx['table_results'])} tables · "
                f"⏱ {elapsed:.0f}ms · 💰 ${result['cost_usd']:.5f}")
    else:
        pages  = _ret_b.retrieve(message)
        result = _gen.generate_approach_b(message, pages)
        for p in pages:
            ip = p.get("image_path", "")
            if ip and Path(ip).exists():
                try:
                    images.append(PILImage.open(ip))
                except Exception:
                    pass
        elapsed = (time.perf_counter() - t0) * 1000
        meta = (f"📄 {len(pages)} pages · "
                f"⏱ {elapsed:.0f}ms · 💰 ${result['cost_usd']:.5f}")

    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": f"{result['answer']}\n\n*{meta}*"})
    all_images = (gallery_images or []) + images
    return history, all_images


EXAMPLES = [
    "What is the scaled dot-product attention formula?",
    "How does BERT use masked language modeling?",
    "How does LoRA reduce trainable parameters?",
    "What is classifier-free guidance in Stable Diffusion?",
    "How does DALL-E 2 use CLIP embeddings?",
    "What training data did LLaMA use?",
]

with gr.Blocks(title="ColPali Multimodal RAG") as demo:
    gr.Markdown("""# 🔬 ColPali Multimodal RAG
**Two retrieval architectures on 10 foundational AI papers.**

| | Approach A | Approach B |
|---|---|---|
| Index | FAISS (text + figures + tables) | Qdrant page images |
| Hit Rate | 60% | 80% |
| Latency | ~272ms | ~600ms |
""")

    with gr.Row():
        approach = gr.Radio(
            choices=["Approach A — Structured (FAISS)", "Approach B — Visual (page images)"],
            value="Approach A — Structured (FAISS)",
            label="Retrieval Approach",
        )
        with gr.Column():
            status = gr.Textbox(value=_status, label="Status", interactive=False)
            load_btn = gr.Button("Load Models", variant="secondary")

    chatbot  = gr.Chatbot(height=450, label="Chat")
    msg      = gr.Textbox(placeholder="Ask about the 10 AI papers…", label="Question")
    with gr.Row():
        send  = gr.Button("Ask ✨", variant="primary")
        clear = gr.Button("🗑 Clear")

    gallery = gr.Gallery(label="Retrieved figures / pages", columns=3, height=250)

    gr.Examples(examples=EXAMPLES, inputs=msg)

    load_btn.click(load_components, outputs=status)
    send.click(chat, inputs=[msg, chatbot, approach, gallery], outputs=[chatbot, gallery]).then(lambda: "", outputs=msg)
    msg.submit(chat, inputs=[msg, chatbot, approach, gallery], outputs=[chatbot, gallery]).then(lambda: "", outputs=msg)
    clear.click(lambda: ([], []), outputs=[chatbot, gallery])


if __name__ == "__main__":
    print("Open: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
