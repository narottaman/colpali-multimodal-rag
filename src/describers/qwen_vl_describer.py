"""
qwen_vl_describer.py — Approach A: Describe figures with Qwen2-VL-7B on Sol A100.

Takes figure elements from DoclingExtractor (element_type == "figure"),
generates 2-3 sentence descriptions via Qwen2-VL-7B-Instruct,
and overwrites element["text"] with the VLM description for embedding.
"""

from __future__ import annotations
import base64
import io
import logging
import time
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("qwen_vl_describer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


FIGURE_PROMPT = (
    "You are analyzing a figure extracted from an AI/ML research paper. "
    "Describe what this figure shows in exactly 2-3 sentences. "
    "Focus on: what type of figure it is (diagram, graph, table, architecture), "
    "what the axes or components represent, and the key insight or trend visible. "
    "Be specific and technical. Do not say 'the figure shows' — start directly with the content."
)


class QwenVLDescriber:
    """
    Runs Qwen2-VL-7B-Instruct on Sol A100 to describe extracted figures.
    Batch size 1 to avoid OOM on 7B model with high-res images.
    """

    def __init__(self, cfg: dict, wandb_run=None):
        self.cfg = cfg
        self.wandb_run = wandb_run
        model_name = cfg["models"]["qwen_vl"]
        device = cfg["models"].get("qwen_vl_device", "cuda")
        self.max_new_tokens = cfg["models"].get("qwen_vl_max_new_tokens", 200)

        log.info(f"Loading {model_name} on {device} …")
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = device
        log.info("Qwen2-VL loaded.")

    def describe_figures(self, elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        For every element with element_type == "figure" and a non-empty image_base64,
        call Qwen2-VL and overwrite element["text"] with the description.
        Returns the full element list (all types) with figures updated in place.
        """
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        figures = [e for e in elements if e["element_type"] == "figure" and e.get("image_base64")]
        log.info(f"Describing {len(figures)} figures …")
        t0 = time.perf_counter()
        desc_lengths = []

        for i, elem in enumerate(figures):
            try:
                # Decode base64 → PIL Image
                img_bytes = base64.b64decode(elem["image_base64"])
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": FIGURE_PROMPT},
                    ],
                }]

                text_input = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text_input],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )

                # Trim input tokens from output
                trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
                description = self.processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()

                elem["text"] = description
                desc_lengths.append(len(description))

                if (i + 1) % 5 == 0:
                    log.info(f"  Described {i+1}/{len(figures)} figures …")

            except Exception as exc:
                log.warning(f"Failed to describe figure {elem['id']}: {exc}")
                # Keep the caption placeholder — element remains indexable

        elapsed = time.perf_counter() - t0
        avg_len = sum(desc_lengths) / max(len(desc_lengths), 1)
        log.info(f"Done: {len(figures)} figures in {elapsed:.1f}s — avg desc {avg_len:.0f} chars")

        if self.wandb_run:
            self.wandb_run.log({
                "describe/figures_processed": len(figures),
                "describe/avg_description_chars": avg_len,
                "describe/elapsed_sec": elapsed,
                "describe/throughput_fig_per_min": len(figures) / (elapsed / 60),
            })

        return elements