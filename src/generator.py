"""
generator.py — Gemini 2.5 Flash Lite answer generator for both approaches.

Takes retrieved context (text chunks + optional figure images + optional tables)
and calls Gemini to produce an ADHD-friendly structured answer.
Tracks token usage for cost_per_query_usd metric in W&B eval.
"""

from __future__ import annotations
import base64
import logging
import os
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("generator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

SYSTEM_PROMPT = """You are an expert AI research assistant. Answer questions about AI/ML papers.

STRICT FORMAT — always follow exactly:
**[One bold sentence: the direct answer]**
• Bullet point 1 (specific, technical)
• Bullet point 2
• Bullet point 3
• Bullet point 4 (optional)
• Bullet point 5 (optional, max 5 total)
🔑 Key Takeaway: [one memorable sentence]

RULES:
- Max 150 words total in your answer
- Never write paragraphs — only bullets
- If a figure is provided, reference it explicitly in a bullet
- If a table is provided, summarize its key numbers in a bullet
- Be specific: cite model names, numbers, percentages from the context
- If you don't know, say so in one bullet — never hallucinate"""


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class GeminiGenerator:
    """Generate ADHD-friendly answers using Gemini 2.5 Flash Lite."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        gen_cfg = cfg["models"]
        self.model_name = gen_cfg["gemini_model"]
        self.max_tokens = gen_cfg.get("gemini_max_output_tokens", 512)
        self.temperature = gen_cfg.get("gemini_temperature", 0.3)
        eval_cfg = cfg.get("eval", {})
        self.cost_in = eval_cfg.get("gemini_cost_per_1k_input_tokens", 0.000075)
        self.cost_out = eval_cfg.get("gemini_cost_per_1k_output_tokens", 0.0003)

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        from google import genai
        self.client = genai.Client(api_key=api_key)

    def generate_approach_a(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Generate answer from Approach A hybrid retrieval context.
        context: output of HybridRetriever.retrieve()
        """
        from google.genai import types

        contents = []
        # Add text context
        if context.get("context_str"):
            contents.append(types.Part.from_text(
                f"RETRIEVED CONTEXT:\n{context['context_str']}\n\nQUESTION: {query}"
            ))

        # Add figure images inline
        for fig in context.get("figure_results", []):
            b64 = fig.get("image_base64", "")
            if b64:
                try:
                    img_bytes = base64.b64decode(b64)
                    contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
                    contents.append(types.Part.from_text(
                        f"[Figure from {fig.get('filename', '')} p.{fig.get('page_num', '')}]"
                    ))
                except Exception:
                    pass

        return self._call_gemini(contents, query)

    def generate_approach_b(self, query: str, page_results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Generate answer from Approach B ColPali page retrieval.
        page_results: output of ColPaliRetriever.retrieve()
        """
        from google.genai import types
        from PIL import Image as PILImage
        import io

        contents = [types.Part.from_text(f"QUESTION: {query}\n\nRelevant paper pages are attached below. Answer using what you see in them.")]

        for page in page_results:
            img_path = page.get("image_path", "")
            if img_path and Path(img_path).exists():
                try:
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                    contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
                    contents.append(types.Part.from_text(
                        f"[Page {page['page_num']} of {page['filename']} — relevance score: {page['score']:.3f}]"
                    ))
                except Exception as exc:
                    log.warning(f"Could not load page image {img_path}: {exc}")

        return self._call_gemini(contents, query)

    def _call_gemini(self, contents: list, query: str) -> dict[str, Any]:
        """Internal: call Gemini and return structured result dict."""
        from google.genai import types

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )
            answer_text = response.text or ""
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            cost = (input_tokens / 1000 * self.cost_in) + (output_tokens / 1000 * self.cost_out)

        except Exception as exc:
            log.error(f"Gemini call failed: {exc}")
            answer_text = "**Could not generate answer.**\n• Gemini API error occurred.\n🔑 Key Takeaway: Check API key and quota."
            input_tokens, output_tokens, cost = 0, 0, 0.0

        return {
            "answer": answer_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        }

    def judge_relevancy(self, query: str, answer: str) -> int:
        """
        Ask Gemini to self-judge answer relevancy on a 1-5 scale.
        Returns int 1-5. Used in run_eval.py.
        """
        from google.genai import types

        judge_prompt = (
            f"Question: {query}\n\nAnswer: {answer}\n\n"
            "Rate how well this answer addresses the question on a scale of 1 to 5:\n"
            "1=completely irrelevant, 2=mostly irrelevant, 3=partially relevant, "
            "4=mostly relevant, 5=perfectly relevant.\n"
            "Respond with ONLY a single digit 1-5. Nothing else."
        )
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[types.Part.from_text(judge_prompt)],
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
            )
            score = int(response.text.strip()[0])
            return max(1, min(5, score))
        except Exception:
            return 3  # neutral fallback