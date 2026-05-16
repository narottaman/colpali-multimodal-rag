# ColPali Multimodal RAG

**Benchmarking two state-of-the-art multimodal document retrieval architectures on 10 foundational AI/ML papers.**

[![W&B](https://img.shields.io/badge/Tracked%20with-W%26B-yellow)](https://wandb.ai/ngangada-arizona-state-university/colpali-multimodal-rag)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![A100](https://img.shields.io/badge/GPU-A100%2080GB-green)](https://www.nvidia.com/en-us/data-center/a100/)
[![ASU Sol](https://img.shields.io/badge/HPC-ASU%20Sol-maroon)](https://cores.research.asu.edu/research-computing/about-sol)

---

## Motivation

Standard RAG pipelines embed only text — they discard figures, diagrams, and tables that often contain the most important information in research papers. This project benchmarks two architectures that preserve visual semantics:

- **Approach A** — Structured multimodal RAG: Docling extracts figures and tables, Qwen2-VL-7B describes each figure in natural language, ChromaDB indexes text + VLM descriptions + tables separately, and FAISS retrieves across all three collections at query time.
- **Approach B** — End-to-end visual RAG (ColPali): PDF pages are rendered as images and embedded as patch-level multi-vectors by ColQwen2 — no OCR, no text extraction. Qdrant stores and retrieves via late-interaction MaxSim scoring, and Gemini reads the raw page images to answer.

---

## Results

Evaluated on 10 questions across 10 foundational AI papers. Both approaches use Gemini 2.5 Flash Lite for answer generation.

| Metric | Approach A | Approach B | Winner |
|---|---|---|---|
| **Context Hit Rate** | 60% | **80%** | 🏆 B |
| **Answer Relevancy (1–5)** | 3.0 | 3.0 | Tie |
| **Avg Latency** | **272ms** | 602ms | 🏆 A |
| **Cost per Query** | ~$0.00 | ~$0.00 | Tie |
| Figures Retrieved | ✅ Yes (129 indexed) | ✅ Yes (whole pages) | — |
| Tables Retrieved | ✅ Yes (136 indexed) | ✅ Via page image | — |

### Key Findings

**Approach B retrieves the correct page 80% of the time vs 60% for Approach A.** ColQwen2's patch embeddings capture visual layout, diagrams, and spatial relationships that text-only embeddings miss entirely — particularly for figure-heavy papers like DALL-E 2 (24 figures) and Stable Diffusion (34 figures).

**Approach A is 2.2× faster** (272ms vs 602ms) because it searches pre-computed FAISS indexes with a small sentence-transformer, while Approach B runs ColQwen2 at query time to embed the question into patch vectors before searching Qdrant.

**Both cost effectively $0 per query** with Gemini 2.5 Flash Lite — the model is fast enough that token costs are negligible for research-scale workloads.

**Answer relevancy is equal at 3.0/5** on this 10-question smoke test. This reflects question difficulty rather than a fundamental difference — both approaches retrieve meaningful context, but hard questions (e.g., exact formula recall) require precise chunk-level matching that neither fully achieves at this scale.

---

## Architecture

### Approach A — Structured Multimodal RAG

```
PDF (10 papers)
    │
    ▼
Docling DocumentConverter
(do_table_structure=True, generate_picture_images=True)
    │
    ├─── Text paragraphs (1,204 chunks)
    ├─── Tables → markdown + JSON rows (136 tables)
    └─── Figures → PIL images (129 figures)
              │
              ▼
         Qwen2-VL-7B-Instruct (A100)
         "Describe this figure in 2-3 sentences"
         avg 579 chars/description, 20.4 figs/min
              │
              ▼
    sentence-transformers/all-MiniLM-L6-v2
    → FAISS IndexFlatIP (cosine, normalized)
    → 3 collections: text / figures / tables
              │
              ▼
    At query time:
    embed query → search all 3 → merge results
              │
              ▼
    Gemini 2.5 Flash Lite
    ADHD-friendly answer + inline figures + tables
```

### Approach B — ColPali End-to-End Visual RAG

```
PDF (10 papers)
    │
    ▼
PyMuPDF page renderer (150 DPI)
→ 267 PNG page images
    │
    ▼
ColQwen2-v1.0 (A100)
→ patch embeddings per page (~196 patches × 128-dim)
→ 122 pages/min throughput
    │
    ▼
Qdrant (local, multi-vector native)
→ MaxSim late interaction scoring
    │
    ▼
At query time:
ColQwen2 embeds question → Qdrant MaxSim → top-3 pages
    │
    ▼
Gemini 2.5 Flash Lite reads raw page images
→ ADHD-friendly answer
```

---

## Corpus

10 foundational AI/ML papers from ArXiv:

| Paper | ArXiv ID | Pages | Figures | Tables |
|---|---|---|---|---|
| Attention Is All You Need | 1706.03762 | 15 | 6 | 4 |
| BERT | 1810.04805 | 16 | 5 | 8 |
| GPT-3 | 2005.14165 | 75 | 35 | 50 |
| ResNet | 1512.03385 | 12 | 7 | 15 |
| Adam Optimizer | 1412.6980 | 15 | 4 | 0 |
| GANs | 1406.2661 | 9 | 2 | 2 |
| DALL-E 2 | 2204.06125 | 27 | 24 | 3 |
| Stable Diffusion | 2112.10752 | 45 | 34 | 18 |
| LoRA | 2106.09685 | 26 | 8 | 18 |
| LLaMA | 2302.13971 | 27 | 4 | 18 |
| **Total** | | **267 pages** | **129 figures** | **136 tables** |

---

## Extraction Stats (Approach A)

| Paper | Text Chunks | Tables | Figures | Time |
|---|---|---|---|---|
| GANs | 45 | 2 | 2 | 28.7s |
| Adam | 63 | 0 | 4 | 4.0s |
| ResNet | 110 | 15 | 7 | 7.4s |
| Attention | 72 | 4 | 6 | 7.9s |
| BERT | 120 | 8 | 5 | 8.2s |
| GPT-3 | 297 | 50 | 35 | 69.8s |
| LoRA | 126 | 18 | 8 | 14.8s |
| Stable Diffusion | 160 | 18 | 34 | 57.5s |
| DALL-E 2 | 81 | 3 | 24 | 77.2s |
| LLaMA | 130 | 18 | 4 | 44.4s |
| **Total** | **1,204** | **136** | **129** | **~13 min** |

Qwen2-VL described all 129 figures in **6.3 minutes** at 20.4 figures/min, averaging 579 characters per description.

---

## Build Performance

| Pipeline | Step | Time |
|---|---|---|
| **Approach A** | Docling extraction (10 PDFs) | ~5 min |
| | Qwen2-VL figure description (129 figs) | 6.3 min |
| | FAISS embedding + indexing (1,469 elements) | 1.4s |
| | **Total** | **13 min** |
| **Approach B** | PyMuPDF page rendering (267 pages) | 30s |
| | ColQwen2 embedding + Qdrant indexing | 2.2 min |
| | **Total** | **3.3 min** |

---

## Setup

### Prerequisites

- ASU Sol HPC account with GPU access (`grp_cbaral` allocation)
- HuggingFace token (for Qwen2-VL model access)
- Gemini API key (for generation and eval judging)
- W&B account (for experiment tracking)

### Environment

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Create venv (Python 3.10 — avoids Sol's Mamba sqlite3 conflict)
uv venv $HOME/.venv --python 3.10
source $HOME/.venv/bin/activate

# Install packages
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt

# Pin versions for compatibility with torch 2.6 on Sol
uv pip install "transformers==4.49.0" "peft==0.14.0" \
               "torchao==0.6.1" "colpali-engine==0.3.1"
```

### Configuration

```bash
cp .env.example .env
# Fill in: GEMINI_API_KEY, WANDB_API_KEY, HF_TOKEN
```

### Build Both Indexes

```bash
cd /scratch/ngangada/portfolio/colpali-multimodal-rag
mkdir -p logs

sbatch sol/approach_a.slurm   # ~13 min on A100
sbatch sol/approach_b.slurm   # ~3.3 min on A100
```

### Run Evaluation

```bash
sbatch sol/run_eval.slurm     # ~20 min for 20 questions
```

### Launch Gradio UI (laptop)

```bash
pip install colpali-engine qdrant-client pillow pymupdf \
            google-genai gradio pyyaml python-dotenv \
            sentence-transformers faiss-cpu

# Copy indexes from Sol first
scp -r ngangada@sol.asu.edu:/scratch/ngangada/portfolio/colpali-multimodal-rag/data ./

python app/chat.py
# Open http://localhost:7860
```

---

## Project Structure

```
colpali-multimodal-rag/
├── src/
│   ├── extractors/
│   │   ├── docling_extractor.py    # Approach A: Docling PDF extraction
│   │   └── page_renderer.py        # Approach B: PDF → PNG pages
│   ├── describers/
│   │   └── qwen_vl_describer.py    # Approach A: Qwen2-VL figure captions
│   ├── indexers/
│   │   ├── chroma_indexer.py       # Approach A: FAISS + numpy persistence
│   │   └── qdrant_indexer.py       # Approach B: ColQwen2 + Qdrant
│   ├── retrievers/
│   │   ├── hybrid_retriever.py     # Approach A: FAISS search over 3 collections
│   │   └── colpali_retriever.py    # Approach B: MaxSim late interaction
│   └── generator.py                # Gemini 2.5 Flash Lite, ADHD prompt
├── scripts/
│   ├── build_approach_a.py         # End-to-end Approach A pipeline
│   ├── build_approach_b.py         # End-to-end Approach B pipeline
│   └── run_eval.py                 # Compare both, log to W&B
├── app/
│   └── chat.py                     # Gradio UI
├── sol/
│   ├── approach_a.slurm            # SLURM job for Approach A
│   ├── approach_b.slurm            # SLURM job for Approach B
│   └── run_eval.slurm              # SLURM job for evaluation
├── configs/
│   └── config.yaml                 # All paths, model names, hyperparameters
└── data/
    ├── raw/pdfs/                   # 10 ArXiv papers
    ├── processed/                  # Manifests, rendered pages
    ├── chroma_db/                  # Approach A FAISS indexes
    ├── qdrant_db/                  # Approach B Qdrant collection
    └── eval/                       # QA pairs + results
```

---

## Response Format

Both approaches return ADHD-friendly structured answers:

```
**[One bold direct answer sentence]**
• Bullet point 1 (specific, technical)
• Bullet point 2
• Bullet point 3 (max 5 bullets)
[Figure shown inline if retrieved]
[Markdown table shown if retrieved]
🔑 Key Takeaway: [one memorable sentence]
Max 150 words. Never paragraphs. Always bullets.
```

---

## W&B Experiment Tracking

All runs tracked at: [wandb.ai/ngangada-arizona-state-university/colpali-multimodal-rag](https://wandb.ai/ngangada-arizona-state-university/colpali-multimodal-rag)

Tracked metrics per approach per question:
- `context_hit_rate` — did the correct page/chunk get retrieved?
- `answer_relevancy` — Gemini self-judge score (1–5)
- `latency_ms` — end-to-end query time
- `cost_usd` — Gemini token cost
- `figures_retrieved` / `tables_retrieved` — multimodal retrieval rate

---

## Technical Notes

**Why not ChromaDB PersistentClient?** ASU Sol's system `sitecustomize.py` forces Mamba's `libsqlite3.so` (pre-3.23.0) before any venv code runs. ChromaDB's SQLite dependency fails. Solution: use FAISS + numpy `.npz` files for Approach A persistence, and Qdrant local mode (no SQLite) for Approach B.

**Why Python 3.10 via uv?** Sol's default Mamba Python 3.12 has the SQLite conflict above. `uv venv --python 3.10` creates a fully isolated environment using a different Python binary that doesn't inherit Sol's system hooks.

**Why `colpali-engine==0.3.1`?** Version 0.3.16 imports Gemma3 which requires `transformers>=5.0`. But `transformers 5.x` breaks Qwen2-VL. Pinning to 0.3.1 keeps everything on `transformers 4.49.0`.

---

## Portfolio Context

Built as a portfolio project targeting AI/ML engineering roles. Part of a broader series:

- **rag-document-assistant** — ChromaDB + HNSW + W&B hyperparameter sweeps + Gradio + Docker
- **colpali-multimodal-rag** — This project: visual RAG comparison on research papers ← you are here
- **grpo-logic-puzzles** — GRPO/RL training on Qwen2.5 with VERL on ASU Sol A100

**Stack:** Python · PyTorch · HuggingFace Transformers · Docling · ColPali · Qwen2-VL · FAISS · Qdrant · ChromaDB · Gemini API · W&B · Gradio · SLURM · ASU Sol A100