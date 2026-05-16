#!/bin/bash
#SBATCH --job-name=colpali_build_all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ngangada/portfolio/colpali-multimodal-rag/logs/build_all_%j.out
#SBATCH --error=/scratch/ngangada/portfolio/colpali-multimodal-rag/logs/build_all_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ngangada@asu.edu

# =============================================================================
# build_all.slurm — Run BOTH pipelines end-to-end on a single A100 job.
#
# Submit with:
#   cd /scratch/ngangada/portfolio/colpali-multimodal-rag
#   mkdir -p logs
#   sbatch sol/build_all.slurm
#
# Runs in order:
#   1. pip install (idempotent)
#   2. Approach A: Docling → Qwen2-VL → ChromaDB
#   3. Approach B: PyMuPDF → ColQwen2 → Qdrant
#   4. Eval: compare both, log to W&B
# =============================================================================

set -e  # Exit immediately on any error

echo "============================================================"
echo " ColPali Multimodal RAG — Full Build"
echo " Job ID : $SLURM_JOB_ID"
echo " Node   : $SLURMD_NODENAME"
echo " GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo " Start  : $(date)"
echo "============================================================"

# ── Environment ───────────────────────────────────────────────────────────────
module load cuda/12.1 2>/dev/null || true
source ~/envs/rag/bin/activate

PROJECT=/scratch/ngangada/portfolio/colpali-multimodal-rag
cd $PROJECT
mkdir -p logs data/processed data/eval/results

echo ""
echo "── Installing requirements ──────────────────────────────────"
pip install -r requirements.txt --quiet
pip install python-dotenv --quiet  # ensure dotenv available

# ── Approach A ────────────────────────────────────────────────────────────────
echo ""
echo "── APPROACH A: Docling + Qwen2-VL + ChromaDB ───────────────"
echo "Start: $(date)"

python scripts/build_approach_a.py \
    --config configs/config.yaml \
    2>&1 | tee logs/approach_a_${SLURM_JOB_ID}.log

echo "Approach A done: $(date)"

# ── Approach B ────────────────────────────────────────────────────────────────
echo ""
echo "── APPROACH B: ColQwen2 + Qdrant ───────────────────────────"
echo "Start: $(date)"

python scripts/build_approach_b.py \
    --config configs/config.yaml \
    2>&1 | tee logs/approach_b_${SLURM_JOB_ID}.log

echo "Approach B done: $(date)"

# ── Eval (optional — runs if qa_pairs.json has entries) ──────────────────────
echo ""
echo "── EVAL: Comparing both approaches ─────────────────────────"
echo "Start: $(date)"

# Quick smoke test: first 5 questions only (full eval run separately)
python scripts/run_eval.py \
    --config configs/config.yaml \
    --max_questions 5 \
    2>&1 | tee logs/eval_${SLURM_JOB_ID}.log

echo "Eval done: $(date)"

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Build complete!"
echo " End: $(date)"
echo ""
echo " Outputs:"
echo "   ChromaDB : $PROJECT/data/chroma_db"
echo "   Qdrant   : $PROJECT/data/qdrant_db"
echo "   Pages    : $PROJECT/data/processed/pages"
echo "   Manifests: $PROJECT/data/processed/"
echo "   Eval     : $PROJECT/data/eval/results/"
echo "   Logs     : $PROJECT/logs/"
echo "============================================================"
