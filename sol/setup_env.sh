#!/bin/bash
# =============================================================================
# setup_env.sh — Create the colpali venv from scratch on Sol.
#
# Run ONCE before submitting any SLURM jobs:
#   cd /scratch/ngangada/portfolio/colpali-multimodal-rag
#   bash sol/setup_env.sh
#
# Creates: ~/envs/rag/
# Takes:   ~15-20 min (downloads torch + all packages)
# =============================================================================

set -e

VENV_DIR="$HOME/envs/rag"
PROJECT="/scratch/ngangada/portfolio/colpali-multimodal-rag"

echo "============================================================"
echo " ColPali — Environment Setup"
echo " Venv target : $VENV_DIR"
echo " Project     : $PROJECT"
echo " Start       : $(date)"
echo "============================================================"

# ── Load Sol modules ──────────────────────────────────────────────────────────
echo ""
echo "── Loading modules ──────────────────────────────────────────"
module purge
module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || echo "No cuda module — using system CUDA"
module load python/3.11 2>/dev/null || module load python 2>/dev/null || echo "No python module — using system python"
echo "Python: $(which python3) — $(python3 --version)"

# ── Create venv ───────────────────────────────────────────────────────────────
echo ""
echo "── Creating venv at $VENV_DIR ───────────────────────────────"
if [ -d "$VENV_DIR" ]; then
    echo "Venv already exists — skipping creation (delete $VENV_DIR to rebuild)"
else
    python3 -m venv "$VENV_DIR"
    echo "Venv created."
fi

source "$VENV_DIR/bin/activate"
echo "Active venv: $(which python) — $(python --version)"

# ── Upgrade pip ───────────────────────────────────────────────────────────────
echo ""
echo "── Upgrading pip ────────────────────────────────────────────"
pip install --upgrade pip setuptools wheel

# ── Install PyTorch first (CUDA 12.1 wheel) ───────────────────────────────────
echo ""
echo "── Installing PyTorch (CUDA 12.1) ──────────────────────────"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ── Verify CUDA is visible ────────────────────────────────────────────────────
python - << 'PYCHECK'
import torch
print(f"  torch      : {torch.__version__}")
print(f"  CUDA avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
else:
    print("  WARNING: CUDA not available — GPU jobs will fail!")
PYCHECK

# ── Install all other requirements ────────────────────────────────────────────
echo ""
echo "── Installing project requirements ─────────────────────────"
cd "$PROJECT"
pip install -r requirements.txt

# ── Verify all key imports ────────────────────────────────────────────────────
echo ""
echo "── Verifying imports ────────────────────────────────────────"
python - << 'PYCHECK'
import sys
packages = [
    "torch", "transformers", "sentence_transformers",
    "docling", "chromadb", "colpali_engine",
    "qdrant_client", "fitz", "PIL",
    "wandb", "gradio", "google.genai",
    "yaml", "dotenv", "tqdm", "pandas",
]
failed = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  OK   {pkg}")
    except ImportError as e:
        print(f"  FAIL {pkg}: {e}")
        failed.append(pkg)

print()
if failed:
    print(f"FAILED packages: {failed}")
    print("Re-run: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("All packages installed successfully!")
PYCHECK

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup complete!"
echo " Activate with: source $VENV_DIR/bin/activate"
echo " End: $(date)"
echo "============================================================"
