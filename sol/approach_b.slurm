#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=colpali_b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --output=/scratch/ngangada/portfolio/colpali-multimodal-rag/logs/approach_b_%j.out
#SBATCH --error=/scratch/ngangada/portfolio/colpali-multimodal-rag/logs/approach_b_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ngangada@asu.edu


set -e
PROJECT=/scratch/ngangada/portfolio/colpali-multimodal-rag
VENV="$HOME/envs/rag"

echo "============================================================"
echo " APPROACH B: ColQwen2 + Qdrant"
echo " Job   : $SLURM_JOB_ID"
echo " Node  : $SLURMD_NODENAME"
echo " GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo " Start : $(date)"
echo "============================================================"

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load cuda-12.4.1-gcc-12.1.0
source "$HOME/.venv/bin/activate"
echo "Python : $(which python) — $(python --version)"
echo "CUDA   : $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU ok : $(python -c 'import torch; print(torch.cuda.is_available())')"

# ── Ensure pysqlite3-binary is installed (fixes Sol's old system SQLite) ──────
echo ""
echo "── SQLite patch ─────────────────────────────────────────────"
pip install pysqlite3-binary --quiet

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cat > "$SITE_PACKAGES/sitecustomize.py" << 'PYEOF'
try:
    import pysqlite3 as _pysqlite3
    import sys
    sys.modules["sqlite3"] = _pysqlite3
except ImportError:
    pass
PYEOF
echo "  sitecustomize.py written → $SITE_PACKAGES"
python -c "import pysqlite3; print(f'  SQLite version: {pysqlite3.sqlite_version}')"

# ── Directories ───────────────────────────────────────────────────────────────
cd $PROJECT
mkdir -p logs data/processed/pages data/qdrant_db

# ── Import check ──────────────────────────────────────────────────────────────
echo ""
echo "── Import check ─────────────────────────────────────────────"
python - << 'PYCHECK'
import sys
failed = []
for pkg in ["fitz", "colpali_engine", "qdrant_client",
            "wandb", "torch", "PIL", "yaml", "dotenv"]:
    try:
        __import__(pkg)
        print(f"  OK   {pkg}")
    except Exception as e:
        print(f"  FAIL {pkg}: {e}")
        failed.append(pkg)
if failed:
    print(f"\nMissing: {failed}")
    sys.exit(1)
print("All imports OK")
PYCHECK

# ── Run pipeline ──────────────────────────────────────────────────────────────
echo ""
echo "── Running Approach B pipeline ──────────────────────────────"
echo "Start: $(date)"

python scripts/build_approach_b.py \
    --config configs/config.yaml \
    2>&1 | tee logs/approach_b_pipeline_${SLURM_JOB_ID}.log

EXIT_CODE=${PIPESTATUS[0]}

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo " APPROACH B — SUCCESS"
    echo " Qdrant DB : $PROJECT/data/qdrant_db/"
    echo " Pages     : $PROJECT/data/processed/pages/"
    echo " Manifest  : $PROJECT/data/processed/approach_b_manifest.json"
    echo " Log       : $PROJECT/logs/approach_b_pipeline_${SLURM_JOB_ID}.log"
    if [ -f "$PROJECT/data/processed/approach_b_manifest.json" ]; then
        python - << 'PYSUMMARY'
import json
from collections import Counter
with open("data/processed/approach_b_manifest.json") as f:
    data = json.load(f)
by_pdf = Counter(p["pdf_stem"] for p in data)
print(f"  Pages indexed: {len(data)} total")
for pdf, count in sorted(by_pdf.items()):
    print(f"    {pdf}: {count} pages")
PYSUMMARY
    fi
else
    echo " APPROACH B — FAILED (exit code $EXIT_CODE)"
    echo " Check: $PROJECT/logs/approach_b_pipeline_${SLURM_JOB_ID}.log"
fi
echo " End: $(date)"
echo "============================================================"
exit $EXIT_CODE
