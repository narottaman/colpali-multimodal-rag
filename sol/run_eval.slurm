#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=colpali_eval
#SBATCH --partition=gpu
#SBATCH -p public
#SBATCH -q public
#SBATCH --nodes=1
#SBATCH -N 1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ngangada/portfolio/colpali-multimodal-rag/logs/eval_%j.out
#SBATCH --error=/scratch/ngangada/portfolio/colpali-multimodal-rag/logs/eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ngangada@asu.edu

set -e
PROJECT=/scratch/ngangada/portfolio/colpali-multimodal-rag

echo "============================================================"
echo " EVAL: Approach A vs B comparison"
echo " Job   : $SLURM_JOB_ID"
echo " Node  : $SLURMD_NODENAME"
echo " GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo " Start : $(date)"
echo "============================================================"

module purge
module load cuda-12.4.1-gcc-12.1.0
source "$HOME/.venv/bin/activate"

echo "Python : $(which python) — $(python --version)"
echo "CUDA   : $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU ok : $(python -c 'import torch; print(torch.cuda.is_available())')"

cd $PROJECT
mkdir -p logs data/eval/results

echo ""
echo "── Running eval (10 questions smoke test) ───────────────────"
python scripts/run_eval.py \
    --config configs/config.yaml \
    --max_questions 10 \
    2>&1 | tee logs/eval_${SLURM_JOB_ID}.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo " EVAL — SUCCESS"
    echo " Results : $PROJECT/data/eval/results/"
    echo " W&B     : check wandb.ai for eval_comparison run"
    if [ -f "$PROJECT/data/eval/results/eval_summary.json" ]; then
        python - << 'PYSUMMARY'
import json
s = json.load(open("data/eval/results/eval_summary.json"))
print("\n  === RESULTS ===")
for approach, m in s.items():
    print(f"\n  {approach}:")
    for k, v in m.items():
        print(f"    {k}: {v:.4f}")
PYSUMMARY
    fi
else
    echo " EVAL — FAILED (exit code $EXIT_CODE)"
    echo " Check: $PROJECT/logs/eval_${SLURM_JOB_ID}.log"
fi
echo " End: $(date)"
echo "============================================================"
exit $EXIT_CODE
