#!/bin/bash
#SBATCH --job-name=mm_tool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=54:00:00
#SBATCH --output=mm_tool_%j.out
#SBATCH --error=mm_tool_%j.err

set -euo pipefail
xml_catalog_files_libxml2=""

# Environment setup (same as before — safe and necessary)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

# Use local scratch for temp files
export TMPDIR="${SLURM_TMPDIR:-/tmp}/${USER}/ag_tmp_fixed"
mkdir -p "$TMPDIR"
export JOBLIB_TEMP_FOLDER="$TMPDIR"
export MPLCONFIGDIR="${TMPDIR}/mpl"; mkdir -p "$MPLCONFIGDIR"
export MPLBACKEND=Agg

# HF cache — offline mode
export HF_CACHE_ROOT="/share/lab_goecks/paulo/hf_cache"
export HF_HOME="$HF_CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda environment if available
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate pycaret_env || true
fi

# Go to submit directory (safe)
cd "$SLURM_SUBMIT_DIR"

# === EXACT COMMAND YOU REQUESTED ===
echo "Starting fixed AutoGluon Multimodal experiment..."

srun -u python /share/lab_goecks/paulo/gleam/tools/tabularlearner/pycaret_train.py \
  --input_file "../../loris_processed/Chowell_train.tsv" \
  --target_col  "22" \
  --output_dir "." \
  --model_type "classification" \
  --cross_validation \
  --cross_validation_folds 10 \
  --test_file "../../loris_processed/Chowell_test.tsv" \
  --random_seed 42 \
  --probability_threshold 0.29 \
  --best_model_metric "Accuracy" \
