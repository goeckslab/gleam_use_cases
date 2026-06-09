#!/bin/bash
#SBATCH --job-name=mm_tool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=54:00:00
#SBATCH --output=mm_tool_%j.out
#SBATCH --error=mm_tool_%j.err

set -euo pipefail
xml_catalog_files_libxml2=""

# Environment setup
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

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pycaret_env

# Force Conda env Python to the front of PATH
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r

# Debug: confirm the job is using the intended environment
echo "CONDA_PREFIX=$CONDA_PREFIX"
which python
python --version
python -m pip -V
python -c "import sys; print(sys.executable)"


# Go to submit directory
cd "$SLURM_SUBMIT_DIR"

echo "Starting Tabular Learner experiment..."

srun -u python /share/lab_goecks/paulo/gleam/tools/tabularlearner/pycaret_train.py \
  --input_file "../../loris_processed/Chowell_train.tsv" \
  --target_col "22" \
  --output_dir "." \
  --model_type "classification" \
  --test_file "../../loris_processed/Chowell_test.tsv" \
  --random_seed 42 \
  --threshold_mode auto \
  --threshold_metric "F1" \
  --best_model_metric "Accuracy"
