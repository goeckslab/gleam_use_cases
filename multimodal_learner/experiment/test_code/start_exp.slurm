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

# Avoid unbound variable issues from libraries/activation scripts
xml_catalog_files_libxml2=""

# Environment setup
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

# Use local scratch for temp files
export TMPDIR="${SLURM_TMPDIR:-/tmp}/${USER}/ag_tmp_fixed"
mkdir -p "$TMPDIR"

export JOBLIB_TEMP_FOLDER="$TMPDIR"

export MPLCONFIGDIR="${TMPDIR}/mpl"
mkdir -p "$MPLCONFIGDIR"
export MPLBACKEND=Agg

# HF cache — offline mode
export HF_CACHE_ROOT="/share/lab_goecks/paulo/hf_cache"
export HF_HOME="$HF_CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# PyTorch CUDA allocation config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SSL certificates
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE="$REQUESTS_CA_BUNDLE"

# Activate conda environment.
# Temporarily disable `set -u` because some conda activation scripts,
# including MKL/libblas scripts, may reference unset variables.
if command -v conda >/dev/null 2>&1; then
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate autogluon_env
  set -u
else
  echo "ERROR: conda command not found"
  exit 1
fi

# Force Conda env Python to the front of PATH.
# Needed because the cluster/module Python may appear before Conda in PATH.
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r

# Go to submit directory
cd "$SLURM_SUBMIT_DIR"

echo "Starting fixed AutoGluon Multimodal experiment..."
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Submit directory: ${SLURM_SUBMIT_DIR}"
echo "Running on host: $(hostname)"
echo "Active conda env: ${CONDA_DEFAULT_ENV:-none}"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-none}"
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"
echo "Python version: $(python --version)"
echo "Pip version: $(python -m pip -V)"
python -c "import sys; print('Python executable:', sys.executable)"
echo "HF_HOME: ${HF_HOME}"
echo "HUGGINGFACE_HUB_CACHE: ${HUGGINGFACE_HUB_CACHE}"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-not set}"

srun -u python /share/lab_goecks/paulo/gleam/tools/multimodallearner/multimodal_learner.py \
  --input_csv_train "../../3gb_dataset/HANCOCK_train_split_3GB_jpeg.csv" \
  --input_csv_test "../../3gb_dataset/HANCOCK_test_split_3GB_jpeg.csv" \
  --images_zip "../../3gb_dataset/CD3_CD8_images_3GB_jpeg.zip" \
  --target_column "target" \
  --sample_id_column "patient_id" \
  --backbone_image "caformer_b36.sail_in22k_ft_in1k" \
  --backbone_text "google/electra-base-discriminator" \
  --missing_image_strategy false \
  --output_json "results.json" \
  --output_html "report.html" \
  --output_config "config.yaml" \
  --output_model_archive "trained_model.zip" \
  --preset "medium_quality" \
  --eval_metric "roc_auc" \
  --random_seed 42 \
  --threshold_metric auto

echo "Experiment completed!"
