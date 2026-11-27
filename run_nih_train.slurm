#!/bin/bash
#SBATCH --job-name=nih_age_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
# #SBATCH --qos=private
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=you@asu.edu

set -euo pipefail
mkdir -p logs

# ===== CPU/繪圖/快取 =====
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-/scratch/ykong26/datasets/NIH_ChestXRay/.torch}"

# ===== 重要路徑 =====
REPO_DIR="${REPO_DIR:-/scratch/ykong26/DNN_Xray}"   # 你的訓練腳本所在資料夾
PYFILE="${PYFILE:-train_DNN.py}"                     # 你的訓練腳本檔名
DATA_ROOT="${DATA_ROOT:-/scratch/ykong26/datasets/NIH_ChestXray}"
SAVE_DIR="${SAVE_DIR:-${DATA_ROOT}/checkpoints}"

# 超參（可在 sbatch 時用 --export 覆蓋）
BACKBONE="${BACKBONE:-resnet50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
IMG_SIZE="${IMG_SIZE:-320}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# ===== Conda =====
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found"; exit 1
fi
CONDA_ENV="${CONDA_ENV:-ark_2d_scratch}"
conda activate "${CONDA_ENV}" || { echo "ERROR: fail to activate ${CONDA_ENV}"; exit 1; }

# ===== 檢查並切換到專案目錄 =====
[ -d "${REPO_DIR}" ] || { echo "ERROR: REPO_DIR not found: ${REPO_DIR}"; exit 2; }
cd "${REPO_DIR}"
[ -f "${PYFILE}" ] || { echo "ERROR: PYFILE not found in ${REPO_DIR}: ${PYFILE}"; exit 3; }
[ -d "${DATA_ROOT}" ] || { echo "ERROR: DATA_ROOT not found: ${DATA_ROOT}"; exit 4; }
mkdir -p "${SAVE_DIR}"

echo "Repo dir    : ${REPO_DIR}"
echo "Conda env   : ${CONDA_DEFAULT_ENV}"
echo "PYFILE      : ${PYFILE}"
echo "DATA_ROOT   : ${DATA_ROOT}"
echo "SAVE_DIR    : ${SAVE_DIR}"
echo "BACKBONE    : ${BACKBONE}"
echo "BATCH/EPOCH : ${BATCH_SIZE} / ${NUM_EPOCHS}"
echo "LR/WD       : ${LR} / ${WEIGHT_DECAY}"
echo "IMG_SIZE    : ${IMG_SIZE}"
echo "NUM_WORKERS : ${NUM_WORKERS}"

# 可選：確認處理後的 CSV 是否存在
if [ ! -f "${DATA_ROOT}/processed/train_set.csv" ] || [ ! -f "${DATA_ROOT}/processed/val_set.csv" ]; then
  echo "ERROR: processed CSVs not found under ${DATA_ROOT}/processed"; exit 5;
fi

# ===== 啟動訓練 =====
set -x
srun --ntasks=1 python "${PYFILE}" \
  --data_root "${DATA_ROOT}" \
  --save_dir "${SAVE_DIR}" \
  --backbone "${BACKBONE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --img_size "${IMG_SIZE}" \
  --num_workers "${NUM_WORKERS}"
set +x
