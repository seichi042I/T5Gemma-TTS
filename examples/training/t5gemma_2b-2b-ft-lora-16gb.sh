#!/usr/bin/env bash

set -euo pipefail

# 16GB VRAM向け省メモリLoRA学習スクリプト
# This script is optimized for fine-tuning T5Gemma-TTS with LoRA on GPUs with 16GB VRAM.
# Uses gradient checkpointing, reduced batch sizes, and memory-efficient settings.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ベースモデル設定
T5GEMMA_MODEL_NAME=google/t5gemma-2b-2b-ul2
XCODEC2_MODEL_NAME=NandemoGHS/Anime-XCodec2-44.1kHz-v2

# プリトレインモデルのパス（存在しない場合はHuggingFaceベースモデルから学習）
# HuggingFaceからダウンロード済みの場合: pretrained.pth
# または公式モデルを変換した場合のパスを指定
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${PROJECT_ROOT}/pretrained.pth}"

# データセットディレクトリ（必要に応じて変更）
# RISE用: datasets/rise
# 汎用: datasets/your_dataset
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/datasets/rise}"

# 出力ディレクトリ
EXP_NAME="${EXP_NAME:-rise_lora_16gb}"
EXP_ROOT="${PROJECT_ROOT}/runs/${EXP_NAME}"

# GPU設定
NUM_GPUS="${NUM_GPUS:-1}"

# === 16GB VRAM向け最適化設定 ===
# バッチサイズ・トークン数を削減してメモリ使用量を抑制
BATCH_SIZE=1
MAX_NUM_TOKENS=2000      # デフォルト30000→8000に削減
VAL_MAX_NUM_TOKENS=2000  # デフォルト5000→2000に削減

# 音声長制限（長い音声はメモリを大量消費）
AUDIO_MAX_LENGTH=20      # デフォルト40秒→20秒
AUDIO_MIN_LENGTH=0.2
TEXT_MAX_LENGTH=300      # デフォルト500→300

# 学習設定
NUM_STEPS="${NUM_STEPS:-1000}"
LR=0.035
WARMUP_FRAC=0.02
GRADIENT_ACCUMULATION_STEPS=16  # バッチが小さい分を蓄積で補う

# 検証・保存頻度
VAL_EVERY=500
PRINT_EVERY=10
SAVE_EVERY=500

# LoRA設定（省メモリ版）
LORA_R=4                 # デフォルト8→4
LORA_ALPHA=8             # デフォルト16→8
LORA_DROPOUT=0.05
# gate_proj, up_proj, down_projを除外してメモリ削減
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"

# Neighbor/Prompt設定
NEIGHBOR_PROB=0.5
NEIGHBOR_FOLDER="${NEIGHBOR_FOLDER_NAME:-neighbors}"

# T5Gemma特殊トークン設定
X_SEP_TOKEN=255999
N_SPECIAL=5
AUDIO_VOCAB_SIZE=65536

# データセット設定
DATASET_DIRS="['${DATASET_ROOT}']"
MANIFEST_NAMES="['manifest_final']"

# ディレクトリ作成
mkdir -p "${EXP_ROOT}"

# プリトレインモデルの存在確認
LOAD_MODEL_ARG=""
if [[ -f "${PRETRAINED_MODEL_PATH}" ]]; then
  LOAD_MODEL_ARG="--load_model_from ${PRETRAINED_MODEL_PATH}"
  echo "プリトレインモデル: ${PRETRAINED_MODEL_PATH}"
else
  echo "プリトレインモデルが見つかりません: ${PRETRAINED_MODEL_PATH}"
  echo "HuggingFaceベースモデル(${T5GEMMA_MODEL_NAME})から直接学習します。"
  echo ""
  echo "注意: 公式プリトレインモデルを使用する場合は以下を実行:"
  echo "  python scripts/export_t5gemma_voice_hf.py などで変換後、"
  echo "  PRETRAINED_MODEL_PATH=/path/to/model.pth を指定してください。"
fi

echo ""
echo "============================================"
echo "T5Gemma-TTS LoRA Fine-tuning (16GB VRAM)"
echo "============================================"
echo "Dataset: ${DATASET_ROOT}"
echo "Output:  ${EXP_ROOT}"
echo "GPUs:    ${NUM_GPUS}"
echo "Steps:   ${NUM_STEPS}"
echo "Max tokens/batch: ${MAX_NUM_TOKENS}"
echo "LoRA rank: ${LORA_R}, alpha: ${LORA_ALPHA}"
echo "============================================"
echo ""
echo "TensorBoard監視: tensorboard --logdir=${EXP_ROOT}"
echo ""

export CUDA_VISIBLE_DEVICES=0

torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_ROOT}/main.py" \
  --model_arch t5gemma \
  --t5gemma_model_name "${T5GEMMA_MODEL_NAME}" \
  --text_input_type text \
  --text_tokenizer_name "${T5GEMMA_MODEL_NAME}" \
  --audio_tokenizer xcodec2 \
  --xcodec2_model_name "${XCODEC2_MODEL_NAME}" \
  --audio_vocab_size "${AUDIO_VOCAB_SIZE}" \
  --progress_scale 2000 \
  --neighbor_prompt_prob "${NEIGHBOR_PROB}" \
  --neighbor_folder_name "${NEIGHBOR_FOLDER}" \
  --n_special "${N_SPECIAL}" \
  --x_sep_token "${X_SEP_TOKEN}" \
  --no_loss_on_prefix 1 \
  --min_prompt_len 0.5 \
  --audio_max_length "${AUDIO_MAX_LENGTH}" \
  --audio_min_length "${AUDIO_MIN_LENGTH}" \
  --text_max_length "${TEXT_MAX_LENGTH}" \
  --encodec_sr 50 \
  --dataset_dir "${DATASET_DIRS}" \
  --manifest_name "${MANIFEST_NAMES}" \
  --encodec_folder_name xcodec2_1cb \
  --audio_folder_name audio \
  --target_time_stretch_prob 0 \
  --time_stretch_prob 0 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers 2 \
  --max_num_tokens "${MAX_NUM_TOKENS}" \
  --val_max_num_tokens "${VAL_MAX_NUM_TOKENS}" \
  --num_steps "${NUM_STEPS}" \
  --lr "${LR}" \
  --warmup_fraction "${WARMUP_FRAC}" \
  --precision bfloat16 \
  --print_every_n_steps "${PRINT_EVERY}" \
  --val_every_n_steps "${VAL_EVERY}" \
  --inference_every_n_steps 100000000 \
  --save_every_n_steps "${SAVE_EVERY}" \
  --tb_write_every_n_steps 1 \
  --seed 1 \
  --exp_dir "${EXP_ROOT}" \
  --drop_long 1 \
  --pad_x 0 \
  --text_pad_token 0 \
  --num_buckets 20 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --optimizer_name "ScaledAdam" \
  --pseudo_epoch_size 5000 \
  --reduce_lr_start_step 5000 \
  --reduce_lr_start_epoch 6 \
  --clipping_update_period 1000 \
  --validation_sample_cap 10000 \
  --t5_gradient_checkpointing 1 \
  --prune_text_modules 2 \
  --compile 0 \
  --attn_implementation sdpa \
  --ddp_find_unused_parameters 0 \
  ${LOAD_MODEL_ARG} \
  --use_lora 1 \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --disable_wandb 1

echo ""
echo "============================================"
echo "Training completed!"
echo "Model saved to: ${EXP_ROOT}"
echo "View logs: tensorboard --logdir=${EXP_ROOT}"
echo "============================================"

