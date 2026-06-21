#!/bin/bash

# 数据集为 ReplayNVAS
# 以7个视角作为训练集，1个视角作为测试集。

set -e

echo "Starting per-scene Audio 3DGS training (viewpoint-based, paper style)..."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CONFIG_FILE="configs/audio_3dgs_replaynvas_viewpoint.yaml"
TEST_VIEWPOINT=${1:-8}  # override: bash ... 8
shift 1 || true

# Scenes priority:
#   1) positional scenes (one or more args starting with "SC-")
#   2) env: A3DGS_SCENES="SC-1044,SC-1052,..." (comma/space separated)
#   3) env: A3DGS_SCENE="SC-1044"
#   4) default: SC-1084
SCENES_FROM_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    SC-*)
      SCENES_FROM_ARGS+=("$1")
      shift 1
      ;;
    *)
      break
      ;;
  esac
done

EXTRA_ARGS=("$@")

SCENES_SPEC=""
if [ ${#SCENES_FROM_ARGS[@]} -gt 0 ]; then
  SCENES_SPEC="${SCENES_FROM_ARGS[*]}"
elif [ -n "${A3DGS_SCENES:-}" ]; then
  SCENES_SPEC="${A3DGS_SCENES}"
elif [ -n "${A3DGS_SCENE:-}" ]; then
  SCENES_SPEC="${A3DGS_SCENE}"
else
  SCENES_SPEC="SC-1074"
fi

SCENES_SPEC="${SCENES_SPEC//,/ }"
read -r -a SCENES <<< "${SCENES_SPEC}"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file $CONFIG_FILE not found!"; exit 1
fi

DATASET_PATH="data/avcloud_data/ReplayNVAS"
if [ ! -d "$DATASET_PATH" ]; then
  echo "Error: ReplayNVAS dataset not found at $DATASET_PATH"; exit 1
fi

echo "Config: $CONFIG_FILE"
echo "Dataset: $DATASET_PATH"
echo "Test viewpoint: $TEST_VIEWPOINT"
echo "Scenes: ${SCENES[*]}"
if [ "${A3DGS_USE_METADATA:-0}" != "0" ]; then
  echo "使用 metadata_v2.json 构建样本列表（公平对比NVAS）"
  MD_FLAGS=(--use-metadata --metadata-file metadata_v2.json)
else
  MD_FLAGS=()
fi

# 可选特定viewpoint作为输入音频源
if [ -n "${A3DGS_INPUT_SOURCE:-}" ]; then
  if [ "${A3DGS_INPUT_SOURCE}" = "viewpoint" ]; then
    echo "输入音频源: viewpoint, vp=${A3DGS_INPUT_VP:-0}"
    INP_FLAGS=(--input-source viewpoint --input-viewpoint "${A3DGS_INPUT_VP:-0}")
  else
    echo "输入音频源: near"
    INP_FLAGS=(--input-source near)
  fi
else
  INP_FLAGS=()
fi

for SCENE in "${SCENES[@]}"; do
  echo "\n=== Training scene: $SCENE (test vp=$TEST_VIEWPOINT) ==="
  # 可选可用的训练视角, e.g., A3DGS_TRAIN_VP="2,3,4,5,6,7"
  if [ -n "${A3DGS_TRAIN_VP:-}" ]; then
    echo "训练视角: ${A3DGS_TRAIN_VP} (将排除测试视角${TEST_VIEWPOINT})"
    TVP_FLAGS=(--train-viewpoints "${A3DGS_TRAIN_VP}")
  else
    TVP_FLAGS=()
  fi
  python tools/train_audio_3dgs_viewpoint.py \
    --cfg "$CONFIG_FILE" \
    --test-viewpoint "$TEST_VIEWPOINT" \
    --selected-scenes "$SCENE" \
    "${MD_FLAGS[@]}" \
    "${INP_FLAGS[@]}" \
    "${TVP_FLAGS[@]}" \
    "${EXTRA_ARGS[@]}"
done

echo "All per-scene trainings completed."
