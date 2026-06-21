#!/bin/bash

# 数据集为 ReplayNVAS
# 每一帧分别训练一个独立模型
# 用法示例：
#   CUDA_VISIBLE_DEVICES=0 \
#   A3DGS_USE_METADATA=1 \
#   A3DGS_TRAIN_VP="1,2,3,4,5,6,7" \
#   A3DGS_INPUT_SOURCE=viewpoint \
#   A3DGS_INPUT_VP=7 \
#   bash train_audio_3dgs_replaynvas_viewpoint_per_scene_all_frames.sh 8
#
# 参数：
#   $1: 测试视角 ID（1-8），默认 8
#   $2...: 先读取连续的场景 ID（SC-xxxx，可多个），剩余参数原样透传给单帧训练脚本
#         例如：
#         bash train_audio_3dgs_replaynvas_viewpoint_per_scene_all_frames.sh 8 SC-1052 train.vis_coupling_weight 1e-4
#
# 场景也可通过环境变量指定：
#   A3DGS_SCENES="SC-1044,SC-1052,..." 或 A3DGS_SCENE="SC-1044"

set -e

TEST_VIEWPOINT=${1:-8}
shift 1 || true

# 与单帧脚本保持一致：
#   1) 连续的 SC-* 位置参数视为 scene 列表
#   2) 剩余参数视为额外 override，透传给 train_audio_3dgs_replaynvas_viewpoint_per_scene.sh
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
  SCENES_SPEC="SC-1052"
fi

SCENES_SPEC="${SCENES_SPEC//,/ }"
read -r -a SCENES <<< "${SCENES_SPEC}"

echo "Starting per-frame training (test vp=${TEST_VIEWPOINT})"
echo "Scenes: ${SCENES[*]}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  echo "Extra overrides: ${EXTRA_ARGS[*]}"
fi

# 根据是否使用 metadata 选择帧列表：
# A3DGS_USE_METADATA=1 时，只遍历 metadata_v2.json 中该场景实际使用的帧；
# 否则退回到扫描文件夹下的所有数字帧目录。
META_PATH="data/avcloud_data/ReplayNVAS/v3/metadata_v2.json"

for SCENE in "${SCENES[@]}"; do
  FRAME_ROOT="data/avcloud_data/ReplayNVAS/v3/${SCENE}"

  if [ ! -d "${FRAME_ROOT}" ]; then
    echo "Warning: Frame root directory not found: ${FRAME_ROOT} (skip scene ${SCENE})"
    continue
  fi

  echo ""
  echo "=== Starting scene ${SCENE} (test vp=${TEST_VIEWPOINT}) ==="
  echo "Frame root: ${FRAME_ROOT}"

  FRAME_IDS=""
  if [ "${A3DGS_USE_METADATA:-1}" != "0" ] && [ -f "${META_PATH}" ]; then
    FRAME_IDS=$(python - << PY
import json

meta_path = "${META_PATH}"
scene = "${SCENE}"
frames = set()
with open(meta_path, "r") as f:
    meta = json.load(f)
for k in meta.keys():
    parts = k.strip("/").split("/")
    if len(parts) < 2:
        continue
    sc, frame = parts[-2], parts[-1]
    if sc == scene and frame.isdigit():
        frames.add(int(frame))
for fid in sorted(frames):
    print(fid)
PY
)
  else
    FRAME_IDS=$(ls "${FRAME_ROOT}" | sort -n | grep -E '^[0-9]+$' || true)
  fi

  if [ -z "${FRAME_IDS}" ]; then
    echo "Warning: No frame IDs found for scene ${SCENE} (check metadata or directory)."
    continue
  fi

  # 遍历该场景下的所有候选帧
  for FRAME_ID in ${FRAME_IDS}; do

    echo ""
    echo "=== Training scene ${SCENE}, frame ${FRAME_ID} (test vp=${TEST_VIEWPOINT}) ==="

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    A3DGS_USE_METADATA="${A3DGS_USE_METADATA:-1}" \
    A3DGS_TRAIN_VP="${A3DGS_TRAIN_VP:-1,2,3,4,5,6,7,8}" \
    A3DGS_INPUT_SOURCE="${A3DGS_INPUT_SOURCE:-viewpoint}" \
    A3DGS_INPUT_VP="${A3DGS_INPUT_VP:-7}" \
    A3DGS_SCENE="${SCENE}" \
    A3DGS_SCENES="${SCENE}" \
    A3DGS_FRAME_ID="${FRAME_ID}" \
    bash train_audio_3dgs_replaynvas_viewpoint_per_scene.sh "${TEST_VIEWPOINT}" "${EXTRA_ARGS[@]}" || {
      echo "Warning: training failed for scene ${SCENE}, frame ${FRAME_ID}, skipping."
      continue
    }
  done

  echo ""
  echo "All frames for scene ${SCENE} have been trained."
done

echo ""
echo "All requested scenes completed."
