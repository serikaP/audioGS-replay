#!/bin/bash

# ReplayNVAS per-scene viewpoint testing for AudioGS.
#
# - Iterates frames for each scene (metadata_v2.json when A3DGS_USE_METADATA!=0).
# - Only evaluates frames that have a trained checkpoint under:
#     3dgs_result/replayNVAS/<SCENE>/viewpoint_<TEST_VP>/frame_<FRAME_ID>/
#
# Example:
#   CUDA_VISIBLE_DEVICES=1 \
#   A3DGS_USE_METADATA=1 \
#   A3DGS_INPUT_SOURCE=viewpoint \
#   A3DGS_INPUT_VP=7 \
#   bash test_audio_3dgs_replaynvas_viewpoint_per_scene_all_frames.sh 8 SC-1044

set -e

TEST_VIEWPOINT=${1:-8}
shift 1 || true

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
  SCENES_SPEC="SC-1044"
fi

SCENES_SPEC="${SCENES_SPEC//,/ }"
read -r -a SCENES <<< "${SCENES_SPEC}"

CFG="configs/audio_3dgs_replaynvas_viewpoint.yaml"
META_PATH="data/avcloud_data/ReplayNVAS/v3/metadata_v2.json"

METRIC_MODE="${A3DGS_METRIC_MODE:-avcloud}"
SAVE_AUDIO="${A3DGS_SAVE_AUDIO:-0}"
VISUALIZE="${A3DGS_VISUALIZE:-0}"
STATIC_SOURCE="${A3DGS_STATIC_SOURCE:-0}"
NO_DPAM="${A3DGS_NO_DPAM:-0}"

echo "AudioGS per-frame testing (test viewpoint=${TEST_VIEWPOINT})"
echo "Scenes: ${SCENES[*]}"
echo "Config: ${CFG}"
echo "Metric mode: ${METRIC_MODE}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  echo "Extra overrides: ${EXTRA_ARGS[*]}"
fi

has_checkpoint() {
  frame_dir="$1"
  [ -f "${frame_dir}/best_model.pth" ] && return 0
  [ -f "${frame_dir}/latest_model.pth" ] && return 0
  [ -f "${frame_dir}/checkpoint_latest.pth" ] && return 0
  find "${frame_dir}" -maxdepth 1 -type f -name 'checkpoint_*.pth' | grep -q .
}

for SCENE in "${SCENES[@]}"; do
  FRAME_ROOT="data/avcloud_data/ReplayNVAS/v3/${SCENE}"
  if [ ! -d "${FRAME_ROOT}" ]; then
    echo "Warning: Frame root directory not found: ${FRAME_ROOT} (skip scene ${SCENE})"
    continue
  fi

  echo ""
  echo "=== Scene ${SCENE} (test vp=${TEST_VIEWPOINT}) ==="

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
    echo "Warning: No frame IDs found for scene ${SCENE}."
    continue
  fi

  ROOT_OUT="work_dirs/audio_3dgs_replaynvas_viewpoint/test_all_frames/${SCENE}/viewpoint_${TEST_VIEWPOINT}"
  mkdir -p "${ROOT_OUT}"
  echo "Output root: ${ROOT_OUT}"

  MODEL_DIR="3dgs_result/replayNVAS/${SCENE}/viewpoint_${TEST_VIEWPOINT}"
  if [ ! -d "${MODEL_DIR}" ]; then
    echo "Warning: Trained model directory not found: ${MODEL_DIR}"
    continue
  fi

  EVAL_COUNT=0

  for FRAME_ID in ${FRAME_IDS}; do
    FRAME_CKPT_DIR="${MODEL_DIR}/frame_${FRAME_ID}"
    if [ ! -d "${FRAME_CKPT_DIR}" ]; then
      continue
    fi
    if ! has_checkpoint "${FRAME_CKPT_DIR}"; then
      continue
    fi

    echo ""
    echo "=== Testing scene ${SCENE}, frame ${FRAME_ID} (test vp=${TEST_VIEWPOINT}) ==="

    OUT_DIR="${ROOT_OUT}/frame_${FRAME_ID}/audiogs_eval"
    mkdir -p "${OUT_DIR}"

    EXTRA_FLAGS=()
    if [ "${A3DGS_USE_METADATA:-1}" != "0" ]; then
      EXTRA_FLAGS+=(--use-metadata --metadata-file metadata_v2.json)
    fi
    if [ "${SAVE_AUDIO}" != "0" ]; then
      EXTRA_FLAGS+=(--save-audio)
    fi
    if [ "${VISUALIZE}" != "0" ]; then
      EXTRA_FLAGS+=(--visualize)
    fi
    if [ "${STATIC_SOURCE}" != "0" ]; then
      EXTRA_FLAGS+=(--static-source)
    fi
    if [ "${NO_DPAM}" != "0" ]; then
      EXTRA_FLAGS+=(--no-dpam)
    fi

    INP_SRC="${A3DGS_INPUT_SOURCE:-viewpoint}"
    INP_VP="${A3DGS_INPUT_VP:-7}"

    A3DGS_FRAME_ID="${FRAME_ID}" \
    python test_audio_3dgs_viewpoint.py \
      --cfg "${CFG}" \
      --model-dir "${FRAME_CKPT_DIR}" \
      --test-viewpoint "${TEST_VIEWPOINT}" \
      --selected-scenes "${SCENE}" \
      --frame-id "${FRAME_ID}" \
      --metric-mode "${METRIC_MODE}" \
      --input-source "${INP_SRC}" \
      --input-viewpoint "${INP_VP}" \
      --output-dir "${OUT_DIR}" \
      "${EXTRA_FLAGS[@]}" \
      "${EXTRA_ARGS[@]}"
    EVAL_COUNT=$((EVAL_COUNT + 1))
  done

  if [ "${EVAL_COUNT}" -eq 0 ]; then
    echo "Warning: no frames were evaluated for ${SCENE}."
    echo "         Checked metadata frames under ${MODEL_DIR}/frame_* but found no usable checkpoints."
    continue
  fi

  echo ""
  echo "Evaluated ${EVAL_COUNT} frame(s) for scene ${SCENE}."
  echo "Aggregating per-frame metrics for scene ${SCENE}..."

  ROOT_OUT_ABS="${ROOT_OUT}"
  TEST_VP_STR="${TEST_VIEWPOINT}"

  python - << PY
import glob
import json
import os

import numpy as np

root_out = "${ROOT_OUT_ABS}"
test_vp = "${TEST_VP_STR}"
scene = "${SCENE}"

def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def frame_sort_key(frame_id):
    return (0, int(frame_id)) if str(frame_id).isdigit() else (1, str(frame_id))

def aggregate(pattern, out_name):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found for pattern: {pattern}")
        return

    per_frame = {}
    metric_names = set()

    for path in files:
        parts = path.split(os.sep)
        try:
            frame_dir = parts[-3]
            frame_id = frame_dir.split("_", 1)[1]
        except Exception:
            frame_id = "unknown"

        with open(path, "r") as f:
            data = json.load(f)

        avg = data.get("average_metrics", {}) or {}
        metrics = {
            key: float(value)
            for key, value in avg.items()
            if is_number(value)
        }
        if not metrics:
            continue
        per_frame[frame_id] = metrics
        metric_names.update(metrics.keys())

    if not per_frame:
        print(f"No numeric average_metrics found for pattern: {pattern}")
        return

    overall = {}
    for metric in sorted(metric_names):
        values = [metrics[metric] for metrics in per_frame.values() if metric in metrics]
        if values:
            overall[metric] = float(np.mean(values))
            overall[f"{metric}_std"] = float(np.std(values))

    summary = {
        "scene": scene,
        "test_viewpoint": int(test_vp),
        "evaluated_frames": sorted(per_frame.keys(), key=frame_sort_key),
        "num_evaluated_frames": len(per_frame),
        "per_frame_metrics": {
            frame_id: per_frame[frame_id]
            for frame_id in sorted(per_frame.keys(), key=frame_sort_key)
        },
        "overall_metrics": overall,
    }

    summary_path = os.path.join(root_out, out_name)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

aggregate(
    os.path.join(root_out, "frame_*", "audiogs_eval", "test_results.json"),
    f"audiogs_all_frames_summary_vp_{test_vp}.json",
)
aggregate(
    os.path.join(root_out, "frame_*", f"mono_baseline_results_viewpoint_{test_vp}", "mono_baseline_results.json"),
    f"mono_baseline_all_frames_summary_vp_{test_vp}.json",
)
PY
done

echo ""
echo "Done."
