# AudioGS ReplayNVAS Reproduction

This repository contains the AudioGS ReplayNVAS viewpoint-split training and
testing code used for single-frame and all-frame experiments.

## Environment

Use the lab environment that already has the original dependencies installed,
or install:

```bash
pip install -r requirements.txt
```

## Data Layout

Put ReplayNVAS under:

```text
data/avcloud_data/ReplayNVAS/
  camera_positions_fixed_rotation.json
  v3/
    metadata_v2.json
    SC-1044/
      13/
        near.wav
        1.wav
        2.wav
        ...
        8.wav
```

The default config uses Replay metadata poses:

```yaml
dataset:
  pose_source: replay_metadata
  replay_metadata_file: data/Replay/metadata.sqlite
```

So also provide:

```text
data/Replay/metadata.sqlite
```

If you want to use `camera_positions_fixed_rotation.json` instead, override:

```bash
dataset.pose_source fixed_rotation
```

## Single-Frame Training

```bash
A3DGS_USE_METADATA=1 \
A3DGS_TRAIN_VP="1,2,3,4,5,6,7" \
A3DGS_INPUT_SOURCE=viewpoint \
A3DGS_INPUT_VP=7 \
A3DGS_FRAME_ID=13 \
A3DGS_SCENES="SC-1044" \
bash train_audio_3dgs_replaynvas_viewpoint_per_scene.sh 8
```

Checkpoints are saved to:

```text
3dgs_result/replayNVAS/SC-1044/viewpoint_8/frame_13/
```

## Single-Frame Testing

```bash
A3DGS_USE_METADATA=1 \
A3DGS_INPUT_SOURCE=viewpoint \
A3DGS_INPUT_VP=7 \
A3DGS_FRAME_ID=13 \
python test_audio_3dgs_viewpoint.py \
  --cfg configs/audio_3dgs_replaynvas_viewpoint.yaml \
  --test-viewpoint 8 \
  --selected-scenes SC-1044 \
  --output-dir work_dirs/audio_3dgs_replaynvas_viewpoint/test_per_scene/SC-1044/viewpoint_8_frame_13 \
  --model-dir . \
  --frame-id 13 \
  --save-audio --visualize --metric-mode avcloud
```

## All-Frame Training

Train every metadata frame for one or more scenes:

```bash
CUDA_VISIBLE_DEVICES=0 \
A3DGS_USE_METADATA=1 \
A3DGS_TRAIN_VP="1,2,3,4,5,6,7" \
A3DGS_INPUT_SOURCE=viewpoint \
A3DGS_INPUT_VP=7 \
A3DGS_SCENES="SC-1044,SC-1052" \
bash train_audio_3dgs_replaynvas_viewpoint_per_scene_all_frames.sh 8
```

You can also pass scenes as positional arguments:

```bash
bash train_audio_3dgs_replaynvas_viewpoint_per_scene_all_frames.sh 8 SC-1044 SC-1052
```

## All-Frame Testing

The script evaluates only frames that already have checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 \
A3DGS_USE_METADATA=1 \
A3DGS_INPUT_SOURCE=viewpoint \
A3DGS_INPUT_VP=7 \
bash test_audio_3dgs_replaynvas_viewpoint_per_scene_all_frames.sh 8 SC-1044 SC-1052
```

Summaries are written under:

```text
work_dirs/audio_3dgs_replaynvas_viewpoint/test_all_frames/<SCENE>/viewpoint_<VP>/
```

## Useful Environment Variables

- `A3DGS_SCENES`: comma/space separated scenes, for example `SC-1044,SC-1052`.
- `A3DGS_SCENE`: single scene fallback.
- `A3DGS_FRAME_ID`: single frame id for per-frame training/testing.
- `A3DGS_TRAIN_VP`: training viewpoints, for example `1,2,3,4,5,6,7`.
- `A3DGS_INPUT_SOURCE`: `near` or `viewpoint`.
- `A3DGS_INPUT_VP`: input viewpoint id when `A3DGS_INPUT_SOURCE=viewpoint`.
- `A3DGS_METRIC_MODE`: `avcloud` or `nvas`.
- `A3DGS_SAVE_AUDIO`: set to `1` in all-frame testing to save wav outputs.
- `A3DGS_VISUALIZE`: set to `1` in all-frame testing to save figures.

## Notes Before Uploading

Do not commit datasets, checkpoints, generated audio, or result folders. They
are ignored by `.gitignore`.
