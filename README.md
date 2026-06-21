# AudioGS ReplayNVAS Reproduction

This repository contains the AudioGS ReplayNVAS viewpoint-split training and
testing code used for single-frame and all-frame experiments.

## Environment

Use the lab environment that already has the original dependencies installed,
or install:

```bash
pip install -r requirements.txt
```

## Data Download

Download ReplayNVAS v3 audio data:

```bash
mkdir -p data/avcloud_data/ReplayNVAS
wget https://dl.fbaipublicfiles.com/large_objects/nvas/v3.zip -O data/avcloud_data/ReplayNVAS/v3.zip
unzip data/avcloud_data/ReplayNVAS/v3.zip -d data/avcloud_data/ReplayNVAS
```

Download Replay metadata:

```bash
mkdir -p data/Replay
wget https://dl.fbaipublicfiles.com/replay/v0/metadata.zip -O data/Replay/metadata.zip
unzip data/Replay/metadata.zip -d data/Replay
```

After extraction, make sure these paths exist:

```text
data/avcloud_data/ReplayNVAS/v3/
data/Replay/metadata.sqlite
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
A3DGS_USE_METADATA=1 \
A3DGS_TRAIN_VP="1,2,3,4,5,6,7" \
A3DGS_INPUT_SOURCE=viewpoint \
A3DGS_INPUT_VP=7 \
A3DGS_SCENES="SC-1044,SC-1052,SC-1074,SC-1084,SC-1107" \
bash train_audio_3dgs_replaynvas_viewpoint_per_scene_all_frames.sh 8
```

## All-Frame Testing

The script evaluates only frames that already have checkpoints:

```bash
A3DGS_USE_METADATA=1 \
A3DGS_INPUT_SOURCE=viewpoint \
A3DGS_INPUT_VP=7 \
bash test_audio_3dgs_replaynvas_viewpoint_per_scene_all_frames.sh 8 SC-1044,SC-1052,SC-1074,SC-1084,SC-1107
```

Summaries are written under:

```text
work_dirs/audio_3dgs_replaynvas_viewpoint/test_all_frames/<SCENE>/viewpoint_<VP>/
```
