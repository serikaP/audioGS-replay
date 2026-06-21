#!/usr/bin/env python3
"""
Testing script for Audio 3D Gaussian Splatting with viewpoint-based trained models
Specifically designed for models trained with train_audio_3dgs_replaynvas_viewpoint.sh
"""

import sys
import os
import argparse
import torch
import numpy as np
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')
import librosa
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'tools'))
sys.path.insert(0, str(PROJECT_ROOT))

import _init_paths
from configs import cfg, update_config
from importlib import import_module as impm

plt.rcParams['font.family'] = 'serif'
# 尝试按顺序寻找字体：先找 Times New Roman，找不到就找 DejaVu Serif (Linux标配)，再找不到就用通用 serif
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'serif']
plt.rcParams['axes.unicode_minus'] = False


def _resolve_selected_scenes(args, cfg) -> List[str]:
    if args.selected_scenes:
        return list(args.selected_scenes)

    selected = list(getattr(getattr(cfg, 'dataset', object()), 'selected_scenes', []) or [])
    if selected:
        return [str(s).strip() for s in selected if str(s).strip()]

    scope = str(getattr(getattr(cfg, 'dataset', object()), 'scene_scope', '') or '').strip()
    if scope and scope.lower() != 'multi':
        return [scope]

    dataset_name = str(getattr(getattr(cfg, 'dataset', object()), 'name', '') or '').lower()
    if 'replaynvas' in dataset_name:
        return ['SC-1027', 'SC-1024', 'SC-1040', 'SC-1042', 'SC-1044', 'SC-1052']

    data_root = str(getattr(getattr(cfg, 'dataset', object()), 'data_root', '') or '').rstrip('/').rstrip('\\')
    if data_root:
        fallback = os.path.basename(data_root)
        if fallback:
            return [fallback]

    return ['default']


def _resolve_viewpoint_loader(cfg):
    dataset_module = impm(str(cfg.dataset.name))
    make_loader = getattr(dataset_module, 'make_viewpoint_data_loader', None)
    if callable(make_loader):
        return make_loader
    raise AttributeError(
        f"Dataset module {cfg.dataset.name} does not provide make_viewpoint_data_loader(...)"
    )


def _apply_common_test_cfg(args, cfg):
    try:
        cfg.defrost()
        if getattr(args, "frame_id", None) is not None:
            cfg.dataset.frame_scope = int(args.frame_id)
        if getattr(args, 'input_source', None):
            cfg.dataset.input_source = str(args.input_source)
            cfg.dataset.input_viewpoint = int(getattr(args, 'input_viewpoint', 0) or 0)
        else:
            env_src = os.environ.get('A3DGS_INPUT_SOURCE', '').strip()
            env_vp = os.environ.get('A3DGS_INPUT_VP', '').strip()
            if env_src in ('near', 'viewpoint'):
                cfg.dataset.input_source = env_src
            if env_vp.isdigit():
                cfg.dataset.input_viewpoint = int(env_vp)
        cfg.dataset.test_viewpoint = int(args.test_viewpoint)
        cfg.dataset.selected_scenes = list(args.selected_scenes)
        cfg.dataset.scene_scope = args.selected_scenes[0] if len(args.selected_scenes) == 1 else 'multi'
        if getattr(args, 'use_metadata', False):
            cfg.dataset.use_metadata = True
            cfg.dataset.metadata_file = str(getattr(args, 'metadata_file', 'metadata_v2.json'))
        cfg.freeze()
    except Exception:
        pass
    try:
        if getattr(args, "frame_id", None) is not None:
            os.environ['A3DGS_FRAME_ID'] = str(int(args.frame_id))
    except Exception:
        pass


def _standard_checkpoint_dir(cfg):
    if hasattr(cfg.dataset, 'name') and 'replaynvas' in cfg.dataset.name.lower():
        model_dir = os.path.join("3dgs_result", "replayNVAS")
    else:
        video_name = str(getattr(cfg.dataset, 'video', 1))
        sr = cfg.dataset.sr
        model_dir = os.path.join("3dgs_result", f"audio_3dgs_{video_name}_{sr}")

    scene_scope = str(getattr(cfg.dataset, 'scene_scope', '') or '').strip()
    if scene_scope and scene_scope.lower() != 'multi':
        model_dir = os.path.join(model_dir, scene_scope)
    elif scene_scope.lower() == 'multi':
        model_dir = os.path.join(model_dir, 'multi')

    viewpoint = getattr(cfg.dataset, 'test_viewpoint', None)
    try:
        is_valid_vp = int(viewpoint) > 0
    except Exception:
        is_valid_vp = False
    if is_valid_vp:
        model_dir = os.path.join(model_dir, f"viewpoint_{int(viewpoint)}")

    frame_scope = getattr(cfg.dataset, 'frame_scope', None)
    if not frame_scope:
        frame_scope = os.environ.get('A3DGS_FRAME_ID', '').strip()
    if frame_scope:
        try:
            frame_scope = str(int(frame_scope))
        except Exception:
            frame_scope = str(frame_scope)
        model_dir = os.path.join(model_dir, f"frame_{frame_scope}")

    return model_dir


def _safe_sample_name(scene_id: str, sample_index: int) -> str:
    safe_sid = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(scene_id))
    return f"{safe_sid}_sample_{sample_index}"


def _save_audio_bundle(dest_dir: str, sample_name: str, sr: int, wavs: dict):
    os.makedirs(dest_dir, exist_ok=True)
    cleaned = {}
    for key, value in wavs.items():
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.stack([arr, arr], axis=0)
        elif arr.ndim == 2 and arr.shape[0] not in (1, 2) and arr.shape[1] in (1, 2):
            arr = arr.T
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = np.vstack([arr, arr])
        if arr.ndim != 2:
            continue
        cleaned[key] = arr
    if not cleaned:
        return

    try:
        import soundfile as sf
        for key, arr in cleaned.items():
            sf.write(os.path.join(dest_dir, f"{sample_name}_{key}.wav"), arr.T, int(sr))
        return
    except Exception:
        pass

    try:
        from scipy.io import wavfile
        for key, arr in cleaned.items():
            wav_i16 = np.clip(arr, -1.0, 1.0)
            wav_i16 = (wav_i16 * 32767.0).astype(np.int16)
            wavfile.write(os.path.join(dest_dir, f"{sample_name}_{key}.wav"), int(sr), wav_i16.T)
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description='Test Audio 3DGS Viewpoint-based Model')
    parser.add_argument(
        '--model-dir',
        required=True,
        type=str,
        help='Path to trained model directory (e.g., work_dirs/audio_3dgs_replaynvas_viewpoint/2025-09-07-23-38/viewpoint_8)')
    parser.add_argument(
        '--cfg',
        default='configs/audio_3dgs_replaynvas_viewpoint.yaml',
        type=str,
        help='Config file path')
    parser.add_argument(
        '--test-viewpoint',
        type=int,
        default=None,
        help='Test viewpoint (1-8). If not specified, will try to infer from model-dir')
    parser.add_argument(
        '--selected-scenes',
        nargs='+',
        default=None,
        help='Logical scene ids to use for testing')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for test results')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Explicit checkpoint .pth path to load (overrides auto-detect)')
    parser.add_argument(
        '--frame-id',
        type=int,
        default=None,
        help='Optional frame/clip id for per-clip training/testing (used for dataset filtering and checkpoint dir frame_xxx)')
    parser.add_argument(
        '--save-audio',
        action='store_true',
        help='Save synthesized audio files')
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Save qualitative plots (spec + waveform) for a subset of samples')
    parser.add_argument(
        '--vis-font-size',
        type=float,
        default=None,
        help='Font size (points) for visualization plots (axes labels/ticks).')
    parser.add_argument(
        '--vis-title-font-size',
        type=float,
        default=None,
        help='Font size (points) for visualization plot titles (defaults to --vis-font-size).')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use for testing')
    parser.add_argument(
        '--gl-refine',
        action='store_true',
        help='Apply Griffin-Lim refinement on predicted magnitudes (per ear)')
    parser.add_argument(
        '--gl-iters',
        type=int,
        default=32,
        help='Number of iterations for Griffin-Lim when --gl-refine is set')
    parser.add_argument(
        '--no-dpam',
        action='store_true',
        help='Disable DPAM metric even if cdpam is available')
    parser.add_argument(
        '--metric-mode',
        type=str,
        default='avcloud',
        choices=['avcloud', 'nvas'],
        help='Metric definition: avcloud (default) or nvas (STFT L2 for MAG)')
    # 选择特定视角作为输入音频
    parser.add_argument('--input-source', type=str, default=None, choices=['near','viewpoint'],
                        help='Use near.wav (default) or a specific viewpoint as input audio')
    parser.add_argument('--input-viewpoint', type=int, default=0,
                        help='When --input-source viewpoint, use this viewpoint id as input')
    # 使用 metadata_v2.json 选择帧（与avcloud/ViGAS相同）
    parser.add_argument(
        '--use-metadata',
        action='store_true',
        help='Use metadata_v2.json to build per-scene frame list (fair comparison with NVAS)')
    parser.add_argument(
        '--metadata-file',
        type=str,
        default='metadata_v2.json',
        help='Metadata filename under data_root/v3 (default: metadata_v2.json)')
    parser.add_argument(
        '--eval-denorm',
        action='store_true',
        default=1,
        help='Denormalize pred/gt by per-scene RMS max before evaluation if dataset used scene normalization')
    parser.add_argument(
        '--print-lr-ratio',
        action='store_true',
        help='Print per-sample L/R energy ratios (pred vs GT) in dB')
    parser.add_argument(
        '--add-env-back',
        action='store_true',
        help='At evaluation, add environment residual (raw - bandpass) back to prediction, and compare against raw GT')
    parser.add_argument(
        '--baseline-mode',
        type=str,
        default='auto',
        choices=['auto', 'mono', 'passthrough'],
        help='Mono baseline mode: mono (duplicate mono), passthrough (use input stereo), or auto (passthrough when input_source=viewpoint>0)'
    )
    parser.add_argument(
        '--baseline-only',
        action='store_true',
        help='Only run mono baseline testing (skip Audio 3DGS model evaluation and comparison)',
    )
    parser.add_argument(
        '--static-source',
        action='store_true',
        help='Use static source STFT cached in the model (ignore per-sample source_audio and condition only on cam_pose)',
    )
    parser.add_argument(
        '--save-masks',
        action='store_true',
        help='When used with --visualize and GS-only models, also save Mono/Diff mask spectrograms (and their masked magnitudes).',
    )

    parser.add_argument(
        'opts',
        help="Modify config options using the command-line (YACS style), e.g. dataset.pose_source gs_cameras",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    args = parser.parse_args()
    
    # Fallback 
    try:
        if args.input_source is None:
            env_src = os.environ.get('A3DGS_INPUT_SOURCE', '').strip()
            if env_src in ('near', 'viewpoint'):
                args.input_source = env_src
        if (not getattr(args, 'input_viewpoint', 0)) and os.environ.get('A3DGS_INPUT_VP', '').strip().isdigit():
            args.input_viewpoint = int(os.environ.get('A3DGS_INPUT_VP').strip())
    except Exception:
        pass
    
    # Normalize potential non-ASCII hyphens in selected_scenes
    if args.selected_scenes is not None:
        dash_variants = ['\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '\u2212']
        def norm_scene(s):
            if not isinstance(s, str):
                return s
            for dv in dash_variants:
                s = s.replace(dv, '-')
            return s.strip()
        args.selected_scenes = [norm_scene(s) for s in args.selected_scenes]
    
    # 从模型路径推断出 test viewpoint
    if args.test_viewpoint is None:
        model_dir_name = Path(args.model_dir).name
        if 'viewpoint_' in model_dir_name:
            try:
                args.test_viewpoint = int(model_dir_name.split('viewpoint_')[1])
                print(f"Inferred test viewpoint: {args.test_viewpoint}")
            except:
                print("Could not infer test viewpoint from model directory name")
                print("Please specify --test-viewpoint manually")
                sys.exit(1)
        else:
            print("Could not find viewpoint information in model directory")
            print("Please specify --test-viewpoint manually")
            sys.exit(1)
    
    # Set output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, f'test_results_viewpoint_{args.test_viewpoint}')
    
    return args


def load_model_checkpoint(model_dir, device, cfg):
    """Load the trained model from checkpoint"""
    
    # Audio3DGSTrainer 保存checkpoint的目录结构：
    # For ReplayNVAS: 3dgs_result/replayNVAS/
    # For others: 3dgs_result/audio_3dgs_{video}_{sr}/
    
    preferred_dir = None
    try:
        import re
        # Try to infer SC-xxxx and viewpoint from provided model_dir path
        m_scene = re.search(r"SC-\d{4}", model_dir)
        m_vp = re.search(r"viewpoint_(\d)", model_dir)
        if m_scene and m_vp:
            scene_from_path = m_scene.group(0)
            vp_from_path = int(m_vp.group(1))
            preferred_dir = os.path.join("3dgs_result", "replayNVAS", scene_from_path, f"viewpoint_{vp_from_path}")
    except Exception:
        preferred_dir = None

    # Determine the standard save directory based on dataset/cfg
    standard_model_dir = _standard_checkpoint_dir(cfg)
    
    # Look for checkpoint files
    checkpoint_files = [
        'latest_model.pth',
        'best_model.pth',         
        'model_final.pth',
        'checkpoint_latest.pth'
    ]
    
    # Also check for numbered checkpoints if the directory exists
    if os.path.exists(standard_model_dir):
        try:
            numbered_checkpoints = [f for f in os.listdir(standard_model_dir) 
                                  if f.startswith('checkpoint_') and f.endswith('.pth')]
            checkpoint_files.extend(sorted(numbered_checkpoints, reverse=True))  # Latest first
        except:
            pass
    
    checkpoint_path = None
    
    # First try the preferred_dir inferred from model_dir path
    if preferred_dir and os.path.exists(preferred_dir):
        print(f"Checking preferred model directory (from path): {preferred_dir}")
        for fname in checkpoint_files:
            full_path = os.path.join(preferred_dir, fname)
            if os.path.exists(full_path):
                checkpoint_path = full_path
                break

    # Then try the standard model directory derived from cfg
    if checkpoint_path is None and os.path.exists(standard_model_dir):
        print(f"Checking standard model directory: {standard_model_dir}")
        for fname in checkpoint_files:
            full_path = os.path.join(standard_model_dir, fname)
            if os.path.exists(full_path):
                checkpoint_path = full_path
                break
    
    # If not found, try the provided model_dir
    if checkpoint_path is None:
        print(f"Standard directory not found or no checkpoints, checking provided model_dir: {model_dir}")
        for fname in checkpoint_files:
            full_path = os.path.join(model_dir, fname)
            if os.path.exists(full_path):
                checkpoint_path = full_path
                break
    
    # Look in 3dgs_result subdirectory of model_dir
    if checkpoint_path is None:
        result_subdir_path = os.path.join(model_dir, '3dgs_result')
        if os.path.exists(result_subdir_path):
            for root, dirs, files in os.walk(result_subdir_path):
                for fname in checkpoint_files[:4]:  # Only check main checkpoint names
                    if fname in files:
                        checkpoint_path = os.path.join(root, fname)
                        break
                if checkpoint_path:
                    break
    
    if checkpoint_path is None:
        print(f"No checkpoint found in:")
        print(f"  1. Standard directory: {standard_model_dir}")
        print(f"  2. Provided model_dir: {model_dir}")
        print(f"  3. Subdirectories of model_dir")
        print("Looking for:", checkpoint_files[:4])  # Don't print all numbered checkpoints
        return None
        
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def compute_nvas_metrics(predicted_audio, target_audio, sr=16000, metric_mode='avcloud'):
    """Compute NVAS evaluation metrics

    metric_mode:
      - 'avcloud': magnitude abs-diff mean (default)
      - 'nvas': STFT L2 distance 
    """
    
    metrics = {}
    
    # Helper: convert to numpy stereo [2, T]
    def to_np_stereo(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        # If mono [T]
        if x.ndim == 1:
            return np.stack([x, x], axis=0)
        # If [2, T]
        if x.ndim == 2 and x.shape[0] == 2:
            return x
        # If [T, 2]
        if x.ndim == 2 and x.shape[1] == 2:
            return x.T
        # If batched [B, 2, T] or [B, T, 2] with B==1
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
            if x.shape[0] == 2:
                return x
            if x.shape[1] == 2:
                return x.T
        # Fallback: flatten to mono and duplicate
        x = x.reshape(-1)
        return np.stack([x, x], axis=0)

    # Canonicalize shapes
    predicted_audio = to_np_stereo(predicted_audio)
    target_audio = to_np_stereo(target_audio)
    pred_left, pred_right = predicted_audio[0], predicted_audio[1]
    tgt_left, tgt_right = target_audio[0], target_audio[1]
    
    # 1. MAG
    def compute_mag_avcloud_style(predicted_audio, target_audio):
        """使用AV-Cloud的MAG计算方式"""
        # 转换为tensor
        if torch.is_tensor(predicted_audio):
            pred_tensor = predicted_audio
        else:
            pred_tensor = torch.from_numpy(predicted_audio).float()
            
        if torch.is_tensor(target_audio):
            tgt_tensor = target_audio
        else:
            tgt_tensor = torch.from_numpy(target_audio).float()
        
        # 确保相同长度
        min_len = min(pred_tensor.shape[-1], tgt_tensor.shape[-1])
        pred_tensor = pred_tensor[..., :min_len]
        tgt_tensor = tgt_tensor[..., :min_len]
        
        # AV-Cloud的eval_mag函数
        def eval_mag(wav):
            # wav shape: [channels, length] -> 需要增加batch维度: [1, channels, length]
            if wav.dim() == 2:
                wav = wav.unsqueeze(0)  # [1, channels, length]
            
            # 分别处理每个声道
            mags = []
            for ch in range(wav.shape[1]):
                ch_wav = wav[:, ch:ch+1, :]  # [1, 1, length]
                stft = torch.stft(ch_wav.squeeze(0), n_fft=512, hop_length=160, win_length=400,
                                window=torch.hamming_window(400, device=wav.device), 
                                pad_mode='constant', return_complex=True)
                mag = stft.abs()
                mags.append(mag)
            return mags
        
        # 计算左右声道的幅度谱图
        pred_mags = eval_mag(pred_tensor)  # [左声道mag, 右声道mag]
        tgt_mags = eval_mag(tgt_tensor)
        
        pred_spec_l, pred_spec_r = pred_mags[0], pred_mags[1]
        tgt_spec_l, tgt_spec_r = tgt_mags[0], tgt_mags[1]
        
        # 按照AV-Cloud的计算方式
        # .pow(2).sqrt() 等价于 .abs()，但这里保持与原代码一致
        mag_l = (pred_spec_l - tgt_spec_l).pow(2).sqrt().mean()
        mag_r = (pred_spec_r - tgt_spec_r).pow(2).sqrt().mean()
        
        # 两个声道相加
        mag_total = ((mag_l + mag_r)).item()
        
        return mag_total

    def compute_mag_nvas_style(predicted_audio, target_audio):
        # Use STFT L2 over real/imag parts per channel and sum channels (NVAS style)
        import numpy as np
        import librosa
        def stft_l2(ch_pred, ch_tgt):
            p = librosa.stft(np.asfortranarray(ch_pred), n_fft=512, hop_length=160, win_length=400, center=True)
            t = librosa.stft(np.asfortranarray(ch_tgt), n_fft=512, hop_length=160, win_length=400, center=True)
            pr = np.real(p); pi = np.imag(p)
            tr = np.real(t); ti = np.imag(t)
            return float(np.mean((pr - tr) ** 2 + (pi - ti) ** 2))
        return stft_l2(predicted_audio[0], target_audio[0]) + stft_l2(predicted_audio[1], target_audio[1])

    metrics['MAG'] = compute_mag_nvas_style(predicted_audio, target_audio) if metric_mode == 'nvas' else compute_mag_avcloud_style(predicted_audio, target_audio)
    
    # 2. ENV: Envelope Distance (使用AV-Cloud的计算方式)
    def envelope_distance_avcloud_style(predicted_binaural, gt_binaural):
        from scipy.signal import hilbert
        
        # 左声道
        pred_env_channel1 = np.abs(hilbert(predicted_binaural[0]))
        gt_env_channel1 = np.abs(hilbert(gt_binaural[0]))
        channel1_distance = np.sqrt(np.mean((gt_env_channel1 - pred_env_channel1)**2))  # RMSE
        
        # 右声道
        pred_env_channel2 = np.abs(hilbert(predicted_binaural[1]))
        gt_env_channel2 = np.abs(hilbert(gt_binaural[1]))
        channel2_distance = np.sqrt(np.mean((gt_env_channel2 - pred_env_channel2)**2))  # RMSE
        
        # 两个声道的距离相加
        envelope_distance = channel1_distance + channel2_distance
        return float(envelope_distance)
    
    metrics['ENV'] = envelope_distance_avcloud_style(predicted_audio, target_audio)
    
    # 3. LRE 与 LR 比值（dB）
    eps = 1e-5  # match AV-Cloud
    pred_l_energy = float(np.sum(pred_left.astype(np.float64) ** 2))
    pred_r_energy = float(np.sum(pred_right.astype(np.float64) ** 2))
    tgt_l_energy = float(np.sum(tgt_left.astype(np.float64) ** 2))
    tgt_r_energy = float(np.sum(tgt_right.astype(np.float64) ** 2))
    pred_lr_ratio_db = float(10.0 * np.log10((pred_l_energy + eps) / (pred_r_energy + eps)))
    tgt_lr_ratio_db = float(10.0 * np.log10((tgt_l_energy + eps) / (tgt_r_energy + eps)))

    metrics['LRE'] = float(abs(pred_lr_ratio_db - tgt_lr_ratio_db))
    metrics['LR_ratio_pred_db'] = pred_lr_ratio_db
    metrics['LR_ratio_gt_db'] = tgt_lr_ratio_db
    
    # 4. RTE: RT60 Error (使用AV-Cloud的方法)
    def compute_rte_avcloud_style(predicted_audio, target_audio, device='cpu'):
        """使用AV-Cloud的RT60估算器计算RTE"""
        try:
            # 导入必要的模块
            from libs.models.vigas.visual_net import VisualNet
            
            # 加载RT60估算器
            rt60_estimator = VisualNet(use_rgb=False, use_depth=False, use_audio=True)
            pretrained_weights = 'data/avcloud_data/models/rt60_estimator.pth'
            checkpoint = torch.load(pretrained_weights, map_location='cpu')
            rt60_estimator.load_state_dict(checkpoint['predictor'])
            rt60_estimator.to(device=device).eval()
            
            # 定义RT60估算函数
            def estimate_rt60(estimator, wav):
                if torch.is_tensor(wav):
                    wav = wav.to(device)
                else:
                    wav = torch.from_numpy(wav).float().to(device)
                
                # 确保wav是2D: [channels, length] 或 [batch, length]
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                
                stft = torch.stft(wav, n_fft=512, hop_length=160, win_length=400,
                                window=torch.hamming_window(400, device=wav.device), 
                                pad_mode='constant', return_complex=True)
                spec = torch.log1p(stft.abs()).unsqueeze(1)  # Add channel dim for model
                
                with torch.no_grad():
                    estimated_rt60 = estimator(spec.float())
                return estimated_rt60
            
            # 转换为tensor并确保正确形状
            if torch.is_tensor(predicted_audio):
                pred_tensor = predicted_audio.cpu()
            else:
                pred_tensor = torch.from_numpy(predicted_audio).float()
            
            if torch.is_tensor(target_audio):
                tgt_tensor = target_audio.cpu()
            else:
                tgt_tensor = torch.from_numpy(target_audio).float()
            
            # 确保相同长度
            min_len = min(pred_tensor.shape[-1], tgt_tensor.shape[-1])
            pred_tensor = pred_tensor[..., :min_len]
            tgt_tensor = tgt_tensor[..., :min_len]
            
            # Reshape for RT60 estimation: [2, T] -> [2, T]
            pred_reshaped = pred_tensor.reshape(-1, pred_tensor.shape[-1])
            tgt_reshaped = tgt_tensor.reshape(-1, tgt_tensor.shape[-1])
            
            # 估算RT60
            pred_rt60 = estimate_rt60(rt60_estimator, pred_reshaped)
            tgt_rt60 = estimate_rt60(rt60_estimator, tgt_reshaped)
            
            # 计算平均绝对误差
            rte = (pred_rt60 - tgt_rt60).abs().mean().item()
            
            return rte
            
        except Exception as e:
            print(f"Warning: Could not compute RTE using AV-Cloud method: {e}")
            # 退回到简化版本
            return 0.0
    
    metrics['RTE'] = compute_rte_avcloud_style(predicted_audio, target_audio)
    
    return metrics


# Optional DPAM metric (AV-Cloud style)
def load_dpam(no_dpam=False, scope_dir=None):
    if no_dpam:
        return None, None, None
    try:
        import cdpam  # noqa: F401
    except Exception as e:
        print(f"Warning: DPAM unavailable (import cdpam failed): {e}")
        return None, None, None
    try:
        import soundfile as sf
    except Exception as e:
        print(f"Warning: soundfile unavailable for DPAM audio I/O: {e}")
        return None, None, None
    try:
        import cdpam
        model = cdpam.CDPAM()
    except Exception as e:
        print(f"Warning: Failed to initialize CDPAM model: {e}")
        return None, None, None
    base_dir = scope_dir or os.path.join('work_dirs', 'audio_3dgs_replaynvas_viewpoint', 'dpam_tmp')
    os.makedirs(os.path.join(base_dir, 'pred'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'gt'), exist_ok=True)
    return model, sf, base_dir


def compute_dpam(dpam_model, sf_mod, base_dir, pred_stereo, tgt_stereo, sr):
    if dpam_model is None or sf_mod is None:
        return None
    try:
        pred_path = os.path.join(base_dir, 'pred', '1.wav')
        gt_path = os.path.join(base_dir, 'gt', '1.wav')
        # Convert tensors to numpy if needed and ensure [T, 2]
        if torch.is_tensor(pred_stereo):
            pred_np = pred_stereo.detach().cpu().numpy().T
        else:
            pred_np = np.asarray(pred_stereo).T
        if torch.is_tensor(tgt_stereo):
            tgt_np = tgt_stereo.detach().cpu().numpy().T
        else:
            tgt_np = np.asarray(tgt_stereo).T
        sf_mod.write(pred_path, pred_np, int(sr))
        sf_mod.write(gt_path, tgt_np, int(sr))
        import cdpam
        wav_ref = cdpam.load_audio(gt_path)
        wav_out = cdpam.load_audio(pred_path)
        with torch.no_grad():
            val = dpam_model.forward(wav_ref, wav_out)
        return float(val.data.cpu().numpy()[0])
    except Exception as e:
        print(f"Warning: DPAM evaluation failed: {e}")
        return None


def test_mono_baseline(args):
    """Test mono baseline using same data loader"""
    print("\n=== Mono Baseline Testing ===")
    
    # Create baseline output directory 
    baseline_output_dir = os.path.join(os.path.dirname(args.output_dir), f'mono_baseline_results_viewpoint_{args.test_viewpoint}')
    os.makedirs(baseline_output_dir, exist_ok=True)
    print(f"Baseline output directory: {baseline_output_dir}")
    
    # Load config
    class Args:
        def __init__(self, yaml_file, opts=None):
            self.yaml_file = yaml_file
            self.opts = list(opts) if opts is not None else []
            self.distributed = False
            self.local_rank = 0
            self.resume = None
    
    config_args = Args(args.cfg, getattr(args, "opts", None))
    update_config(cfg, config_args)
    args.selected_scenes = _resolve_selected_scenes(args, cfg)
    _apply_common_test_cfg(args, cfg)
    make_viewpoint_data_loader = _resolve_viewpoint_loader(cfg)
    
    # Create test data loader (same as main model)
    test_loader = make_viewpoint_data_loader(
        cfg,
        split='test', 
        test_viewpoint=args.test_viewpoint,
        selected_scenes=args.selected_scenes,
        distributed=False
    )
    
    print(f"Baseline test samples: {len(test_loader.dataset)}")
    try:
        print(f"Baseline input source: {getattr(cfg.dataset, 'input_source', 'near')}, input_viewpoint: {getattr(cfg.dataset, 'input_viewpoint', 0)}")
    except Exception:
        pass
    
    # Prepare DPAM
    try:
        scope_dir = _standard_checkpoint_dir(cfg)
    except Exception:
        scope_dir = None
    dpam_model, dpam_sf, dpam_base = load_dpam(args.no_dpam, scope_dir)

    # Testing loop
    all_metrics = []
    total_samples = 0
    
    # Decide baseline behavior
    try:
        use_passthrough = (str(getattr(args, 'baseline_mode', 'auto')) == 'passthrough')
        if str(getattr(args, 'baseline_mode', 'auto')) == 'auto':
            use_passthrough = (str(getattr(cfg.dataset, 'input_source', 'near')) == 'viewpoint' and int(getattr(cfg.dataset, 'input_viewpoint', 0) or 0) > 0)
    except Exception:
        use_passthrough = False
    print(f"Baseline mode: {'passthrough' if use_passthrough else 'mono'}")
    # Select device for baseline path
    device = torch.device(getattr(args, 'device', 'cuda'))

    for batch_idx, batch in enumerate(test_loader):
        source_audio = batch['source_audio'].to(device)
        target_binaural = batch['target_binaural'].to(device)
        target_binaural_raw = batch.get('target_binaural_raw', None)
        if target_binaural_raw is not None:
            target_binaural_raw = target_binaural_raw.to(device)
        env_residual_batch = batch.get('env_residual', None)
        if env_residual_batch is not None:
            env_residual_batch = env_residual_batch.to(device)
        # DO NOT reassign from batch after moving to device
        scene_ids = batch['scene_id']
        norm_factors = batch.get('norm_factor', None)
        
        if use_passthrough:
            # Viewpoint passthrough baseline: use the input stereo directly
            # Ensure shape [B, 2, T]
            if source_audio.dim() == 2:
                predicted_binaural = source_audio.unsqueeze(0)
            elif source_audio.dim() == 3 and source_audio.shape[1] == 2:
                predicted_binaural = source_audio
            else:
                # Fallback: if source is mono for some reason, duplicate
                if source_audio.dim() == 3:
                    src_mono = source_audio[:, 0:1]
                else:
                    src_mono = source_audio.unsqueeze(1)
                predicted_binaural = src_mono.repeat(1, 2, 1)
        else:
            # Mono copy baseline: average source audio and copy to both channels
            # This matches the paper's "MonoMono: Duplicates the source audio to synthesize binaural audio"
            if source_audio.dim() > 2 and source_audio.shape[1] == 2:
                source_mono = source_audio.mean(1, keepdim=True)  # [B, 1, T]
            else:
                source_mono = source_audio[:, 0:1]  # [B, 1, T]
            predicted_binaural = source_mono.repeat(1, 2, 1)  # [B, 2, T]
        
        # Debug: Print statistics for first batch to understand the data
        if batch_idx == 0:
            print(f"Debug - Baseline Statistics:")
            if use_passthrough:
                # Print L/R RMS and LR ratio of input stereo
                pl = predicted_binaural[:, 0]
                pr = predicted_binaural[:, 1]
                print(f"  Pred L RMS: {torch.sqrt(pl.pow(2).mean()).item():.6f}")
                print(f"  Pred R RMS: {torch.sqrt(pr.pow(2).mean()).item():.6f}")
                el = pl.pow(2).sum(); er = pr.pow(2).sum()
                lr_db = 10 * torch.log10((el + 1e-8) / (er + 1e-8))
                print(f"  Pred L/R energy ratio (dB): {lr_db.item():.3f}")
            else:
                print(f"  Source mono shape: {source_mono.shape}")
                print(f"  Source mono range: [{source_mono.min().item():.6f}, {source_mono.max().item():.6f}]")
                print(f"  Source mono RMS: {torch.sqrt(source_mono.pow(2).mean()).item():.6f}")
            
            print(f"  Target binaural shape: {target_binaural.shape}")
            print(f"  Target L range: [{target_binaural[:, 0:1].min().item():.6f}, {target_binaural[:, 0:1].max().item():.6f}]")
            print(f"  Target R range: [{target_binaural[:, 1:2].min().item():.6f}, {target_binaural[:, 1:2].max().item():.6f}]")
            print(f"  Target L RMS: {torch.sqrt(target_binaural[:, 0:1].pow(2).mean()).item():.6f}")
            print(f"  Target R RMS: {torch.sqrt(target_binaural[:, 1:2].pow(2).mean()).item():.6f}")
            
            # Calculate target L/R energy ratio
            target_l_energy = target_binaural[:, 0].pow(2).sum()
            target_r_energy = target_binaural[:, 1].pow(2).sum()
            target_lr_ratio = 10 * torch.log10((target_l_energy + 1e-8) / (target_r_energy + 1e-8))
            print(f"  Target L/R energy ratio (dB): {target_lr_ratio.item():.3f}")
            
            if not use_passthrough:
                # Calculate mono baseline L/R energy ratio (should be 0 since identical)
                mono_l_energy = predicted_binaural[:, 0].pow(2).sum()  
                mono_r_energy = predicted_binaural[:, 1].pow(2).sum()
                mono_lr_ratio = 10 * torch.log10((mono_l_energy + 1e-8) / (mono_r_energy + 1e-8))
                print(f"  Mono L/R energy ratio (dB): {mono_lr_ratio.item():.3f}")
            print()
        
        # Compute metrics for each sample
        batch_size = target_binaural.size(0)
        for i in range(batch_size):
            pred = predicted_binaural[i]
            # Use raw GT and optionally add env back for baseline too (consistency)
            if getattr(args, 'add_env_back', False) and (target_binaural_raw is not None):
                tgt = target_binaural_raw[i]
                if env_residual_batch is not None:
                    er = env_residual_batch[i]
                    min_len = min(pred.shape[-1], tgt.shape[-1], er.shape[-1])
                    pred = pred[..., :min_len] + er[..., :min_len]
                    tgt = tgt[..., :min_len]
                else:
                    min_len = min(pred.shape[-1], tgt.shape[-1])
                    pred = pred[..., :min_len]
                    tgt = tgt[..., :min_len]
            else:
                tgt = target_binaural[i]
                min_len = min(pred.shape[-1], tgt.shape[-1])
                pred = pred[..., :min_len]
                tgt = tgt[..., :min_len]
            
            # Ensure same length
            min_len = min(pred.shape[-1], tgt.shape[-1])
            pred = pred[..., :min_len]
            tgt = tgt[..., :min_len]
            # Optional denormalization back to original scene scale
            if args.eval_denorm and norm_factors is not None:
                factor = norm_factors[i] if torch.is_tensor(norm_factors) else norm_factors
                if torch.is_tensor(factor):
                    factor = factor.item()
                pred = pred * factor
                tgt = tgt * factor
            
            metrics = compute_nvas_metrics(pred, tgt, cfg.dataset.sr, metric_mode=args.metric_mode)
            # Optional DPAM
            if dpam_model is not None and dpam_sf is not None:
                try:
                    dpam_val = compute_dpam(dpam_model, dpam_sf, dpam_base, pred, tgt, cfg.dataset.sr)
                    if dpam_val is not None:
                        metrics['DPAM'] = dpam_val
                except Exception:
                    pass
            metrics['scene_id'] = scene_ids[i]
            # Optional printing of LR ratios per sample
            if getattr(args, 'print_lr_ratio', False):
                sid = metrics.get('scene_id', f'sample_{batch_idx}_{i}')
                plr = metrics.get('LR_ratio_pred_db', None)
                tlr = metrics.get('LR_ratio_gt_db', None)
                if plr is not None and tlr is not None:
                    print(f"[LR-Ratio] {sid}: pred={plr:.3f} dB | gt={tlr:.3f} dB | LRE={metrics.get('LRE', float('nan')):.3f} dB")
            all_metrics.append(metrics)
            if args.save_audio:
                src = source_audio[i]
                src = src[..., :min_len]
                if args.eval_denorm and norm_factors is not None:
                    factor = norm_factors[i] if torch.is_tensor(norm_factors) else norm_factors
                    if torch.is_tensor(factor):
                        factor = factor.item()
                    src = src * factor
                sample_name = _safe_sample_name(scene_ids[i], total_samples + i)
                _save_audio_bundle(
                    os.path.join(baseline_output_dir, 'audio_samples'),
                    sample_name,
                    cfg.dataset.sr,
                    {
                        'input': src.detach().cpu().numpy(),
                        'predicted': pred.detach().cpu().numpy(),
                        'target': tgt.detach().cpu().numpy(),
                    },
                )
        
        total_samples += batch_size
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(test_loader)} batches ({total_samples} samples)")
    
    # Calculate average metrics
    if len(all_metrics) == 0:
        print("No samples were successfully processed!")
        return False
    
    avg_metrics = {}
    metric_names = ['MAG', 'ENV', 'LRE', 'RTE', 'DPAM']
    
    for metric in metric_names:
        values = [m[metric] for m in all_metrics if metric in m]
        if values:
            avg_metrics[metric] = np.mean(values)
            avg_metrics[f'{metric}_std'] = np.std(values)
    
    # Print results  
    print(f"\n=== Mono Baseline Results ===")
    print(f"Total samples evaluated: {len(all_metrics)}")
    print(f"Test viewpoint: {args.test_viewpoint}")
    print()
    
    print("Mono Baseline Performance:")
    for metric in metric_names:
        if metric in avg_metrics:
            print(f"{metric}: {avg_metrics[metric]:.6f} ± {avg_metrics[f'{metric}_std']:.6f}")
    
    # Save results
    results_file = os.path.join(baseline_output_dir, 'mono_baseline_results.json')
    results = {
        'baseline_type': ('viewpoint_passthrough' if use_passthrough else 'mono_copy'),
        'description': ('Use input viewpoint stereo as prediction' if use_passthrough else 'Simple mono baseline: copy source audio average to both channels'),
        'test_viewpoint': args.test_viewpoint,
        'selected_scenes': args.selected_scenes,
        'total_samples': len(all_metrics),
        'average_metrics': avg_metrics,
        'per_sample_metrics': all_metrics
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Baseline results saved to: {results_file}")
    if args.save_audio:
        print(f"Baseline audio samples saved to: {os.path.join(baseline_output_dir, 'audio_samples')}")
    return True


def test_model(args):
    """Main testing function"""

    # Load config
    class Args:
        def __init__(self, yaml_file, opts=None):
            self.yaml_file = yaml_file
            self.opts = list(opts) if opts is not None else []
            self.distributed = False
            self.local_rank = 0
            self.resume = None
    
    config_args = Args(args.cfg, getattr(args, "opts", None))
    update_config(cfg, config_args)
    args.selected_scenes = _resolve_selected_scenes(args, cfg)
    _apply_common_test_cfg(args, cfg)
    make_viewpoint_data_loader = _resolve_viewpoint_loader(cfg)

    # Setup
    print("=== Audio 3DGS Viewpoint Model Testing ===")
    print(f"Model directory: {args.model_dir}")
    print(f"Test viewpoint: {args.test_viewpoint}")
    print(f"Selected scenes: {args.selected_scenes}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create test data loader
    print("Creating test data loader...")
    test_loader = make_viewpoint_data_loader(
        cfg,
        split='test',
        test_viewpoint=args.test_viewpoint,
        selected_scenes=args.selected_scenes,
        distributed=False
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Device
    device = torch.device(args.device)

    # Build model
    print("Loading model...")
    model_module = impm(f'libs.models.{cfg.model.file}')
    gaussian_model = None
    scene = None
    if bool(getattr(getattr(cfg, 'model', object()), 'use_visual', False)):
        visual_cfg = getattr(
            getattr(getattr(cfg, 'model', object()), 'shared_gs', object()),
            'visual',
            object(),
        )
        visual_source = str(getattr(visual_cfg, 'feature_source', 'gaussian') or 'gaussian').strip().lower()
        if visual_source not in ('gaussian', ''):
            print(
                f"Using visual feature source {visual_source}; "
                "skipping gaussian scene load at build time."
            )
        else:
            try:
                from libs.datasets.scene import Scene
                from libs.datasets.scene.gaussian_model import GaussianModel

                visual_root = os.path.join(str(cfg.dataset.data_root), 'cam_imags')
                visual_n_points = int(getattr(visual_cfg, 'num_anchors', 256) or 256)
                gaussian_model = GaussianModel(3)
                scene = Scene(
                    visual_root,
                    visual_root,
                    gaussian_model,
                    sh_degree=3,
                    align_grids=None,
                    N_points=visual_n_points,
                    shuffle=False,
                )
                print(
                    f"Loaded frozen visual anchors from {visual_root} "
                    f"with {int(gaussian_model.get_xyz.shape[0])} points"
                )
            except Exception as e:
                print(
                    "Warning: failed to load visual anchors for model.use_visual=True; "
                    f"falling back to audio-only shared-GS. Reason: {e}"
                )
                gaussian_model = None
                scene = None

    model = model_module.build_model(cfg, gaussian_model, scene)
    model = model.to(device)

    # Load checkpoint 
    checkpoint = None
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint explicitly specified: {args.checkpoint}")
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
        except Exception as e:
            print(f"Error loading explicit checkpoint: {e}")
            checkpoint = None
    if checkpoint is None:
        checkpoint = load_model_checkpoint(args.model_dir, device, cfg)
    if checkpoint is None:
        print("Failed to load model checkpoint")
        return False
    # Load model state 
    state = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    # Drop static STFT buffers (re-initialized if needed)
    for k in ['static_source_mag', 'static_phase_L', 'static_phase_R']:
        if k in state:
            state.pop(k, None)
    # Handle potential shape mismatches for newly introduced Hopkins parameters
    # so that older/newer checkpoints remain compatible.
    try:
        import torch as _torch
        for hk in ['Q_param', 'log_Rc_param']:
            if hk in state and hasattr(model, hk):
                ckpt_tensor = state[hk]
                model_tensor = getattr(model, hk)
                if isinstance(model_tensor, _torch.Tensor):
                    if ckpt_tensor.shape != model_tensor.shape:
                        if ckpt_tensor.numel() == model_tensor.numel():
                            state[hk] = ckpt_tensor.view_as(model_tensor)
                            print(f"[Test] Reshaped checkpoint {hk} from {tuple(ckpt_tensor.shape)} to {tuple(model_tensor.shape)}")
                        else:
                            # Incompatible; ignore this key and use model init instead
                            state.pop(hk, None)
                            print(f"[Test] Dropped incompatible checkpoint key {hk} with shape {tuple(ckpt_tensor.shape)}")
    except Exception as _e:
        print(f"[Test] Warning: could not sanitize Hopkins params in state_dict: {_e}")

    model.load_state_dict(state, strict=False)
    print("Loaded model state (non-strict)")

    # If static-source mode requested, (re)initialize static STFT buffers from
    # a reference batch, so that models trained before this change can still
    # populate static_source_mag / static_phase_L/R.
    # NOTE: we do NOT reset distance-attenuation reference (ref_freq_g) here;
    # that reference should stay tied to the source-audio viewpoint as stored
    # in the checkpoint. See Audio3DGSMonoDiffGSOnly.initialize_from_batch.
    if getattr(args, 'static_source', False) and hasattr(model, 'initialize_from_batch'):
        try:
            ref_batch = next(iter(test_loader))
            # Newer models (Audio3DGSMonoDiffGSOnly) accept update_ref_g flag;
            # fall back silently for older signatures.
            try:
                model.initialize_from_batch(ref_batch, device, update_ref_g=False)
            except TypeError:
                model.initialize_from_batch(ref_batch, device)
            print("Initialized static source STFT from reference test batch for static-source mode (ref_freq_g unchanged).")
        except Exception as _e:
            print(f"Warning: could not initialize static source STFT from batch: {_e}")

    # Optional static-source mode: reuse cached source STFT from training.
    if getattr(args, 'static_source', False) and hasattr(model, 'use_static_source'):
        model.use_static_source = True
        print("Static-source mode enabled: model will use cached source STFT and ignore input source_audio.")

    model.eval()
    print("Model loaded and set to evaluation mode")
    
    # Testing loop
    print("Starting evaluation...")
    all_metrics = []
    total_samples = 0
    
    qual_dir = os.path.join(args.output_dir, 'qual') if args.visualize else None
    count_vis = 0

    # DPAM
    try:
        scope_dir = os.path.join('work_dirs', 'audio_3dgs_replaynvas_viewpoint', f"{args.selected_scenes[0] if len(args.selected_scenes)==1 else 'multi'}", f"viewpoint_{args.test_viewpoint}")
    except Exception:
        scope_dir = None
    dpam_model, dpam_sf, dpam_base = load_dpam(args.no_dpam, scope_dir)

    # 可视化辅助函数
    def _to_np_stereo(x):
        import numpy as np
        import torch as _torch
        if _torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 1:
            return np.stack([x, x], axis=0)
        if x.ndim == 2 and x.shape[0] == 2:
            return x
        if x.ndim == 2 and x.shape[1] == 2:
            return x.T
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
            if x.shape[0] == 2:
                return x
            if x.shape[1] == 2:
                return x.T
        x = x.reshape(-1)
        return np.stack([x, x], axis=0)

    def _make_spec(wav, sr):
        """
        Compute log-magnitude STFT for a stereo waveform.

        Args:
            wav: [2, T] float32
            sr:  sample rate (Hz)
        Returns:
            spec_np: [2, F_used, T_frames] numpy array with log1p magnitude,
                     only low-frequency bins kept (up to ~sr/4).
        """
        import torch as _torch

        n_fft = 512
        hop_length = 160
        win_length = 400

        wav_t = _torch.from_numpy(wav)
        stft = _torch.stft(
            wav_t,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=_torch.hamming_window(win_length),
            pad_mode="constant",
            center=True,
            return_complex=True,
        ).abs()
        spec = _torch.log1p(stft)  # [2, F_full, T]
        # Keep low-frequency half only, consistent with earlier behavior.
        spec = spec[:, : spec.shape[1] // 2, :]
        return spec.numpy()

    def _save_wavs(dest_prefix, sr, wavs):
        # wavs: dict name -> [2, T] float32
        try:
            import soundfile as sf
            for k, v in wavs.items():
                try:
                    sf.write(dest_prefix + f'-{k}.wav', v.T, int(sr))
                except Exception:
                    pass
        except Exception:
            # Fallback to scipy
            try:
                from scipy.io import wavfile
                for k, v in wavs.items():
                    try:
                        wavfile.write(dest_prefix + f'-{k}.wav', int(sr), v.T)
                    except Exception:
                        pass
            except Exception:
                pass

    def _plot_masks(dest_prefix, mono_mask, diff_mask, source_mag, atten_ft=None):
        """
        Save Mono/Diff mask spectrograms (and corresponding mono/diff magnitudes)
        for GS-only models (audio_3dgs_mono_diff_gs_only).

        Args:
            mono_mask: [B, F, T] tensor on any device
            diff_mask: [B, F, T] tensor on any device
            source_mag: [B, F, T_spec] tensor on any device (STFT magnitude)
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import torch as _torch

            if mono_mask is None or diff_mask is None:
                return

            # Move to CPU and drop batch dim
            def _to_np(x):
                if _torch.is_tensor(x):
                    x = x.detach().cpu().numpy()
                x = np.asarray(x)
                if x.ndim == 3:
                    return x[0]
                return x

            mono_m = _to_np(mono_mask)
            diff_m = _to_np(diff_mask)
            src_m = _to_np(source_mag)
            g_ft = _to_np(atten_ft) if atten_ft is not None else None

            # Ensure source magnitude has same spatial size for visualization
            if src_m.shape != mono_m.shape:
                # Basic center crop / pad to match; this is only for visualization.
                Fm, Tm = mono_m.shape[-2], mono_m.shape[-1]
                Fs, Ts = src_m.shape[-2], src_m.shape[-1]
                # Crop or pad frequency
                if Fs >= Fm:
                    src_m = src_m[Fs - Fm : Fs, :] if Fs > Fm else src_m
                else:
                    pad_top = (Fm - Fs) // 2
                    pad_bottom = Fm - Fs - pad_top
                    src_m = np.pad(src_m, ((pad_top, pad_bottom), (0, 0)))
                # Crop or pad time
                Fs2, Ts = src_m.shape[-2], src_m.shape[-1]
                if Ts >= Tm:
                    src_m = src_m[:, Ts - Tm : Ts] if Ts > Tm else src_m
                else:
                    pad_left = (Tm - Ts) // 2
                    pad_right = Tm - Ts - pad_left
                    src_m = np.pad(src_m, ((0, 0), (pad_left, pad_right)))

            mono_mag = mono_m * src_m
            diff_mag = diff_m * src_m

            # 1) Mono mask (non-negative, arbitrary scale)
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(mono_m, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title('Mono mask')
            ax.set_xlabel('Time')
            ax.set_ylabel('Freq')
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            try:
                plt.savefig(dest_prefix + '-mono-mask.png')
            finally:
                plt.close(fig)

            # 2) Diff mask (signed)
            fig, ax = plt.subplots(figsize=(6, 4))
            # Use percentile-based scaling to enhance contrast for small masks.
            if diff_m.size > 0:
                abs_dm = np.abs(diff_m)
                vmax = float(np.quantile(abs_dm, 0.99))
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = float(abs_dm.max()) if abs_dm.max() > 0 else 1.0
            else:
                vmax = 1.0
            im = ax.imshow(
                diff_m,
                aspect='auto',
                origin='lower',
                cmap='coolwarm',
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_title('Diff mask')
            ax.set_xlabel('Time')
            ax.set_ylabel('Freq')
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            try:
                plt.savefig(dest_prefix + '-diff-mask.png')
            finally:
                plt.close(fig)

            # 3) Mono magnitude (mask * |STFT(source)|)
            fig, ax = plt.subplots(figsize=(6, 4))
            m_log = np.log1p(np.clip(mono_mag, a_min=0.0, a_max=None))
            im = ax.imshow(m_log, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title('Mono magnitude (log1p)')
            ax.set_xlabel('Time')
            ax.set_ylabel('Freq')
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            try:
                plt.savefig(dest_prefix + '-mono-mag.png')
            finally:
                plt.close(fig)

            # 4) Diff magnitude (mask * |STFT(source)|, signed via mask)
            fig, ax = plt.subplots(figsize=(6, 4))
            # Keep sign from diff_mask, log-magnitude from |diff_mag|
            d_signed_log = np.sign(diff_mag) * np.log1p(np.abs(diff_mag))
            if d_signed_log.size > 0:
                abs_dl = np.abs(d_signed_log)
                vmax = float(np.quantile(abs_dl, 0.99))
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = float(abs_dl.max()) if abs_dl.max() > 0 else 1.0
            else:
                vmax = 1.0
            im = ax.imshow(
                d_signed_log,
                aspect='auto',
                origin='lower',
                cmap='coolwarm',
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_title('Diff magnitude (signed log1p)')
            ax.set_xlabel('Time')
            ax.set_ylabel('Freq')
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            try:
                plt.savefig(dest_prefix + '-diff-mag.png')
            finally:
                plt.close(fig)

            # 5) Distance attenuation g_ft (optional)
            if g_ft is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                # Visualize log10(g_ft) so that 0 ≈ reference distance.
                with np.errstate(divide="ignore", invalid="ignore"):
                    g_log = np.log10(np.clip(g_ft, a_min=1e-6, a_max=None))
                if g_log.size > 0:
                    abs_gl = np.abs(g_log)
                    vmax = float(np.quantile(abs_gl, 0.99))
                    if not np.isfinite(vmax) or vmax <= 0:
                        vmax = float(abs_gl.max()) if abs_gl.max() > 0 else 1.0
                else:
                    vmax = 1.0
                im = ax.imshow(
                    g_log,
                    aspect="auto",
                    origin="lower",
                    cmap="coolwarm",
                    vmin=-vmax,
                    vmax=vmax,
                )
                ax.set_title("Distance attenuation g_ft (log10)")
                ax.set_xlabel("Time")
                ax.set_ylabel("Freq")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                try:
                    plt.savefig(dest_prefix + "-atten-ft.png")
                finally:
                    plt.close(fig)
        except Exception:
            # Visualization is best-effort only; never break evaluation.
            pass

    def _plot_static_mono_spec(dest_prefix, sr, static_mag):
        """
        Plot static-source mono STFT magnitude (e.g., model.static_source_mag).

        Args:
            static_mag: [1, F, T] or [F, T] numpy/tensor
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import torch as _torch

            if static_mag is None:
                return
            if _torch.is_tensor(static_mag):
                static_mag = static_mag.detach().cpu().numpy()
            static_mag = np.asarray(static_mag)
            if static_mag.ndim == 3:
                static_mag = static_mag[0]
            if static_mag.ndim != 2:
                return

            # Log-magnitude for visualization
            mag = np.log1p(np.clip(static_mag, a_min=0.0, a_max=None))
            F_spec, T_spec = mag.shape

            # STFT params (match model/static STFT)
            n_fft = 512
            hop_length = 160
            # kHz per bin
            freq_res = sr / float(n_fft) / 1000.0
            time_res = hop_length / float(sr)

            extent = [0.0, T_spec * time_res, 0.0, (F_spec - 1) * freq_res]

            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(
                mag,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="viridis",
            )
            ax.set_title("Static mono STFT (log1p |source|)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (kHz)")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            try:
                plt.savefig(dest_prefix + "-static-mono-spec.png")
            finally:
                plt.close(fig)
        except Exception:
            # Best-effort; ignore plotting errors
            pass

    def _plot_input_mono(dest_prefix, sr, stereo_wav):
        """
        Plot per-sample input mono (L+R)/2 spectrogram + waveform.

        Args:
            stereo_wav: [2, T] numpy array (after _to_np_stereo/denorm)
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import torch as _torch

            if stereo_wav is None:
                return
            x = np.asarray(stereo_wav)
            if x.ndim != 2:
                return
            # Ensure stereo or mono shape [C, T]
            if x.shape[0] == 1:
                mono = x[0]
            else:
                mono = 0.5 * (x[0] + x[1])

            n_fft = 512
            hop_length = 160
            win_length = 400

            mono_t = _torch.from_numpy(mono)
            spec = _torch.stft(
                mono_t,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=_torch.hamming_window(win_length),
                pad_mode="constant",
                center=True,
                return_complex=True,
            ).abs()
            spec = _torch.log1p(spec)
            # Keep low-frequency half
            spec = spec[: spec.shape[0] // 2, :]
            spec_np = spec.numpy()

            F_spec, T_spec = spec_np.shape
            freq_res = sr / float(n_fft) / 1000.0  # kHz
            time_res = hop_length / float(sr)
            extent = [0.0, T_spec * time_res, 0.0, (F_spec - 1) * freq_res]

            fig, axes = plt.subplots(2, 1, figsize=(4.5, 6))
            ax_spec, ax_wav = axes

            im = ax_spec.imshow(
                spec_np,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="viridis",
            )
            ax_spec.set_title("Input mono (L+R)/2")
            ax_spec.set_ylabel("Frequency (kHz)")
            ax_spec.set_xlabel("")
            fig.colorbar(im, ax=ax_spec)

            t = np.arange(mono.shape[-1]) / float(sr)
            ax_wav.plot(t, mono)
            ax_wav.set_xlabel("Time (s)")
            ax_wav.set_ylabel("Amplitude")

            fig.tight_layout()
            try:
                plt.savefig(dest_prefix + "-input-mono.png")
            finally:
                plt.close(fig)
        except Exception:
            # Best-effort; ignore plotting errors
            pass

    def _plot_input_mono_wave_only(dest_prefix, sr, stereo_wav):
        """
        Plot input mono waveform only (no axes, transparent background),
        suitable for paper figures.

        Saves: {dest_prefix}-input-mono-wave.png
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            if stereo_wav is None:
                return
            x = np.asarray(stereo_wav)
            if x.ndim != 2:
                return
            if x.shape[0] == 1:
                mono = x[0]
            else:
                mono = 0.5 * (x[0] + x[1])

            t = np.arange(mono.shape[-1]) / float(sr)

            fig, ax = plt.subplots(figsize=(4.0, 1.6))
            # Transparent background
            fig.patch.set_alpha(0.0)
            ax.set_facecolor((1.0, 1.0, 1.0, 0.0))

            ax.plot(t, mono, linewidth=1.0, color="black")
            ax.set_xlim(t[0], t[-1])
            # Symmetric y-limits around zero for consistent amplitude visualization
            amp = float(np.max(np.abs(mono))) if mono.size > 0 else 1.0
            if amp <= 0:
                amp = 1.0
            ax.set_ylim(-1.05 * amp, 1.05 * amp)

            # Remove all axes, ticks, and spines for a clean waveform-only figure
            ax.axis("off")

            plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
            try:
                plt.savefig(
                    dest_prefix + "-input-mono-wave.png",
                    dpi=300,
                    transparent=True,
                    bbox_inches="tight",
                    pad_inches=0.0,
                )
            finally:
                plt.close(fig)
        except Exception:
            # Best-effort; ignore plotting errors
            pass

    def _plot_pred_lr_wave_only(dest_prefix, sr, stereo_wav):
        """
        Plot predicted L/R waveforms only (no axes, transparent background),
        suitable for paper figures.

        Saves:
          {dest_prefix}-pred-L-wave.png
          {dest_prefix}-pred-R-wave.png
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            if stereo_wav is None:
                return
            x = np.asarray(stereo_wav)
            if x.ndim != 2 or x.shape[0] < 2:
                return

            t = np.arange(x.shape[-1]) / float(sr)
            # Shared amplitude range for L/R for consistent comparison
            amp = float(np.max(np.abs(x))) if x.size > 0 else 1.0
            if amp <= 0:
                amp = 1.0
            y_min, y_max = -1.05 * amp, 1.05 * amp

            ch_names = ["L", "R"]
            for ch_idx, ch_name in enumerate(ch_names):
                fig, ax = plt.subplots(figsize=(4.0, 1.6))
                fig.patch.set_alpha(0.0)
                ax.set_facecolor((1.0, 1.0, 1.0, 0.0))

                ax.plot(t, x[ch_idx], linewidth=1.0, color="black")
                ax.set_xlim(t[0], t[-1])
                ax.set_ylim(y_min, y_max)
                ax.axis("off")

                plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
                try:
                    plt.savefig(
                        f"{dest_prefix}-pred-{ch_name}-wave.png",
                        dpi=300,
                        transparent=True,
                        bbox_inches="tight",
                        pad_inches=0.0,
                    )
                finally:
                    plt.close(fig)
        except Exception:
            # Best-effort; ignore plotting errors
            pass

    def _plot_lr_side_by_side(dest_prefix, sr, wavs, font_size=None, title_font_size=None):
        # wavs: dict with keys like 'input','pred','tgt' mapping to [2, T]
        import numpy as np
        import matplotlib.pyplot as plt
        names = ['input', 'pred', 'tgt']
        titles = ['Input', 'Prediction', 'Target']

        # Build specs
        specs = {k: _make_spec(v, sr) for k, v in wavs.items()}

        fs = None
        if font_size is not None:
            try:
                fs = float(font_size)
                if not np.isfinite(fs) or fs <= 0:
                    fs = None
            except Exception:
                fs = None

        title_fs = fs
        if title_font_size is not None:
            try:
                title_fs = float(title_font_size)
                if not np.isfinite(title_fs) or title_fs <= 0:
                    title_fs = fs
            except Exception:
                title_fs = fs

        label_kwargs = {"fontsize": fs} if fs is not None else {}
        title_kwargs = {"fontsize": title_fs} if title_fs is not None else {}

        # y-limits for waveforms
        y_lim = 0.0
        for a in wavs.values():
            y_lim = max(y_lim, float(np.max(np.abs(a))))
        y_lim = y_lim + 0.03

        # Per-name mono detection（仅当左右完全一致时视为单声道）
        def is_mono(w):
            return w.shape[0] == 1 or (w.shape[0] == 2 and np.allclose(w[0], w[1], atol=1e-6))

        # 计算总列数（Input/Pred/Tgt 各自若 stereo 则占两列，否则占一列）
        cols = 0
        for name in names:
            cols += 1 if is_mono(wavs[name]) else 2

        fig, axes = plt.subplots(2, cols, figsize=(2.3 * cols, 6))
        if cols == 1:
            axes = np.asarray(axes).reshape(2, 1)

        # STFT params (must match _make_spec)
        n_fft = 512
        hop_length = 160
        # Frequency resolution in kHz per bin
        freq_res = sr / float(n_fft) / 1000.0
        time_res = hop_length / float(sr)  # seconds per frame

        col = 0
        for name, title in zip(names, titles):
            w = wavs[name]
            S = specs[name]
            if is_mono(w):
                # 单声道：仅绘一列
                ax_spec = axes[0][col]
                ax_wav = axes[1][col]
                S0 = S[0] if S.shape[0] == 2 else S.squeeze(0)
                F_spec, T_spec = S0.shape
                # Map STFT frame/time bins to physical units
                extent = [
                    0.0,
                    T_spec * time_res,
                    0.0,
                    (F_spec - 1) * freq_res,
                ]  # [t_min, t_max, f_min, f_max]
                ax_spec.imshow(S0, aspect="auto", origin="lower", extent=extent)
                ax_spec.set_title(f"{title}", **title_kwargs)
                # 顶部一排不需要横坐标标题；纵坐标只在最左一列显示
                ax_spec.set_xlabel("")
                if col == 0:
                    ax_spec.set_ylabel("Frequency (kHz)", **label_kwargs)
                else:
                    ax_spec.set_ylabel("")
                if fs is not None:
                    ax_spec.tick_params(labelsize=fs)

                # Waveform
                t = (
                    np.arange(w[0].shape[-1]) / float(sr)
                    if w.shape[0] == 2
                    else np.arange(w.shape[-1]) / float(sr)
                )
                wav0 = w[0] if w.shape[0] == 2 else w.squeeze(0)
                ax_wav.plot(t, wav0)
                ax_wav.set_ylim(-y_lim, y_lim)
                ax_wav.set_xlabel("Time (s)", **label_kwargs)
                if col == 0:
                    ax_wav.set_ylabel("Amplitude", **label_kwargs)
                else:
                    ax_wav.set_ylabel("")
                if fs is not None:
                    ax_wav.tick_params(labelsize=fs)
                col += 1
            else:
                # 立体声：左/右各占一列
                for ch, ch_name in zip([0, 1], ['L', 'R']):
                    ax_spec = axes[0][col]
                    ax_wav = axes[1][col]
                    F_spec, T_spec = S[ch].shape
                    extent = [
                        0.0,
                        T_spec * time_res,
                        0.0,
                        (F_spec - 1) * freq_res,
                    ]
                    ax_spec.imshow(S[ch], aspect="auto", origin="lower", extent=extent)
                    ax_spec.set_title(f"{title} ({ch_name})", **title_kwargs)
                    ax_spec.set_xlabel("")
                    if col == 0:
                        ax_spec.set_ylabel("Frequency (kHz)", **label_kwargs)
                    else:
                        ax_spec.set_ylabel("")
                    if fs is not None:
                        ax_spec.tick_params(labelsize=fs)

                    t = np.arange(w[ch].shape[-1]) / float(sr)
                    ax_wav.plot(t, w[ch])
                    ax_wav.set_ylim(-y_lim, y_lim)
                    ax_wav.set_xlabel("Time (s)", **label_kwargs)
                    if col == 0:
                        ax_wav.set_ylabel("Amplitude", **label_kwargs)
                    else:
                        ax_wav.set_ylabel("")
                    if fs is not None:
                        ax_wav.tick_params(labelsize=fs)
                    col += 1
        fig.tight_layout()
        try:
            plt.savefig(dest_prefix + f'-plot-lr.png')
        finally:
            plt.close(fig)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move data to device
            cam_pose = batch['cam_pose'].to(device)
            source_audio = batch['source_audio'].to(device)
            target_binaural = batch['target_binaural'].to(device)
            # Optional: input/reference camera pose for viewpoint-source audio.
            # Some models (e.g., geometry-guided phase or shared-GS transfer) require
            # this to compute relative transforms w.r.t. the actual input viewpoint.
            input_cam_pose = batch.get('input_cam_pose', None)
            if input_cam_pose is not None and torch.is_tensor(input_cam_pose):
                input_cam_pose = input_cam_pose.to(device)
            input_viewpoint = batch.get('input_viewpoint', None)
            if input_viewpoint is not None and torch.is_tensor(input_viewpoint):
                input_viewpoint = input_viewpoint.to(device)
            # Optional raw target and environment residual
            target_binaural_raw = batch.get('target_binaural_raw', None)
            if target_binaural_raw is not None:
                target_binaural_raw = target_binaural_raw.to(device)
            env_residual_batch = batch.get('env_residual', None)
            if env_residual_batch is not None:
                env_residual_batch = env_residual_batch.to(device)
            scene_ids = batch['scene_id']
            norm_factors = batch.get('norm_factor', None)
            if norm_factors is not None and torch.is_tensor(norm_factors):
                norm_factors = norm_factors.to(device)
            
            # Debug: Check audio dimensions (only for first batch)
            # if batch_idx == 0:
                # print(f"Sample audio dimensions:")
                # print(f"  Source audio shape: {source_audio.shape}")
                # print(f"  Target audio shape: {target_binaural.shape}")
                # print(f"  Camera pose shape: {cam_pose.shape}")
            
            # Check if audio is too short for STFT
            min_samples_for_stft = 512  # n_fft
            if source_audio.shape[-1] < min_samples_for_stft:
                if batch_idx == 0:
                    print(f"  WARNING: Audio too short ({source_audio.shape[-1]} < {min_samples_for_stft}), padding...")
                # Pad audio to minimum length
                pad_length = min_samples_for_stft - source_audio.shape[-1]
                # For static-source mode, padding is only needed for target;
                # the model may ignore source_audio entirely.
                if not getattr(args, 'static_source', False):
                    source_audio = torch.cat(
                        [source_audio, torch.zeros(*source_audio.shape[:-1], pad_length, device=device)],
                        dim=-1,
                    )
                target_binaural = torch.cat([target_binaural, torch.zeros(*target_binaural.shape[:-1], pad_length, device=device)], dim=-1)
                if batch_idx == 0:
                    print(f"  Padded audio shape: {source_audio.shape}")
            
            # Forward pass
            try:
                raw_model = model.module if hasattr(model, "module") else model
                ref_pose = None
                if (
                    getattr(raw_model, "supports_ref_cam_pose", False)
                    and input_cam_pose is not None
                    and input_viewpoint is not None
                    and torch.is_tensor(input_viewpoint)
                ):
                    # Only provide ref_cam_pose when the sample actually has a valid
                    # input viewpoint (viewpoint-source audio). For near.wav input
                    # the dataset uses input_viewpoint==0 and a zero pose; passing
                    # that would change model behavior (e.g., phase correction).
                    try:
                        use_ref = bool((input_viewpoint > 0).any().item())
                    except Exception:
                        use_ref = False
                    if use_ref:
                        # Robustness: if any element is invalid (<=0), fall back to cam_pose
                        # for that sample so the relative reference becomes identity.
                        try:
                            invalid = (input_viewpoint <= 0).view(-1, 1)
                            ref_pose = torch.where(invalid, cam_pose, input_cam_pose)
                        except Exception:
                            ref_pose = input_cam_pose
                if args.gl_refine:
                    if getattr(args, 'static_source', False):
                        out = model(cam_pose, None, return_mag=True, ref_cam_pose=ref_pose) if ref_pose is not None else model(cam_pose, None, return_mag=True)
                    else:
                        out = model(cam_pose, source_audio, return_mag=True, ref_cam_pose=ref_pose) if ref_pose is not None else model(cam_pose, source_audio, return_mag=True)
                    if isinstance(out, (tuple, list)) and len(out) == 3:
                        predicted_audio, Lmag_all, Rmag_all = out
                    else:
                        predicted_audio = out
                        Lmag_all, Rmag_all = None, None
                else:
                    if getattr(args, 'static_source', False):
                        predicted_audio = model(cam_pose, None, ref_cam_pose=ref_pose) if ref_pose is not None else model(cam_pose, None)
                    else:
                        predicted_audio = model(cam_pose, source_audio, ref_cam_pose=ref_pose) if ref_pose is not None else model(cam_pose, source_audio)  # Correct parameter order: cam_pose first
                if batch_idx == 0:
                    print(f"  ✓ Forward pass successful, output shape: {predicted_audio.shape}")
                
                # Compute metrics for each sample in batch
                batch_size = target_binaural.size(0)

                # Optional Griffin-Lim refinement
                if args.gl_refine:
                    try:
                        import torchaudio
                        glL = torchaudio.transforms.GriffinLim(
                            n_fft=512, n_iter=int(args.gl_iters), hop_length=160, win_length=400,
                            window_fn=torch.hamming_window, power=1.0, momentum=0.99, rand_init=False
                        ).to(device)
                        glR = torchaudio.transforms.GriffinLim(
                            n_fft=512, n_iter=int(args.gl_iters), hop_length=160, win_length=400,
                            window_fn=torch.hamming_window, power=1.0, momentum=0.99, rand_init=False
                        ).to(device)
                        refined = []
                        for i in range(batch_size):
                            if Lmag_all is not None and Rmag_all is not None:
                                Lmag = Lmag_all[i]
                                Rmag = Rmag_all[i]
                                l_wav = glL(Lmag)
                                r_wav = glR(Rmag)
                                tgt_len = target_binaural[i].shape[-1]
                                min_len = min(l_wav.shape[-1], r_wav.shape[-1], tgt_len)
                                l_wav = l_wav[..., :min_len]
                                r_wav = r_wav[..., :min_len]
                                refined.append(torch.stack([l_wav, r_wav], dim=0))
                            else:
                                refined.append(predicted_audio[i])
                        predicted_audio = torch.stack(refined, dim=0)
                    except Exception as _e:
                        if batch_idx == 0:
                            print(f"Warning: Griffin-Lim refinement failed; using raw outputs. Error: {_e}")
                for i in range(batch_size):
                    pred = predicted_audio[i]
                    # Choose target and optionally add back environment residual
                    if getattr(args, 'add_env_back', False) and (target_binaural_raw is not None):
                        tgt = target_binaural_raw[i]
                        if env_residual_batch is not None:
                            er = env_residual_batch[i]
                            min_len = min(pred.shape[-1], tgt.shape[-1], er.shape[-1])
                            pred = pred[..., :min_len] + er[..., :min_len]
                            tgt = tgt[..., :min_len]
                        else:
                            min_len = min(pred.shape[-1], tgt.shape[-1])
                            pred = pred[..., :min_len]
                            tgt = tgt[..., :min_len]
                    else:
                        tgt = target_binaural[i]
                        # Ensure same length for metrics computation
                        min_len = min(pred.shape[-1], tgt.shape[-1])
                        pred = pred[..., :min_len]
                        tgt = tgt[..., :min_len]
                    # Optional denormalization back to original scene scale
                    if args.eval_denorm and norm_factors is not None:
                        factor = norm_factors[i] if torch.is_tensor(norm_factors) else norm_factors
                        if torch.is_tensor(factor):
                            factor = factor.item()
                        pred = pred * factor
                        tgt = tgt * factor
                    
                    metrics = compute_nvas_metrics(pred, tgt, cfg.dataset.sr, metric_mode=args.metric_mode)
                    # Optional: when GL magnitudes available, also report LR ratio in magnitude domain
                    if args.gl_refine and ('Lmag_all' in locals()) and ('Rmag_all' in locals()) and (Lmag_all is not None) and (Rmag_all is not None):
                        try:
                            Lmag = Lmag_all[i].detach().cpu().numpy()
                            Rmag = Rmag_all[i].detach().cpu().numpy()
                            eps = 1e-8
                            el = float(np.sum(Lmag.astype(np.float64) ** 2))
                            er = float(np.sum(Rmag.astype(np.float64) ** 2))
                            metrics['LR_ratio_pred_mag_db'] = float(10.0 * np.log10((el + eps) / (er + eps)))
                        except Exception:
                            pass
                    # Optional DPAM
                    if dpam_model is not None and dpam_sf is not None:
                        try:
                            dpam_val = compute_dpam(dpam_model, dpam_sf, dpam_base, pred, tgt, cfg.dataset.sr)
                            if dpam_val is not None:
                                metrics['DPAM'] = dpam_val
                        except Exception:
                            pass
                    metrics['scene_id'] = scene_ids[i]
                    # Optional printing of LR ratios per sample
                    if getattr(args, 'print_lr_ratio', False):
                        sid = metrics.get('scene_id', f'sample_{batch_idx}_{i}')
                        plr = metrics.get('LR_ratio_pred_db', None)
                        tlr = metrics.get('LR_ratio_gt_db', None)
                        if plr is not None and tlr is not None:
                            extra = ''
                            if 'LR_ratio_pred_mag_db' in metrics:
                                extra = f" | pred_mag={metrics['LR_ratio_pred_mag_db']:.3f} dB"
                            print(f"[LR-Ratio] {sid}: pred={plr:.3f} dB | gt={tlr:.3f} dB | LRE={metrics.get('LRE', float('nan')):.3f} dB{extra}")
                    all_metrics.append(metrics)
                    
                    # Save audio if requested
                    if args.save_audio:
                        pred_np = pred.cpu().numpy()
                        tgt_np = tgt.cpu().numpy()
                        src_np = source_audio[i][..., :pred.shape[-1]].detach().cpu().numpy()
                        if args.eval_denorm and norm_factors is not None:
                            factor = norm_factors[i] if torch.is_tensor(norm_factors) else norm_factors
                            if torch.is_tensor(factor):
                                factor = factor.item()
                            src_np = src_np * float(factor)
                        sample_name = _safe_sample_name(scene_ids[i], total_samples + i)
                        _save_audio_bundle(
                            os.path.join(args.output_dir, 'audio_samples'),
                            sample_name,
                            cfg.dataset.sr,
                            {
                                'input': src_np,
                                'predicted': pred_np,
                                'target': tgt_np,
                            },
                        )
                
                total_samples += batch_size

                # Optional qualitative visualization for a subset of samples
                if args.visualize and qual_dir is not None and count_vis < 64:
                    os.makedirs(qual_dir, exist_ok=True)
                    this_sr = cfg.dataset.sr
                    for i in range(batch_size):
                        if count_vis >= 64:
                            break
                        try:
                            if getattr(args, 'add_env_back', False) and ('target_binaural_raw' in batch):
                                pred_np = _to_np_stereo(predicted_audio[i].detach().cpu().numpy())
                                tgt_np = _to_np_stereo(batch['target_binaural_raw'][i].detach().cpu().numpy())
                                if 'env_residual' in batch:
                                    er_np = _to_np_stereo(batch['env_residual'][i].detach().cpu().numpy())
                                    mlen_e = min(pred_np.shape[-1], er_np.shape[-1])
                                    pred_np = pred_np[:, :mlen_e] + er_np[:, :mlen_e]
                            else:
                                pred_np = _to_np_stereo(predicted_audio[i].detach().cpu().numpy())
                                tgt_np = _to_np_stereo(target_binaural[i].detach().cpu().numpy())
                            inp_np = _to_np_stereo(source_audio[i].detach().cpu().numpy())
                            # Ensure equal length
                            mlen = min(pred_np.shape[-1], tgt_np.shape[-1], inp_np.shape[-1])
                            pred_np = pred_np[:, :mlen]
                            tgt_np = tgt_np[:, :mlen]
                            inp_np = inp_np[:, :mlen]
                            # Denorm for plots to match metrics if applicable
                            if args.eval_denorm and norm_factors is not None:
                                factor = norm_factors[i] if torch.is_tensor(norm_factors) else norm_factors
                                if torch.is_tensor(factor):
                                    factor = factor.item()
                                pred_np = pred_np * float(factor)
                                tgt_np = tgt_np * float(factor)
                                inp_np = inp_np * float(factor)
                            # Build name and save
                            scene_id = batch['scene_id'][i] if 'scene_id' in batch else f'sample_{total_samples + i}'
                            scene_id = str(scene_id)
                            safe_sid = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in scene_id)
                            dest_prefix = os.path.join(qual_dir, f"{count_vis:05d}-{safe_sid}")
                            # Stereo input/outputs for main comparison figure
                            wavs = {'input': inp_np, 'pred': pred_np, 'tgt': tgt_np}
                            _save_wavs(dest_prefix, this_sr, wavs)
                            # Write meta JSON with input/target poses and viewpoints
                            meta = {
                                'scene_id': scene_id,
                                'test_viewpoint': int(args.test_viewpoint),
                                'input_source': str(getattr(cfg.dataset, 'input_source', 'near')),
                            }
                            try:
                                if 'input_cam_pose' in batch:
                                    inp_pose = batch['input_cam_pose'][i].detach().cpu().numpy().tolist()
                                    meta['input_pose'] = inp_pose
                                if 'cam_pose' in batch:
                                    tgt_pose = batch['cam_pose'][i].detach().cpu().numpy().tolist()
                                    meta['target_pose'] = tgt_pose
                                if 'input_viewpoint' in batch:
                                    meta['input_viewpoint'] = int(batch['input_viewpoint'][i].item() if torch.is_tensor(batch['input_viewpoint'][i]) else batch['input_viewpoint'][i])
                                meta['target_viewpoint'] = int(args.test_viewpoint)
                            except Exception:
                                pass
                            try:
                                with open(dest_prefix + '-meta.json', 'w') as f:
                                    json.dump(meta, f, indent=2)
                            except Exception:
                                pass
                            _plot_lr_side_by_side(
                                dest_prefix,
                                this_sr,
                                wavs,
                                font_size=getattr(args, "vis_font_size", None),
                                title_font_size=getattr(args, "vis_title_font_size", None),
                            )
                            # Additional mono view of input (L+R)/2
                            try:
                                _plot_input_mono(dest_prefix, this_sr, inp_np)
                                _plot_input_mono_wave_only(dest_prefix, this_sr, inp_np)
                                _plot_pred_lr_wave_only(dest_prefix, this_sr, pred_np)
                            except Exception:
                                pass
                            # Optional: static mono STFT from model (if available)
                            try:
                                static_mag = None
                                if hasattr(model, "static_source_mag"):
                                    static_mag = model.static_source_mag
                                if static_mag is not None:
                                    _plot_static_mono_spec(dest_prefix, this_sr, static_mag)
                            except Exception:
                                pass
                            # Optional: also visualize Mono/Diff masks for GS-only models
                            if getattr(args, 'save_masks', False):
                                try:
                                    # Only Audio3DGSMonoDiffGSOnly currently supports return_masks.
                                    if getattr(args, 'static_source', False):
                                        out_dbg = model(
                                            cam_pose[i : i + 1],
                                            None,
                                            return_masks=True,
                                            ref_cam_pose=ref_pose[i : i + 1] if ref_pose is not None else None,
                                        )
                                    else:
                                        out_dbg = model(
                                            cam_pose[i : i + 1],
                                            source_audio[i : i + 1],
                                            return_masks=True,
                                            ref_cam_pose=ref_pose[i : i + 1] if ref_pose is not None else None,
                                        )
                                    if isinstance(out_dbg, (tuple, list)) and len(out_dbg) >= 5:
                                        _, mono_mask_dbg, diff_mask_dbg, src_mag_dbg, atten_dbg = out_dbg[:5]
                                        _plot_masks(dest_prefix, mono_mask_dbg, diff_mask_dbg, src_mag_dbg, atten_dbg)
                                except TypeError:
                                    # Model does not support return_masks; skip.
                                    pass
                                except Exception as _e:
                                    print(f"Warning: could not save mono/diff masks for sample {scene_id}: {_e}")
                            count_vis += 1
                        except Exception:
                            # Skip visualization errors per-sample
                            pass
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(test_loader)} batches ({total_samples} samples)")
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                # Only show full traceback for first error
                if batch_idx == 0:
                    import traceback
                    traceback.print_exc()
                continue
                
            # Break after all samples processed (remove debug limitation)
    
    # Calculate average metrics
    print("\n=== Evaluation Results ===")
    
    if len(all_metrics) == 0:
        print("No samples were successfully processed!")
        return False
    
    # Average across all samples
    avg_metrics = {}
    metric_names = ['MAG', 'ENV', 'LRE', 'RTE', 'DPAM']
    
    for metric in metric_names:
        values = [m[metric] for m in all_metrics if metric in m]
        if values:
            avg_metrics[metric] = np.mean(values)
            avg_metrics[f'{metric}_std'] = np.std(values)
    
    # Print results
    print(f"Total samples evaluated: {len(all_metrics)}")
    print(f"Test viewpoint: {args.test_viewpoint}")
    print(f"Scenes: {args.selected_scenes}")
    print()
    
    for metric in metric_names:
        if metric in avg_metrics:
            print(f"{metric}: {avg_metrics[metric]:.6f} ± {avg_metrics[f'{metric}_std']:.6f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'test_results.json')
    results = {
        'model_dir': args.model_dir,
        'test_viewpoint': args.test_viewpoint,
        'selected_scenes': args.selected_scenes,
        'total_samples': len(all_metrics),
        'average_metrics': avg_metrics,
        'per_sample_metrics': all_metrics
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    if args.save_audio:
        print(f"Audio samples saved to: {os.path.join(args.output_dir, 'audio_samples')}")
    
    print("Testing completed successfully!")
    return True


if __name__ == '__main__':
    args = parse_args()

    baseline_only = bool(getattr(args, 'baseline_only', False))

    audio3dgs_success = False
    if not baseline_only:
        print("🚀 Starting Audio 3DGS model testing...")
        audio3dgs_success = test_model(args)
    else:
        print("Skipping Audio 3DGS model testing (baseline-only mode).")

    mono_success = False
    # Run mono baseline either for comparison or as the only evaluation
    if baseline_only or audio3dgs_success:
        if baseline_only:
            print("\n🎯 Starting mono baseline testing (baseline-only mode)...")
        else:
            print("\n🎯 Starting mono baseline testing for comparison...")
        mono_success = test_mono_baseline(args)

    # If not in baseline-only mode and both evaluations succeeded, print comparison
    if (not baseline_only) and audio3dgs_success and mono_success:
        print("\n📊 === Performance Comparison ===")

        # Load and compare results
        audio3dgs_results_path = os.path.join(args.output_dir, 'test_results.json')
        baseline_output_dir = os.path.join(os.path.dirname(args.output_dir), f'mono_baseline_results_viewpoint_{args.test_viewpoint}')
        mono_results_path = os.path.join(baseline_output_dir, 'mono_baseline_results.json')

        try:
            with open(audio3dgs_results_path, 'r') as f:
                audio3dgs_results = json.load(f)
            with open(mono_results_path, 'r') as f:
                mono_results = json.load(f)

            print(f"Viewpoint {args.test_viewpoint} Results Comparison:")
            print("=" * 60)
            print(f"{'Metric':<10} {'Audio 3DGS':<25} {'Source Audio':<25} {'Improvement':<15}")
            print("=" * 60)

            for metric in ['MAG', 'ENV', 'LRE', 'RTE', 'DPAM']:
                if metric in audio3dgs_results['average_metrics'] and metric in mono_results['average_metrics']:
                    audio3dgs_val = audio3dgs_results['average_metrics'][metric]
                    mono_val = mono_results['average_metrics'][metric]

                    # For metrics, lower is better, so improvement = (baseline - model) / baseline * 100
                    if mono_val != 0:
                        improvement = (mono_val - audio3dgs_val) / mono_val * 100
                        improvement_str = f"{improvement:+.1f}%"
                    else:
                        improvement_str = "N/A"

                    # audio3dgs_str = f"{audio3dgs_val:.6f} ± {audio3dgs_results['average_metrics'].get(f'{metric}_std', 0):.6f}"
                    audio3dgs_str = f"{audio3dgs_val:.4f}"
                    # mono_str = f"{mono_val:.6f} ± {mono_results['average_metrics'].get(f'{metric}_std', 0):.6f}"
                    mono_str = f"{mono_val:.4f}"

                    print(f"{metric:<10} {audio3dgs_str:<25} {mono_str:<25} {improvement_str:<15}")

            print("=" * 60)
            print("Note: Positive improvement means Audio 3DGS is better than mono baseline")

        except Exception as e:
            print(f"Could not load results for comparison: {e}")

    # Determine overall success / exit code
    if baseline_only:
        overall_success = mono_success
    else:
        overall_success = audio3dgs_success and mono_success

    sys.exit(0 if overall_success else 1)
