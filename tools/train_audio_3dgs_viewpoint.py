"""
viewpoint分割的Audio 3DGS训练脚本
"""

from __future__ import division, print_function, with_statement

import argparse
import os
import random
from importlib import import_module as impm
from typing import List

import _init_paths
import os as _os
import numpy as np
import torch
import torch.distributed as dist

from configs import cfg, update_config
from libs.utils import misc
from libs.utils.lr_scheduler import ExponentialLR
from libs.utils.utils import create_logger, load_checkpoint


def set_torch_deterministic(seed: int = 42):
    """
    Enable PyTorch/CUDA deterministic behavior and set seeds.
    """
    try:
        _os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        _os.environ.setdefault('PYTHONHASHSEED', str(seed))
    except Exception:
        pass

    # Disable TF32 and cudnn autotune
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enforce deterministic algorithms if available
        try:
            # warn_only=True prevents hard errors from non-deterministic ops (e.g., reflection_pad1d)
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch: fall back to strict (may error on non-deterministic ops)
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass

    # Set PRNG seeds (redundant with later code but safe)
    try:
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description='Audio 3DGS Training - Viewpoint Split')
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        help='experiment configure file name',
        required=True,
        type=str)
    parser.add_argument(
        '--test-viewpoint',
        dest='test_viewpoint',
        default=7,
        type=int,
        help='which viewpoint to use for testing')
    parser.add_argument(
        '--selected-scenes',
        dest='selected_scenes',
        nargs='+',
        default=None,
        help='logical scene ids to use for training/testing')
    parser.add_argument(
        '--distributed',
        action='store_true',
        default=False,
        help='if use distribute train')
    parser.add_argument("--local_rank", type=int, default=0)
    # Audio SR & bandpass for training dataset
    parser.add_argument('--sr', type=int, default=None, help='Override dataset sampling rate (e.g., 16000)')
    parser.add_argument('--apply-bandpass', action='store_true', help='Apply bandpass filtering in dataset')
    parser.add_argument('--bandpass-low', type=float, default=150.0, help='Bandpass lowcut in Hz (default 150)')
    parser.add_argument('--bandpass-high', type=float, default=-1.0, help='Bandpass highcut in Hz (<=0 means Nyquist-1)')
    parser.add_argument('--avcloud-preproc', action='store_true', help='Shortcut: --sr 16000 + --apply-bandpass with low=150, high=nyquist-1')
    # Fair-comparison & viewpoint control
    parser.add_argument('--use-metadata', action='store_true', default=False,
                        help='Use metadata_v2.json to build frame list')
    parser.add_argument('--metadata-file', type=str, default='metadata_v2.json',
                        help='Metadata filename under data_root/v3 (default: metadata_v2.json)')
    parser.add_argument('--train-viewpoints', type=str, default=None,
                        help='Comma/space separated training viewpoints. Default: all except eval viewpoints')
    parser.add_argument('--val-viewpoints', type=str, default=None,
                        help='Comma/space separated validation viewpoints. Default: dataset config or fallback to test viewpoint')
    parser.add_argument('--input-source', type=str, default=None, choices=['near', 'viewpoint'],
                        help='Choose input audio source: near (default) or viewpoint')
    parser.add_argument('--input-viewpoint', type=int, default=0,
                        help='When --input-source viewpoint, use this viewpoint id as input')
    
    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    update_config(cfg, args)
    
    if args.selected_scenes is not None:
        dash_variants = ['\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '\u2212']
        def norm_scene(s):
            if not isinstance(s, str):
                return s
            for dv in dash_variants:
                s = s.replace(dv, '-')
            return s.strip()
        args.selected_scenes = [norm_scene(s) for s in args.selected_scenes]
        
    return args, cfg


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


def _parse_view_list_spec(spec) -> List[int]:
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        out = []
        for value in spec:
            try:
                out.append(int(value))
            except Exception:
                pass
        return sorted(set(out))
    text = str(spec).strip()
    if not text:
        return []
    out = []
    for tok in text.replace(',', ' ').split():
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception:
            pass
    return sorted(set(out))


def _resolve_viewpoint_loader(cfg):
    dataset_module = impm(str(cfg.dataset.name))
    make_loader = getattr(dataset_module, 'make_viewpoint_data_loader', None)
    if callable(make_loader):
        return make_loader
    raise AttributeError(
        f"Dataset module {cfg.dataset.name} does not provide make_viewpoint_data_loader(...)"
    )


def _infer_local_rank(args) -> int:
    try:
        if "LOCAL_RANK" in os.environ:
            return int(os.environ["LOCAL_RANK"])
    except Exception:
        pass
    try:
        return int(getattr(args, "local_rank", 0))
    except Exception:
        return 0


def _want_data_parallel() -> bool:
    v = str(os.environ.get("A3DGS_USE_DP", "")).strip().lower()
    if v in ("0", "false", "no", "n", "off"):
        return False
    if v in ("1", "true", "yes", "y", "on"):
        return True
    cvd = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    # Heuristic: if user explicitly exposes multiple GPUs, assume they want DP.
    if cvd and ("," in cvd or " " in cvd):
        return True
    return False


def main():
    args, cfg = parse_args()
    args.selected_scenes = _resolve_selected_scenes(args, cfg)
    make_viewpoint_data_loader = _resolve_viewpoint_loader(cfg)

    print(f"Using test viewpoint: {args.test_viewpoint}")
    print(f"Selected scenes: {args.selected_scenes}")

    # Deterministic settings
    try:
        set_torch_deterministic(int(getattr(cfg, 'seed', 42)))
        print('Deterministic settings enabled.')
    except Exception as _e:
        print(f'Warning: could not fully enable deterministic settings: {_e}')

    # Setup distributed training if needed
    rank = 0
    local_rank = 0
    if args.distributed:
        local_rank = _infer_local_rank(args)
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        rank = int(dist.get_rank())
        # Reduce stdout spam on non-master ranks
        try:
            misc.setup_for_distributed(rank == 0)
        except Exception:
            pass

    # Propagate viewpoint/scene scope to cfg
    try:
        cfg.defrost()
        cfg.dataset.test_viewpoint = int(args.test_viewpoint)
        cfg.dataset.selected_scenes = list(args.selected_scenes)
        cfg.dataset.scene_scope = args.selected_scenes[0] if len(args.selected_scenes) == 1 else 'multi'
        if getattr(args, 'use_metadata', False):
            cfg.dataset.use_metadata = True
            cfg.dataset.metadata_file = str(getattr(args, 'metadata_file', 'metadata_v2.json'))
        if getattr(args, 'train_viewpoints', None):
            cfg.dataset.train_viewpoints = str(args.train_viewpoints)
        if getattr(args, 'val_viewpoints', None):
            cfg.dataset.val_viewpoints = _parse_view_list_spec(args.val_viewpoints)
        if getattr(args, 'input_source', None):
            cfg.dataset.input_source = str(args.input_source)
            cfg.dataset.input_viewpoint = int(getattr(args, 'input_viewpoint', 0) or 0)
        cfg.freeze()
    except Exception:
        pass

    # Create logger using the existing cfg
    logger, output_dir = create_logger(cfg, rank)
    
    # Create additional subdirectory for this viewpoint
    viewpoint_output_dir = os.path.join(output_dir, f'viewpoint_{args.test_viewpoint}')
    os.makedirs(viewpoint_output_dir, exist_ok=True)
    
    logger.info(f'Using config: {args.yaml_file}')
    logger.info(f'Output directory: {viewpoint_output_dir}')
    logger.info(f'Test viewpoint: {args.test_viewpoint}')
    logger.info(f'Selected scenes: {args.selected_scenes}')
    logger.info(f'Validation viewpoints: {str(getattr(cfg.dataset, "val_viewpoints", "") or "(default=test)")}')    
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    print(f'Set random seed to {cfg.seed}')

    # Apply SR/bandpass overrides for training dataset （与 avcloud/ViGAS 相同的150Hz高通滤波）
    try:
        cfg.defrost()
        if args.avcloud_preproc:
            cfg.dataset.sr = 16000
            if not hasattr(cfg.dataset, 'bandpass'):
                from yacs.config import CfgNode as CN
                cfg.dataset.bandpass = CN()
            cfg.dataset.bandpass.enable = True
            cfg.dataset.bandpass.low_hz = 150.0
            cfg.dataset.bandpass.high_hz = -1.0
        else:
            if args.sr is not None and int(args.sr) > 0:
                cfg.dataset.sr = int(args.sr)
            if not hasattr(cfg.dataset, 'bandpass'):
                from yacs.config import CfgNode as CN
                cfg.dataset.bandpass = CN()
                cfg.dataset.bandpass.enable = False
                cfg.dataset.bandpass.low_hz = 150.0
                cfg.dataset.bandpass.high_hz = -1.0
                cfg.dataset.bandpass.order = 5
            if args.apply_bandpass:
                cfg.dataset.bandpass.enable = True
                cfg.dataset.bandpass.low_hz = float(args.bandpass_low)
                cfg.dataset.bandpass.high_hz = float(args.bandpass_high)
        cfg.freeze()
    except Exception as _e:
        print(f"Warning: could not apply SR/bandpass overrides: {_e}")
    
    # Create datasets with viewpoint splitting
    print('Creating dataset...')
    train_loader = make_viewpoint_data_loader(
        cfg, split='train', 
        test_viewpoint=args.test_viewpoint,
        selected_scenes=args.selected_scenes,
        distributed=args.distributed
    )
    
    # Only validate on rank 0 in distributed mode (avoids duplicated work)
    if args.distributed and rank != 0:
        val_loader = None
    else:
        val_loader = make_viewpoint_data_loader(
            cfg, split='val',
            test_viewpoint=args.test_viewpoint,
            selected_scenes=args.selected_scenes,
            distributed=False
        )
    
    print(f'Train samples: {len(train_loader.dataset)}')
    if val_loader is not None:
        print(f'Test samples: {len(val_loader.dataset)}')
    
    # Create model
    print('Creating model...')
    model_module = impm(f'libs.models.{cfg.model.file}')
    gaussian_model = None
    scene = None
    if bool(getattr(getattr(cfg, 'model', object()), 'use_visual', False)):
        visual_cfg = getattr(getattr(getattr(cfg, 'model', object()), 'shared_gs', object()), 'visual', object())
        visual_source = str(getattr(visual_cfg, 'feature_source', 'gaussian') or 'gaussian').strip().lower()
        if visual_source not in ('gaussian', ''):
            logger.info(
                'Using visual feature source %s; skipping gaussian scene load at build time.',
                visual_source,
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
                logger.info(
                    'Loaded frozen visual anchors from %s with %d points',
                    visual_root,
                    int(gaussian_model.get_xyz.shape[0]),
                )
            except Exception as e:
                logger.warning(
                    'Failed to load visual anchors for model.use_visual=True; '
                    'falling back to audio-only shared-GS. Reason: %s',
                    e,
                )
                gaussian_model = None
                scene = None
    model = model_module.build_model(cfg, gaussian_model, scene)  # Use build_model instead of get_model

    # Move model to device + optional multi-GPU wrappers
    if torch.cuda.is_available():
        if args.distributed:
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(cfg.device)
        model = model.to(device)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank
            )
            logger.info(f"Using DistributedDataParallel (rank={rank}, local_rank={local_rank})")
        else:
            if _want_data_parallel() and torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        print('Model moved to GPU')
    else:
        device = torch.device("cpu")
        model = model.to(device)
        logger.warning("CUDA not available, using CPU")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Create optimizer with parameter groups (boost spatial params LR)
    base_lr = float(cfg.train.lr)
    wd = float(cfg.train.weight_decay)
    sh_mult = float(getattr(cfg.train, 'lr_sh_mult', 1.0))
    rot_mult = float(getattr(cfg.train, 'lr_rot_mult', 1.0))
    xyz_mult = float(getattr(cfg.train, 'lr_xyz_mult', 1.0))
    alpha_mult = float(getattr(cfg.train, 'lr_alpha_mult', 1.0))

    # Collect special parameter tensors for SH / rotation / xyz so they can have dedicated learning-rate multipliers.
    raw_model = model.module if hasattr(model, "module") else model
    sh_params = []
    if hasattr(raw_model, '_sh_coeffs') and isinstance(getattr(raw_model, '_sh_coeffs'), torch.Tensor):
        sh_params.append(raw_model._sh_coeffs)
    if hasattr(raw_model, '_sh_mono') and isinstance(getattr(raw_model, '_sh_mono'), torch.Tensor):
        sh_params.append(raw_model._sh_mono)
    if hasattr(raw_model, '_sh_diff') and isinstance(getattr(raw_model, '_sh_diff'), torch.Tensor):
        sh_params.append(raw_model._sh_diff)

    special_ids = set()
    for p in sh_params:
        special_ids.add(id(p))
    if hasattr(raw_model, '_rotation') and isinstance(getattr(raw_model, '_rotation'), torch.Tensor):
        special_ids.add(id(raw_model._rotation))
    if hasattr(raw_model, '_xyz') and isinstance(getattr(raw_model, '_xyz'), torch.Tensor):
        special_ids.add(id(raw_model._xyz))
    alpha_param = getattr(raw_model, '_alpha_param', None)
    if isinstance(alpha_param, torch.Tensor):
        special_ids.add(id(alpha_param))

    base_params = [p for p in raw_model.parameters() if p.requires_grad and id(p) not in special_ids]

    param_groups = []
    if base_params:
        param_groups.append({'params': base_params, 'lr': base_lr, 'weight_decay': wd, 'name': 'base'})
    # SH coefficients
    if sh_params:
        sh_trainable = [p for p in sh_params if p.requires_grad]
        if sh_trainable:
            param_groups.append({'params': sh_trainable, 'lr': base_lr * sh_mult, 'weight_decay': wd, 'name': 'sh'})
    # Rotation quaternions
    if hasattr(raw_model, '_rotation') and isinstance(getattr(raw_model, '_rotation'), torch.Tensor) and raw_model._rotation.requires_grad:
        param_groups.append({'params': [raw_model._rotation], 'lr': base_lr * rot_mult, 'weight_decay': wd, 'name': 'rot'})
    # Point positions
    if hasattr(raw_model, '_xyz') and isinstance(getattr(raw_model, '_xyz'), torch.Tensor) and raw_model._xyz.requires_grad:
        param_groups.append({'params': [raw_model._xyz], 'lr': base_lr * xyz_mult, 'weight_decay': wd, 'name': 'xyz'})
    # Pointwise attenuation exponent (use_pointwise_alpha)
    if isinstance(alpha_param, torch.Tensor) and alpha_param.requires_grad:
        param_groups.append({'params': [alpha_param], 'lr': base_lr * alpha_mult, 'weight_decay': wd, 'name': 'alpha'})

    optimizer = torch.optim.Adam(param_groups)
    print(
        f"Optimizer param groups: base_lr={base_lr}, sh={base_lr*sh_mult}, "
        f"rot={base_lr*rot_mult}, xyz={base_lr*xyz_mult}, alpha={base_lr*alpha_mult}"
    )
    
    # Create learning rate scheduler (StepLR)
    from torch.optim.lr_scheduler import StepLR
    lr_scheduler = StepLR(optimizer, step_size=cfg.train.lr_decay_step, gamma=cfg.train.lr_decay)
    
    # Create trainer
    trainer_module = impm(f'libs.trainers.{cfg.train.file}')
    trainer = trainer_module.Audio3DGSTrainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        logger=logger
    )
    
    # Start training
    print('Starting training...')
    print(
        f'Training with test viewpoint {args.test_viewpoint} '
        f'and val viewpoints {str(getattr(cfg.dataset, "val_viewpoints", "") or "(default=test)")}'
    )
    trainer.train()
    
    logger.info('Training completed')


if __name__ == '__main__':
    main()
