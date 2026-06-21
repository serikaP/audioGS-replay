"""
Audio 3DGS Trainer
Specialized trainer for Audio 3D Gaussian Splatting without visual dependency
"""

import os
import json
import pickle
import time
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import csv

# Use a non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from libs.utils.misc import SmoothedValue
from libs.utils import misc as dist_misc
from libs.criterions.Criterion_2 import Criterion as CriterionV2
from libs.criterions.MonoDiffMSECriterion import MonoDiffMSECriterion
from libs.criterions.MonoOnlyMSECriterion import MonoOnlyMSECriterion
from libs.evaluators.gen_eval import Evaluator


class _NoOpSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return

    def close(self):
        return


class Audio3DGSTrainer:
    """Trainer for Audio 3D Gaussian Splatting"""
    
    def __init__(self, cfg, model, train_loader, val_loader, optimizer, lr_scheduler, logger):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger = logger

        self.rank = dist_misc.get_rank()
        self.world_size = dist_misc.get_world_size()
        self.is_main_process = dist_misc.is_main_process()
        
        # Verify model is using quaternion format
        raw_model = self._unwrap_model()
        if hasattr(raw_model, '_rotation'):
            rotation_shape = raw_model._rotation.shape
            logger.info(f"Model rotation parameter shape: {rotation_shape}")
            if len(rotation_shape) == 2 and rotation_shape[1] == 4:
                logger.info("✓ Model is using quaternion format (correct)")
            elif len(rotation_shape) == 3:
                logger.warning("✗ Model is using rotation matrix format (old format)")
            else:
                logger.warning(f"✗ Unexpected rotation format: {rotation_shape}")
        
        # Build results directory path with scene/viewpoint scoping for ReplayNVAS
        if hasattr(cfg.dataset, 'name') and 'replaynvas' in cfg.dataset.name.lower():
            result_subdir = "replayNVAS"
            base_dir = os.path.join("3dgs_result", result_subdir)

            # Derive optional scene scope from dataset
            scene_scope = None
            try:
                if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'selected_scenes'):
                    sel = self.train_loader.dataset.selected_scenes
                    if isinstance(sel, (list, tuple)) and len(sel) == 1:
                        scene_scope = sel[0]
                    elif isinstance(sel, (list, tuple)) and len(sel) > 1:
                        scene_scope = 'multi'
            except Exception:
                scene_scope = None

            # Use cfg-provided scene_scope if available
            if hasattr(cfg.dataset, 'scene_scope'):
                scene_scope = getattr(cfg.dataset, 'scene_scope') or scene_scope

            if scene_scope:
                base_dir = os.path.join(base_dir, scene_scope)

            # Include viewpoint if provided
            # Only append viewpoint subdir when using viewpoint-based split (vp in [1..8])
            viewpoint = getattr(cfg.dataset, 'test_viewpoint', None)
            try:
                import numpy as _np
                is_valid_vp = isinstance(viewpoint, (int, _np.integer)) and int(viewpoint) > 0
            except Exception:
                is_valid_vp = isinstance(viewpoint, int) and viewpoint > 0
            if is_valid_vp:
                base_dir = os.path.join(base_dir, f"viewpoint_{int(viewpoint)}")

            # Optional: per-frame/clip scoping (for per-clip training)
            frame_scope = getattr(cfg.dataset, 'frame_scope', None)
            if not frame_scope:
                import os as _os
                frame_scope = _os.environ.get('A3DGS_FRAME_ID', '').strip()
            if frame_scope:
                try:
                    frame_scope = str(int(frame_scope))
                except Exception:
                    frame_scope = str(frame_scope)
                base_dir = os.path.join(base_dir, f"frame_{frame_scope}")

            self.output_dir = base_dir
        else:
            # Default behavior for other datasets
            video_name = str(getattr(cfg.dataset, 'video', 1))
            sr = cfg.dataset.sr
            result_subdir = f"audio_3dgs_{video_name}_{sr}"
            self.output_dir = os.path.join("3dgs_result", result_subdir)
            scene_scope = str(getattr(cfg.dataset, 'scene_scope', '') or '').strip()
            if scene_scope and scene_scope.lower() != 'multi':
                self.output_dir = os.path.join(self.output_dir, scene_scope)
            viewpoint = getattr(cfg.dataset, 'test_viewpoint', None)
            try:
                import numpy as _np
                is_valid_vp = isinstance(viewpoint, (int, _np.integer)) and int(viewpoint) > 0
            except Exception:
                is_valid_vp = isinstance(viewpoint, int) and viewpoint > 0
            if is_valid_vp:
                self.output_dir = os.path.join(self.output_dir, f"viewpoint_{int(viewpoint)}")
            frame_scope = getattr(cfg.dataset, 'frame_scope', None)
            if not frame_scope:
                import os as _os
                frame_scope = _os.environ.get('A3DGS_FRAME_ID', '').strip()
            if frame_scope:
                try:
                    frame_scope = str(int(frame_scope))
                except Exception:
                    frame_scope = str(frame_scope)
                self.output_dir = os.path.join(self.output_dir, f"frame_{frame_scope}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Model checkpoints will be saved to: {self.output_dir}")
        
        # Loss function
        # For the new mono/diff or mono-only models, use simple MSE-based criteria;
        # otherwise default to the existing Criterion_2.
        model_file = str(getattr(getattr(cfg, "model", object()), "file", "") or "")
        if model_file in (
            "audio_3dgs_mono_diff",
            "audio_3dgs_mono_diff_gs_only",
            "audio_3dgs_mono_diff_field",
            # Experimental: scene-shared Gaussians + TF weights (top-K)
            "audio_3dgs_shared_gaussians_gs_only",
        ):
            self.criterion = MonoDiffMSECriterion(cfg)
        elif model_file in ("audio_3dgs_mono_only", "audio_3dgs_mono_gs_only"):
            self.criterion = MonoOnlyMSECriterion(cfg)
        else:
            self.criterion = CriterionV2(cfg)
        self.enhanced_criterion = None
        
        # Initialize enhanced criterion only when explicitly enabled via config
        if float(getattr(cfg.train, 'enhanced_weight', 0)) > 0:
            try:
                from libs.criterions.EnhancedCriterion_1 import EnhancedCriterion
                self.enhanced_criterion = EnhancedCriterion(cfg)
                logger.info("Using enhanced spatial audio loss function")
            except ImportError:
                logger.warning("EnhancedCriterion not found, using original criterion")
        else:
            logger.info("Enhanced spatial audio loss disabled (enhanced_weight <= 0)")
        
        # Evaluator for metrics
        self.evaluator = Evaluator(cfg, 'audio_3dgs', sampling_rate=cfg.dataset.sr)
        
        # Training state
        self.epoch = 0
        self.iter = 0
        self.best_val_loss = float('inf')
        
        # Per-epoch loss history (for CSV + plotting)
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': []  # may be None on epochs without validation
        }

        # Optional: freeze SH parameters for an initial portion of training
        self.sh_params = [p for name, p in raw_model.named_parameters() if name.startswith('_sh')]
        self.xyz_params = []
        if hasattr(raw_model, '_xyz'):
            xyz_p = getattr(raw_model, '_xyz')
            if isinstance(xyz_p, torch.Tensor):
                self.xyz_params.append(xyz_p)
        freeze_epochs_cfg = int(getattr(cfg.train, 'freeze_sh_epochs', 0) or 0)
        freeze_frac = float(getattr(cfg.train, 'freeze_sh_first_frac', 0.0) or 0.0)
        self.freeze_xyz_after_sh_unfreeze = bool(
            getattr(cfg.train, 'freeze_xyz_after_sh_unfreeze', False)
        )
        self._xyz_frozen = False
        if freeze_epochs_cfg > 0:
            self.sh_freeze_epochs = max(0, freeze_epochs_cfg)
        elif freeze_frac > 0:
            self.sh_freeze_epochs = max(0, int(np.ceil(cfg.train.max_epoch * freeze_frac)))
        else:
            self.sh_freeze_epochs = 0
        self._sh_currently_frozen = False
        if self.sh_freeze_epochs > 0 and self.sh_params:
            logger.info(
                f"SH freeze schedule: freezing SH params for first {self.sh_freeze_epochs} epochs "
                f"(max_epoch={cfg.train.max_epoch}, frac={freeze_frac:.3f})"
            )
        elif self.sh_freeze_epochs > 0 and not self.sh_params:
            logger.info("SH freeze schedule requested but no SH parameters were found; ignoring.")
        
        # Tensorboard writer (still use work_dirs for logs)
        if self.is_main_process:
            log_dir = os.path.join(cfg.output_dir, 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = _NoOpSummaryWriter()
        
        # Initialize model with source audio if needed
        self._initialize_model()

        # Optional: visual point-cloud coupling loss for _xyz regularization
        self._vis_coupling_weight = float(getattr(cfg.train, 'vis_coupling_weight', 0.0) or 0.0)
        self._vis_coupling_type = str(getattr(cfg.train, 'vis_coupling_type', 'nn') or 'nn').strip().lower()
        self._vis_coupling_num_samples = int(getattr(cfg.train, 'vis_coupling_num_samples', 4096) or 0)
        self._vis_coupling_warmup_epochs = float(getattr(cfg.train, 'vis_coupling_warmup_epochs', 0.0) or 0.0)
        self._vis_points_pkl = str(getattr(cfg.train, 'vis_coupling_points_pkl', '') or '')
        self._vis_points_ply = str(getattr(cfg.train, 'vis_coupling_points_ply', '') or '')
        self._vis_points_model_frame_cpu = None  # torch.FloatTensor [M,3] in model/world coords (CPU)
        self._vis_points_cache = {}  # device -> torch tensor
        if self._vis_coupling_weight > 0.0:
            try:
                self._setup_visual_point_coupling()
            except Exception as e:
                self.logger.warning(f"Visual coupling setup failed; disabling it. Reason: {e}")
                self._vis_coupling_weight = 0.0

    # ------------------------------------------------------------------
    # Visual point-cloud coupling loss (optional)
    # ------------------------------------------------------------------
    def _resolve_under_data_root(self, p: str) -> str:
        if not p:
            return ""
        p = str(p)
        if os.path.isabs(p):
            return p
        data_root = str(getattr(getattr(self.cfg, 'dataset', object()), 'data_root', '') or '')
        if data_root:
            return os.path.join(data_root, p)
        return p

    def _infer_single_scene_scope(self):
        # Prefer cfg.dataset.scene_scope (set by tools/train_audio_3dgs_viewpoint.py)
        try:
            scene_scope = getattr(getattr(self.cfg, 'dataset', object()), 'scene_scope', None)
            if isinstance(scene_scope, str) and scene_scope and scene_scope.lower() != 'multi':
                return scene_scope.strip()
        except Exception:
            pass
        # Fallback: dataset.selected_scenes when it is a single-scene run
        try:
            ds = getattr(self.train_loader, 'dataset', None)
            sel = getattr(ds, 'selected_scenes', None)
            if isinstance(sel, (list, tuple)) and len(sel) == 1:
                return str(sel[0]).strip()
        except Exception:
            pass
        return None

    def _load_vis_points_gs_np(self) -> np.ndarray:
        """
        Load visual points in COLMAP/GS coordinate frame.

        Prefer the cached `*_align_points_gs.pkl` produced by AV-Cloud (fetchPly),
        which already clusters points3D.ply into a small set of anchors (e.g., 256).
        """
        pkl_path = self._resolve_under_data_root(self._vis_points_pkl)
        if pkl_path and os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                d = pickle.load(f)
            pts = d.get('points', None) if isinstance(d, dict) else None
            if pts is None:
                raise ValueError(f"Invalid points pkl (missing 'points'): {pkl_path}")
            pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
            return pts

        ply_path = self._resolve_under_data_root(self._vis_points_ply)
        if ply_path and os.path.exists(ply_path):
            # Fall back to AV-Cloud's fetchPly (will also cache a *_align_points_gs.pkl).
            try:
                from libs.datasets.utils import fetchPly
                pts, _c, _n = fetchPly(ply_path, align_grids=None, N_points=256)
                pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
                return pts
            except Exception:
                # Last resort: load raw ply vertices (may be large)
                from plyfile import PlyData
                plydata = PlyData.read(ply_path)
                v = plydata['vertex']
                pts = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
                return pts

        raise FileNotFoundError(
            f"Cannot find visual points file. Tried pkl={pkl_path!r}, ply={ply_path!r}"
        )

    def _load_fixed_rotation_centers(self, scene: str) -> dict:
        data_root = str(getattr(getattr(self.cfg, 'dataset', object()), 'data_root', '') or '')
        fixed_pose_file = str(getattr(getattr(self.cfg, 'dataset', object()), 'fixed_pose_file', 'camera_positions_fixed_rotation.json') or 'camera_positions_fixed_rotation.json')
        fixed_path = fixed_pose_file if os.path.isabs(fixed_pose_file) else os.path.join(data_root, fixed_pose_file)
        with open(fixed_path, 'r') as f:
            fixed = json.load(f)
        centers = fixed.get('centers', {}).get(scene, {})
        out = {}
        if isinstance(centers, dict):
            for k, v in centers.items():
                if not isinstance(k, str) or '-' not in k:
                    continue
                try:
                    vp = int(k.split('-')[-1])
                except Exception:
                    continue
                try:
                    out[vp] = np.asarray(v, dtype=np.float32).reshape(3)
                except Exception:
                    continue
        return out

    def _load_gs_centers(self, scene: str) -> dict:
        data_root = str(getattr(getattr(self.cfg, 'dataset', object()), 'data_root', '') or '')
        gs_file = str(getattr(getattr(self.cfg, 'dataset', object()), 'gs_cameras_file', 'cam_imags/gs_cameras.json') or 'cam_imags/gs_cameras.json')
        gs_path = gs_file if os.path.isabs(gs_file) else os.path.join(data_root, gs_file)
        with open(gs_path, 'r') as f:
            cams = json.load(f)
        out = {}
        if isinstance(cams, list):
            prefix = f"{scene}_"
            for cam in cams:
                name = cam.get('img_name', '')
                if not isinstance(name, str) or not name.startswith(prefix):
                    continue
                try:
                    vp = int(name.rsplit('_', 1)[1])
                except Exception:
                    continue
                pos = cam.get('position', None)
                if pos is None:
                    continue
                try:
                    out[vp] = np.asarray(pos, dtype=np.float32).reshape(3)
                except Exception:
                    continue
        return out

    @staticmethod
    def _fit_similarity_transform(A: np.ndarray, B: np.ndarray):
        """
        Umeyama similarity fit: B ~= s * (R @ A) + t
        Returns: (s, R, t)
        """
        A = np.asarray(A, dtype=np.float64).reshape(-1, 3)
        B = np.asarray(B, dtype=np.float64).reshape(-1, 3)
        if A.shape != B.shape or A.shape[0] < 3:
            raise ValueError("Need >=3 paired points for similarity transform.")
        n = A.shape[0]
        muA = A.mean(axis=0)
        muB = B.mean(axis=0)
        Ac = A - muA
        Bc = B - muB
        cov = (Bc.T @ Ac) / float(n)
        U, S, Vt = np.linalg.svd(cov)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        varA = (Ac ** 2).sum() / float(n)
        varA = max(varA, 1e-12)
        s = float(S.sum() / varA)
        t = muB - s * (R @ muA)
        if not np.isfinite(s):
            raise ValueError("Similarity fit produced non-finite scale.")
        return s, R.astype(np.float64), t.astype(np.float64)

    @staticmethod
    def _apply_gs_to_fixed(points_gs: np.ndarray, s_fixed_to_gs: float, R_fixed_to_gs: np.ndarray, t_fixed_to_gs: np.ndarray) -> np.ndarray:
        """
        Given similarity transform: P_gs ~= s * (R @ P_fixed) + t,
        map points from GS frame -> fixed frame:
            P_fixed = (R^T @ (P_gs - t)) / s
        """
        s = float(s_fixed_to_gs)
        if abs(s) < 1e-12:
            raise ValueError("Invalid similarity scale (too small).")
        R = np.asarray(R_fixed_to_gs, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t_fixed_to_gs, dtype=np.float64).reshape(3)
        P = np.asarray(points_gs, dtype=np.float64).reshape(-1, 3)
        Pf = ((P - t[None, :]) @ R.T) / s
        return Pf.astype(np.float32)

    def _setup_visual_point_coupling(self):
        pts_gs = self._load_vis_points_gs_np()  # [M,3] in GS/COLMAP frame
        pose_source = str(getattr(getattr(self.cfg, 'dataset', object()), 'pose_source', 'fixed_rotation') or 'fixed_rotation').strip().lower()

        if pose_source in ('gs', 'gs_cameras', 'colmap'):
            pts_model = pts_gs
            self.logger.info(f"[vis_coupling] Using GS/COLMAP points directly (pose_source={pose_source}).")
        elif pose_source in ('fixed_rotation', 'fixed'):
            scene = self._infer_single_scene_scope()
            if not scene:
                raise RuntimeError(
                    "vis_coupling requires a single scene to estimate fixed<->gs alignment, "
                    "but could not infer scene_scope."
                )
            fixed_centers = self._load_fixed_rotation_centers(scene)
            gs_centers = self._load_gs_centers(scene)
            common = sorted(set(fixed_centers.keys()).intersection(gs_centers.keys()))
            if len(common) < 3:
                raise RuntimeError(
                    f"Not enough common viewpoints to align fixed_rotation to gs_cameras for scene={scene} "
                    f"(common={common}). Consider setting dataset.pose_source=gs_cameras."
                )
            A = np.stack([fixed_centers[v] for v in common], axis=0)  # fixed
            B = np.stack([gs_centers[v] for v in common], axis=0)     # gs
            s, R, t = self._fit_similarity_transform(A, B)  # fixed -> gs
            pts_model = self._apply_gs_to_fixed(pts_gs, s, R, t)
            self.logger.info(
                f"[vis_coupling] Aligned GS points to fixed_rotation frame for scene={scene} "
                f"using {len(common)} shared viewpoints (scale={s:.6g})."
            )
        else:
            raise RuntimeError(f"Unsupported pose_source for vis_coupling: {pose_source}")

        pts_model = np.asarray(pts_model, dtype=np.float32).reshape(-1, 3)
        if pts_model.shape[0] < 1:
            raise ValueError("No visual points loaded for coupling.")

        self._vis_points_model_frame_cpu = torch.from_numpy(pts_model).float().cpu()
        self._vis_points_cache = {}
        self.logger.info(
            f"[vis_coupling] Enabled: type={self._vis_coupling_type}, weight={self._vis_coupling_weight}, "
            f"vis_points={pts_model.shape[0]}, xyz_samples={self._vis_coupling_num_samples}"
        )

    def _get_vis_points_model_frame(self, device: torch.device, raw_model=None) -> torch.Tensor:
        if self._vis_points_model_frame_cpu is None:
            raise RuntimeError("Visual points are not initialized (vis coupling disabled or setup failed).")
        key = str(device)
        cached = self._vis_points_cache.get(key, None)
        if cached is not None and cached.device == device:
            return cached
        pts = self._vis_points_model_frame_cpu.to(device=device, non_blocking=True)
        # Match model coordinate normalization if requested
        if raw_model is not None and bool(getattr(raw_model, 'normalize_world_coords', False)):
            try:
                max_norm = float(getattr(raw_model, 'max_norm', 1.0))
                if max_norm > 0:
                    pts = pts / max_norm
            except Exception:
                pass
        self._vis_points_cache[key] = pts
        return pts

    def _current_vis_coupling_weight(self) -> float:
        w = float(self._vis_coupling_weight)
        warm = float(self._vis_coupling_warmup_epochs)
        if w <= 0.0:
            return 0.0
        if warm > 0.0:
            factor = max(0.0, min(1.0, float(self.epoch) / warm))
            return w * factor
        return w

    def _compute_vis_coupling_loss(self, raw_model) -> torch.Tensor:
        """
        Returns an *unweighted* coupling loss scalar (torch.Tensor on model device).
        """
        if self._vis_coupling_weight <= 0.0:
            return None
        if raw_model is None or (not hasattr(raw_model, '_xyz')):
            return None
        if self._vis_points_model_frame_cpu is None:
            return None

        # Get sanitized xyz in the model's world frame
        try:
            xyz = raw_model._safe_xyz()
        except Exception:
            xyz = raw_model._xyz
        xyz = torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)

        N = int(xyz.shape[0])
        K = int(self._vis_coupling_num_samples)
        if K > 0 and K < N:
            idx = torch.randint(low=0, high=N, size=(K,), device=xyz.device)
            xyz_s = xyz.index_select(0, idx)
        else:
            xyz_s = xyz

        vis = self._get_vis_points_model_frame(xyz_s.device, raw_model=raw_model)

        # Compute one-way (xyz -> vis) nearest-neighbor distance
        diff = xyz_s.unsqueeze(1) - vis.unsqueeze(0)  # [K, M, 3]
        dist2 = (diff * diff).sum(dim=-1)             # [K, M]
        min_d2 = dist2.min(dim=1).values              # [K]
        d = torch.sqrt(torch.clamp(min_d2, min=0.0) + 1e-9)
        loss = d.mean()

        if str(self._vis_coupling_type) == 'chamfer':
            # Symmetric term: vis -> xyz (computed over the same xyz_s subset)
            min_d2_v = dist2.min(dim=0).values  # [M]
            d_v = torch.sqrt(torch.clamp(min_d2_v, min=0.0) + 1e-9)
            loss = 0.5 * (loss + d_v.mean())

        return loss
        
    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _save_history_csv(self):
        """Write loss history to CSV in the output directory."""
        try:
            csv_path = os.path.join(self.output_dir, 'loss_history.csv')
            # Ensure directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            # Write header only if creating a new file or empty
            write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
            # For idempotency, rewrite full file each time to reflect latest values
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss'])
                for e, tr, va in zip(self.history['epoch'], self.history['train_loss'], self.history['val_loss']):
                    writer.writerow([int(e), float(tr) if tr is not None else '', float(va) if va is not None else ''])
        except Exception as e:
            self.logger.warning(f"Failed to save loss history CSV: {e}")

    def _plot_loss_curve(self):
        """Generate and save a PNG loss curve with train/val losses per epoch."""
        try:
            if len(self.history['epoch']) == 0:
                return
            epochs = self.history['epoch']
            train_losses = self.history['train_loss']
            val_losses = self.history['val_loss']

            plt.figure(figsize=(7.5, 5.0), dpi=120)
            # Train loss
            plt.plot(epochs, train_losses, label='Train Loss', color='#1f77b4', marker='o', linewidth=2)
            # Validation loss (filter out None)
            if any(v is not None for v in val_losses):
                val_x = [e for e, v in zip(epochs, val_losses) if v is not None]
                val_y = [v for v in val_losses if v is not None]
                plt.plot(val_x, val_y, label='Val Loss', color='#ff7f0e', marker='s', linewidth=2)

            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training/Validation Loss per Epoch')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.legend()
            plt.tight_layout()

            out_png = os.path.join(self.output_dir, 'loss_curve.png')
            plt.savefig(out_png)
            plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to plot/save loss curve: {e}")

    def _initialize_model(self):
        """Initialize model parameters with source audio"""
        raw_model = self._unwrap_model()
        # Prefer batch-based initialization when available (for field models)
        if hasattr(raw_model, 'initialize_from_batch'):
            try:
                sample_batch = next(iter(self.train_loader))
                device = next(raw_model.parameters()).device
                raw_model.initialize_from_batch(sample_batch, device)
                self.logger.info("Initialized model from reference batch (initialize_from_batch).")
                return
            except Exception as e:
                self.logger.warning(f"Could not initialize model from batch: {e}")

        if hasattr(raw_model, 'initialize_from_source_audio'):
            try:
                # Get a sample from training data for initialization
                sample_batch = next(iter(self.train_loader))
                source_audio = sample_batch['source_audio'][0]  # Take first sample
                device = next(raw_model.parameters()).device
                raw_model.initialize_from_source_audio(source_audio, device)
                self.logger.info("Initialized model with source audio")
            except Exception as e:
                self.logger.warning(f"Could not initialize model with source audio: {e}")

    def _set_sh_requires_grad(self, enabled: bool):
        """Toggle requires_grad for SH parameter tensors (and clear stale grads when freezing)."""
        if not self.sh_params:
            return
        for p in self.sh_params:
            if p is None:
                continue
            if p.requires_grad != enabled:
                p.requires_grad_(enabled)
            if (not enabled) and p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        self._sh_currently_frozen = not enabled

    def _set_xyz_requires_grad(self, enabled: bool):
        """Toggle requires_grad for xyz parameter tensors (and clear stale grads when freezing)."""
        if not self.xyz_params:
            return
        for p in self.xyz_params:
            if p is None:
                continue
            if p.requires_grad != enabled:
                p.requires_grad_(enabled)
            if (not enabled) and p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        self._xyz_frozen = not enabled

    def _maybe_toggle_sh_training(self):
        """Freeze/unfreeze SH params based on epoch schedule."""
        if not self.sh_params or getattr(self, 'sh_freeze_epochs', 0) <= 0:
            return
        should_freeze = self.epoch < self.sh_freeze_epochs
        if should_freeze == self._sh_currently_frozen:
            return
        self._set_sh_requires_grad(not should_freeze)
        state = "frozen" if should_freeze else "unfrozen"
        self.logger.info(
            f"SH params are now {state} (epoch {self.epoch}, freeze_epochs={self.sh_freeze_epochs})"
        )
        # Optionally freeze xyz once SH is unfrozen
        if (not should_freeze) and self.freeze_xyz_after_sh_unfreeze and self.xyz_params and (not self._xyz_frozen):
            self._set_xyz_requires_grad(False)
            self.logger.info("XYZ params are now frozen (post SH unfreeze) per config.")
    
    def train_epoch(self):
        """Train for one epoch"""
        self._maybe_toggle_sh_training()
        self.model.train()
        
        # Metrics
        loss_meter = SmoothedValue()
        batch_time = SmoothedValue()
        data_time = SmoothedValue()
        
        end = time.time()
        
        for i, batch in enumerate(self.train_loader):
            # Measure data loading time
            data_time.update(time.time() - end)

            # Move to GPU
            cam_pose = batch['cam_pose'].cuda(non_blocking=True)
            source_audio = batch['source_audio'].cuda(non_blocking=True)
            input_cam_pose = batch.get('input_cam_pose', None)
            if input_cam_pose is not None and torch.is_tensor(input_cam_pose):
                input_cam_pose = input_cam_pose.cuda(non_blocking=True)
            src_pos = batch.get('src_pos', None)
            if src_pos is not None and torch.is_tensor(src_pos):
                src_pos = src_pos.cuda(non_blocking=True)
            target_binaural = batch['target_binaural'].cuda(non_blocking=True)
            
            # Forward pass
            raw_model = self._unwrap_model()
            if hasattr(raw_model, 'requires_src_pos') and getattr(raw_model, 'requires_src_pos'):
                pred_binaural = self.model(cam_pose, source_audio, src_pos)
            else:
                if hasattr(raw_model, 'supports_ref_cam_pose') and getattr(raw_model, 'supports_ref_cam_pose') and input_cam_pose is not None:
                    pred_binaural = self.model(cam_pose, source_audio, ref_cam_pose=input_cam_pose)
                else:
                    pred_binaural = self.model(cam_pose, source_audio)
            
            # Ensure same length for loss computation
            min_len = min(pred_binaural.shape[-1], target_binaural.shape[-1])
            pred_binaural = pred_binaural[..., :min_len]
            target_binaural = target_binaural[..., :min_len]
            
            # Base loss (STFT magnitude on mono/diff)
            base_loss_dict = self.criterion(pred_binaural, target_binaural, None)
            loss = base_loss_dict['total_loss']

            # Optional enhanced spatial loss (LRE/coherence/phase etc.)
            if self.enhanced_criterion is not None:
                enh_loss_dict = self.enhanced_criterion(pred_binaural, target_binaural, None)
                base_w = float(self.cfg.train.get('enhanced_weight', 0.5))
                # 分段线性预热：前 warmup 轮从 0 -> base_w，避免早期不稳定
                warmup_ep = float(self.cfg.train.get('enhanced_warmup_epochs', 5.0))
                if warmup_ep > 0:
                    factor = max(0.0, min(1.0, self.epoch / warmup_ep))
                else:
                    factor = 1.0
                enh_w = base_w * factor
                loss = loss + enh_w * enh_loss_dict['total_loss']

            # Optional visual point-cloud coupling loss on _xyz (regularization)
            vis_loss_raw = None
            vis_w = 0.0
            try:
                vis_loss_raw = self._compute_vis_coupling_loss(raw_model)
                if vis_loss_raw is not None:
                    vis_w = float(self._current_vis_coupling_weight())
                    if vis_w > 0.0:
                        loss = loss + vis_w * vis_loss_raw
            except Exception:
                vis_loss_raw = None
                vis_w = 0.0
            
            # Backward pass with NaN/Inf guard
            if not torch.isfinite(loss):
                self.logger.warning('Non-finite loss encountered; skipping this batch.')
                continue
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping
            if self.cfg.train.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            
            self.optimizer.step()

            # Sanitize critical model parameters after each optimizer step
            try:
                self._sanitize_model_params()
            except Exception:
                pass
            
            # Update metrics
            loss_meter.update(loss.item(), cam_pose.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Logging
            if i % self.cfg.train.print_freq == 0:
                log_msg = (
                    f'Epoch: [{self.epoch}][{i}/{len(self.train_loader)}] '
                    f'Time: {batch_time.global_avg:.3f}s Data: {data_time.global_avg:.3f}s '
                    f'Loss: {loss_meter.global_avg:.4f} LR: {self.optimizer.param_groups[0]["lr"]:.6f}'
                )
                self.logger.info(log_msg)
                # Optional: log gradient norms of key spatial / SH params
                try:
                    import math as _math
                    # SH gradients: support single field (_sh_coeffs) and mono/diff fields
                    sh_fields = []
                    if hasattr(self.model, '_sh_coeffs'):
                        sh_fields.append(getattr(self.model, '_sh_coeffs'))
                    if hasattr(self.model, '_sh_mono'):
                        sh_fields.append(getattr(self.model, '_sh_mono'))
                    if hasattr(self.model, '_sh_diff'):
                        sh_fields.append(getattr(self.model, '_sh_diff'))

                    sh_norms = []
                    sh_mono_gn = float('nan')
                    sh_diff_gn = float('nan')
                    for p in sh_fields:
                        if p is None or p.grad is None:
                            continue
                        gn = p.grad.norm().item()
                        sh_norms.append(gn)
                        if p is getattr(self.model, '_sh_mono', None):
                            sh_mono_gn = gn
                        if p is getattr(self.model, '_sh_diff', None):
                            sh_diff_gn = gn
                    if sh_norms:
                        sh_gn = sum(sh_norms) / max(len(sh_norms), 1)
                    else:
                        sh_gn = float('nan')

                    # Rotation / XYZ gradients
                    if hasattr(self.model, '_rotation') and getattr(self.model, '_rotation').grad is not None:
                        rot_gn = self.model._rotation.grad.norm().item()
                    else:
                        rot_gn = float('nan')
                    if hasattr(self.model, '_xyz') and getattr(self.model, '_xyz').grad is not None:
                        xyz_gn = self.model._xyz.grad.norm().item()
                    else:
                        xyz_gn = float('nan')

                    # Prefer detailed log when mono/diff fields exist
                    if hasattr(self.model, '_sh_mono') or hasattr(self.model, '_sh_diff'):
                        self.logger.info(
                            f'GradNorms: SHmono={sh_mono_gn:.3e} SHdiff={sh_diff_gn:.3e} Rot={rot_gn:.3e} XYZ={xyz_gn:.3e}'
                        )
                    else:
                        self.logger.info(
                            f'GradNorms: SH={sh_gn:.3e} Rot={rot_gn:.3e} XYZ={xyz_gn:.3e}'
                        )
                except Exception:
                    pass
                
                # Log SH coefficients statistics
                def _log_sh_stats(name: str, sh_tensor: torch.Tensor):
                    if sh_tensor is None:
                        return
                    dc_mean = sh_tensor[:, 0, 0].mean().item()
                    dc_std = sh_tensor[:, 0, 0].std().item()
                    ho = sh_tensor[:, 0, 1:] if sh_tensor.shape[-1] > 1 else None
                    ho_mean = ho.abs().mean().item() if ho is not None else float('nan')
                    self.logger.info(
                        f'{name} Stats: DC_mean={dc_mean:.4f} DC_std={dc_std:.4f} HighOrder_mean={ho_mean:.4f}'
                    )

                if hasattr(raw_model, '_sh_coeffs'):
                    _log_sh_stats('SH', raw_model._sh_coeffs)
                if hasattr(raw_model, '_sh_mono'):
                    _log_sh_stats('SH_mono', raw_model._sh_mono)
                if hasattr(raw_model, '_sh_diff'):
                    _log_sh_stats('SH_diff', raw_model._sh_diff)
                
                # Tensorboard logging
                self.writer.add_scalar('train/loss', loss_meter.global_avg, self.iter)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.iter)
                if vis_loss_raw is not None and vis_w > 0.0:
                    try:
                        self.writer.add_scalar('train/vis_coupling_loss', float(vis_loss_raw.item()), self.iter)
                        self.writer.add_scalar('train/vis_coupling_w', float(vis_w), self.iter)
                    except Exception:
                        pass
                
                # Log SH coefficients to tensorboard
                def _tb_sh(name: str, sh_tensor: torch.Tensor):
                    if sh_tensor is None:
                        return
                    self.writer.add_scalar(f'train/{name}_dc_mean', sh_tensor[:, 0, 0].mean().item(), self.iter)
                    self.writer.add_scalar(f'train/{name}_dc_std', sh_tensor[:, 0, 0].std().item(), self.iter)
                    num_coeffs = sh_tensor.shape[-1]
                    for order in range(num_coeffs):
                        coeffs = sh_tensor[:, 0, order]
                        self.writer.add_scalar(f'train/{name}_order_{order}_mean', coeffs.mean().item(), self.iter)
                        self.writer.add_scalar(f'train/{name}_order_{order}_std', coeffs.std().item(), self.iter)

                if hasattr(raw_model, '_sh_coeffs'):
                    _tb_sh('sh', raw_model._sh_coeffs)
                if hasattr(raw_model, '_sh_mono'):
                    _tb_sh('sh_mono', raw_model._sh_mono)
                if hasattr(raw_model, '_sh_diff'):
                    _tb_sh('sh_diff', raw_model._sh_diff)
                
            self.iter += 1
            
        return loss_meter.global_avg
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        loss_meter = SmoothedValue()
        metrics = {}
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                # Move to GPU
                cam_pose = batch['cam_pose'].cuda(non_blocking=True)
                source_audio = batch['source_audio'].cuda(non_blocking=True)
                src_pos = batch.get('src_pos', None)
                if src_pos is not None and torch.is_tensor(src_pos):
                    src_pos = src_pos.cuda(non_blocking=True)
                target_binaural = batch['target_binaural'].cuda(non_blocking=True)
                
                # Forward pass
                raw_model = self._unwrap_model()
                if hasattr(raw_model, 'requires_src_pos') and getattr(raw_model, 'requires_src_pos'):
                    pred_binaural = self.model(cam_pose, source_audio, src_pos, is_val=True)
                else:
                    input_cam_pose = batch.get('input_cam_pose', None)
                    if input_cam_pose is not None and torch.is_tensor(input_cam_pose):
                        input_cam_pose = input_cam_pose.cuda(non_blocking=True)
                    if hasattr(raw_model, 'supports_ref_cam_pose') and getattr(raw_model, 'supports_ref_cam_pose') and input_cam_pose is not None:
                        pred_binaural = self.model(cam_pose, source_audio, is_val=True, ref_cam_pose=input_cam_pose)
                    else:
                        pred_binaural = self.model(cam_pose, source_audio, is_val=True)
                
                # Ensure same length
                min_len = min(pred_binaural.shape[-1], target_binaural.shape[-1])
                pred_binaural = pred_binaural[..., :min_len]
                target_binaural = target_binaural[..., :min_len]
                
                # Compute validation loss consistent with training
                base_loss_dict = self.criterion(pred_binaural, target_binaural)
                loss = base_loss_dict['total_loss']
                if self.enhanced_criterion is not None:
                    enh_loss_dict = self.enhanced_criterion(pred_binaural, target_binaural)
                    enh_w = float(self.cfg.train.get('enhanced_weight', 0.5))
                    loss = loss + enh_w * enh_loss_dict['total_loss']
                loss_meter.update(loss.item(), cam_pose.size(0))
                # Debug: Check data ranges across all batches
                # Initialize tracking variables
                if i == 0:
                    self.pred_min = float('inf')
                    self.pred_max = float('-inf')
                    self.tgt_min = float('inf')
                    self.tgt_max = float('-inf')
                    self.pred_rms_sum = 0.0
                    self.tgt_rms_sum = 0.0
                    self.batch_count = 0
                
                # Update min/max tracking
                self.pred_min = min(self.pred_min, pred_binaural.min().item())
                self.pred_max = max(self.pred_max, pred_binaural.max().item())
                self.tgt_min = min(self.tgt_min, target_binaural.min().item())
                self.tgt_max = max(self.tgt_max, target_binaural.max().item())
                
                # Update RMS tracking
                self.pred_rms_sum += torch.sqrt(pred_binaural.pow(2).mean()).item()
                self.tgt_rms_sum += torch.sqrt(target_binaural.pow(2).mean()).item()
                self.batch_count += 1
                # Evaluate metrics
                if self.evaluator:
                    # Ensure correct format for evaluator: [2, length]
                    pred_for_eval = pred_binaural.squeeze().cpu().numpy()
                    target_for_eval = target_binaural.squeeze().cpu().numpy()
                    
                    # Ensure stereo format
                    if pred_for_eval.ndim == 1:
                        pred_for_eval = np.stack([pred_for_eval, pred_for_eval])
                    elif pred_for_eval.shape[0] != 2:
                        pred_for_eval = pred_for_eval.T
                        
                    if target_for_eval.ndim == 1:
                        target_for_eval = np.stack([target_for_eval, target_for_eval])
                    elif target_for_eval.shape[0] != 2:
                        target_for_eval = target_for_eval.T
                    
                    batch_metrics = self.evaluator.evaluate(
                        pred_for_eval,
                        target_for_eval,
                        self.cfg.dataset.sr
                    )
                    
                    # Check if evaluator returned valid metrics
                    if batch_metrics is not None:
                        for key, value in batch_metrics.items():
                            if key not in metrics:
                                metrics[key] = SmoothedValue()
                            metrics[key].update(value, cam_pose.size(0))
        
        # Log validation results
        log_str = f'Validation - Loss: {loss_meter.global_avg:.4f}'
        for key, meter in metrics.items():
            log_str += f' {key}: {meter.global_avg:.4f}'
        self.logger.info(log_str)
        
        # Print global statistics if tracked
        if hasattr(self, 'batch_count') and self.batch_count > 0:
            pred_avg_rms = self.pred_rms_sum / self.batch_count
            tgt_avg_rms = self.tgt_rms_sum / self.batch_count
            self.logger.info(f"Global Validation Statistics:")
            self.logger.info(f"  Pred wav range: [{self.pred_min:.6f}, {self.pred_max:.6f}]")
            self.logger.info(f"  Tgt wav range: [{self.tgt_min:.6f}, {self.tgt_max:.6f}]")
            self.logger.info(f"  Pred avg RMS: {pred_avg_rms:.6f}")
            self.logger.info(f"  Tgt avg RMS: {tgt_avg_rms:.6f}")
            self.logger.info(f"  Batches processed: {self.batch_count}")
            
            # Clean up tracking variables
            delattr(self, 'pred_min')
            delattr(self, 'pred_max')
            delattr(self, 'tgt_min')
            delattr(self, 'tgt_max')
            delattr(self, 'pred_rms_sum')
            delattr(self, 'tgt_rms_sum')
            delattr(self, 'batch_count')
        
        # Tensorboard logging
        self.writer.add_scalar('val/loss', loss_meter.global_avg, self.epoch)
        for key, meter in metrics.items():
            self.writer.add_scalar(f'val/{key}', meter.global_avg, self.epoch)
            
        return loss_meter.global_avg, {key: meter.global_avg for key, meter in metrics.items()}

    def _sanitize_model_params(self):
        """Clamp and denoise key learnable tensors to keep training numerically stable."""
        m = self._unwrap_model()
        with torch.no_grad():
            if hasattr(m, '_sh_coeffs') and m._sh_coeffs is not None:
                m._sh_coeffs.data = torch.nan_to_num(m._sh_coeffs.data, nan=0.0, posinf=0.0, neginf=0.0)
                m._sh_coeffs.data.clamp_(-5.0, 5.0)
            if hasattr(m, '_rotation') and m._rotation is not None:
                q = torch.nan_to_num(m._rotation.data, nan=0.0, posinf=0.0, neginf=0.0)
                norms = torch.norm(q, dim=-1, keepdim=True)
                bad = (norms < 1e-8) | (~torch.isfinite(norms))
                if bad.any():
                    ident = q.new_zeros((1, 4))
                    ident[0, 0] = 1.0
                    q[bad.squeeze(-1)] = ident
                m._rotation.data.copy_(torch.nn.functional.normalize(q, dim=-1, eps=1e-8))
            if hasattr(m, '_xyz') and m._xyz is not None:
                m._xyz.data = torch.nan_to_num(m._xyz.data, nan=0.0, posinf=0.0, neginf=0.0)
                bound = float(getattr(m, 'max_norm', 310.0)) * 50.0
                m._xyz.data.clamp_(-bound, bound)
            # Optional: keep Hopkins parameters in a reasonable numeric range
            if getattr(m, 'use_hopkins_atten', False) and hasattr(m, 'log_Rc_param') and m.log_Rc_param is not None:
                # Clamp log_Rc to avoid extreme Rc values (roughly [log 1, log 1e4])
                try:
                    m.log_Rc_param.data = torch.nan_to_num(m.log_Rc_param.data, nan=0.0, posinf=0.0, neginf=0.0)
                    m.log_Rc_param.data.clamp_(min=-2.0, max=10.0)
                except Exception:
                    pass
            if getattr(m, 'use_hopkins_atten', False) and hasattr(m, 'Q_param') and m.Q_param is not None:
                # Keep Q non-negative and within a broad but stable range.
                try:
                    m.Q_param.data = torch.nan_to_num(m.Q_param.data, nan=0.0, posinf=0.0, neginf=0.0)
                    m.Q_param.data.clamp_(min=0.0, max=50.0)
                except Exception:
                    pass
    
    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting training for {self.cfg.train.max_epoch} epochs")
        
        for epoch in range(self.cfg.train.max_epoch):
            self.epoch = epoch

            # Shuffle distributed sampler each epoch
            try:
                sampler = getattr(self.train_loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
            except Exception:
                pass
            
            # Train one epoch
            train_loss = self.train_epoch()
            val_loss = None
            
            # Validate
            if self.is_main_process and self.val_loader is not None and epoch % self.cfg.train.val_freq == 0:
                val_loss, val_metrics = self.validate()
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
                    self.logger.info(f"New best validation loss: {val_loss:.4f}")
            
            # Record history and persist artifacts each epoch
            if self.is_main_process:
                self.history['epoch'].append(epoch)
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self._save_history_csv()
                self._plot_loss_curve()
            
            # Update learning rate
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            # Save regular checkpoint
            # 仅保留一个“最新”checkpoint 文件，减少磁盘占用。
            # 满足原有 save_freq 条件，或者在最后一个 epoch 强制保存，
            # 都会覆盖写入同一个 checkpoint_latest.pth。
            if self.is_main_process and ((epoch + 1) == self.cfg.train.max_epoch or epoch % self.cfg.train.save_freq == 0):
                self.save_checkpoint()

        # Log final Hopkins parameters (if any) for interpretability
        try:
            m = self._unwrap_model()
            if getattr(m, 'use_hopkins_atten', False) and hasattr(m, 'Q_param') and hasattr(m, 'log_Rc_param'):
                with torch.no_grad():
                    Q_val = float(torch.clamp(m.Q_param, min=0.0).mean().item()) if m.Q_param is not None else float('nan')
                    Rc_val = float(torch.exp(m.log_Rc_param).mean().item()) if m.log_Rc_param is not None else float('nan')
                self.logger.info(
                    f"Final Hopkins params (averaged): Q ≈ {Q_val:.3f}, R_c ≈ {Rc_val:.3f}"
                )
        except Exception as _e:
            self.logger.warning(f"Could not log Hopkins parameters at training end: {_e}")

        self.logger.info("Training completed")
        self.writer.close()
        
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        if not self.is_main_process:
            return
        state = {
            'epoch': self.epoch,
            'iter': self.iter,
            # Save raw model weights (no "module." prefix) for compatibility with eval scripts
            'model_state_dict': self._unwrap_model().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'cfg': self.cfg
        }
        
        if self.lr_scheduler:
            state['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        # Save regular checkpoint（统一写入同一个文件，始终覆盖旧版本）
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(state, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(state, best_path)
            
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self._unwrap_model().load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'lr_scheduler_state_dict' in checkpoint and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
        self.epoch = checkpoint.get('epoch', 0)
        self.iter = checkpoint.get('iter', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")


def build_trainer(cfg, model, train_loader, val_loader, optimizer, lr_scheduler, logger):
    """Build trainer instance"""
    return Audio3DGSTrainer(cfg, model, train_loader, val_loader, optimizer, lr_scheduler, logger)
