"""
Mono/Diff Audio 3D Gaussian Splatting

- Based on libs/models/audio_3dgs.py
- Key changes:
  * Split SH coefficients into two independent sets: mono and diff.
  * Use mono SH field for the mono branch features, diff SH field for the diff branch.
  * Keep the overall interface compatible with Audio3DGS (cam_pose, source_audio) so that
    existing trainers and datasets can be reused.
"""

import json
import math
import os
import re
import sqlite3
import struct
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.models.audio_3dgs import DualBranchAudioUNet
from libs.utils.sh_utils import eval_sh
from libs.datasets.scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_extrinsics_text


class Audio3DGSMonoDiff(nn.Module):
    """
    Audio 3D Gaussian Splatting with separate SH fields for mono and diff.

    This variant keeps the original Audio3DGS structure (source spectrogram +
    dual-branch U-Net) but splits the directional SH field into:
      - mono SH: drives the mono branch features
      - diff SH: drives the diff branch features
    """

    def __init__(self, cfg, freq_num: int = 257, time_num: int = 348):
        super().__init__()

        self.freq_num = freq_num
        self.time_num = time_num
        # Each spectrogram pixel = one 3D point
        self.n_points = freq_num * time_num

        # SH degree and number of coefficients (default 2 -> 9)
        self.sh_degree = int(getattr(cfg.model, "sh_degree", 2))
        self.num_sh_coeffs = (self.sh_degree + 1) ** 2

        # Audio point parameters (learnable)
        # Position in 3D space for each spectrogram pixel.
        # Default: random init in a small ball around origin; can optionally
        # be re-anchored around the current scene's camera-center mean.
        try:
            self.xyz_init_std = float(getattr(getattr(cfg, "model", object()), "xyz_init_std", 10.0))
        except Exception:
            self.xyz_init_std = 10.0
        self._xyz = nn.Parameter(torch.randn(self.n_points, 3) * float(self.xyz_init_std))

        # Two sets of SH coefficients (0..sh_degree):
        #  - mono field: drives mono branch
        #  - diff field: drives diff branch
        # Shape: [n_points, 1, (sh_degree+1)^2]
        self._sh_mono = nn.Parameter(torch.zeros(self.n_points, 1, self.num_sh_coeffs))
        self._sh_diff = nn.Parameter(torch.zeros(self.n_points, 1, self.num_sh_coeffs))

        # Optional random init for higher-order SH to break symmetry
        sh_rand_std = float(getattr(cfg.model, "sh_rand_init_std", 0.0))
        if sh_rand_std > 0:
            with torch.no_grad():
                if self.num_sh_coeffs > 1:
                    noise = sh_rand_std * torch.randn(self.n_points, self.num_sh_coeffs - 1)
                    self._sh_mono[:, 0, 1:] = noise
                    # Independent noise for diff field to avoid collapse
                    self._sh_diff[:, 0, 1:] = sh_rand_std * torch.randn(
                        self.n_points, self.num_sh_coeffs - 1
                    )

        # Rotation quaternion for each point (w, x, y, z).
        # Initialize with identity quaternion [1, 0, 0, 0] plus small random perturbation.
        rotation_init = torch.zeros(self.n_points, 4)
        rotation_init[:, 0] = 1.0
        rotation_init[:, 1:] = 0.01 * torch.randn(self.n_points, 3)
        self._rotation = nn.Parameter(rotation_init)

        # Time-frequency coordinate mapping (fixed)
        self.register_buffer("tf_coords", self._create_tf_grid())

        # Whether to incorporate camera orientation when forming SH directions
        self.use_cam_rotation = bool(
            getattr(getattr(cfg, "model", object()), "use_cam_rotation", False)
        )
        # Whether to rotate viewing directions into each point's local frame before SH eval.
        # When False, SH is evaluated directly in world/camera frame (closer to vanilla 3DGS).
        self.use_point_rotation = bool(
            getattr(getattr(cfg, "model", object()), "use_point_rotation", True)
        )
        # Optional: flip camera Y (down->up) before SH to match z-up conventions
        self.flip_cam_y_for_sh = bool(
            getattr(getattr(cfg, "model", object()), "flip_cam_y_for_sh", False)
        )
        # Whether to inject stereo cues (ILD) into diff branch
        self.use_stereo_cues = bool(
            getattr(getattr(cfg, "model", object()), "use_stereo_cues", False)
        )
        # Optional: also feed inv_distance and/or side magnitude into diff branch
        self.diff_use_inv_distance = bool(
            getattr(getattr(cfg, "model", object()), "diff_use_inv_distance", False)
        )
        self.diff_use_side_mag = bool(
            getattr(getattr(cfg, "model", object()), "diff_use_side_mag", False)
        )
        # Activation for mono mask (when not in abs_mag_output mode)
        self.mono_mask_activation = str(
            getattr(getattr(cfg, "model", object()), "mono_mask_activation", "sigmoid")
            or "sigmoid"
        ).lower()

        # Dual-branch rendering network
        use_groupnorm = bool(getattr(cfg.model, "use_groupnorm", True))
        diff_in_channels = 1
        if self.use_stereo_cues:
            diff_in_channels += 1  # ILD
        if self.diff_use_inv_distance:
            diff_in_channels += 1
        if self.diff_use_side_mag:
            diff_in_channels += 1
        self.renderer = DualBranchAudioUNet(
            use_groupnorm=use_groupnorm, diff_in_channels=diff_in_channels
        )

        # Normalization factor for relative positions
        self.max_norm = float(getattr(cfg.model, "max_norm", 10.0))

        # Optional: store xyz and microphone positions in max_norm-normalized
        # coordinates for more stable optimization. When enabled, _xyz is
        # effectively parameterized in units of (world / max_norm).
        self.normalize_world_coords = bool(
            getattr(getattr(cfg, "model", object()), "normalize_world_coords", False)
        )

        # Optional: initialize _xyz from SfM point cloud (COLMAP points3D).
        # When enabled, this overrides the random init and skips xyz_anchor_to_scene.
        initialized_from_sfm = self._maybe_init_xyz_from_sfm(cfg)
        if not initialized_from_sfm:
            # Default behavior: optionally normalize random init
            if self.normalize_world_coords:
                with torch.no_grad():
                    self._xyz.div_(float(self.max_norm))

            # Optional: re-anchor _xyz around scene camera centers (ReplayNVAS-style).
            # Controlled by:
            #   cfg.model.xyz_anchor_to_scene: bool
            #   cfg.model.xyz_anchor_radius: float (world-units radius for random offsets)
            self._maybe_anchor_xyz_to_scene(cfg)

    def _maybe_init_xyz_from_sfm(self, cfg) -> bool:
        """
        Optionally initialize _xyz from the ReplayNVAS COLMAP/SfM sparse point cloud.

        Expected to be used together with `cfg.dataset.pose_source='gs_cameras'` so that
        camera poses and SfM points share the same world frame.
        """
        try:
            model_cfg = getattr(cfg, "model", object())
            use_sfm = bool(getattr(model_cfg, "xyz_init_from_sfm", False))
            if not use_sfm:
                return False

            dataset_cfg = getattr(cfg, "dataset", object())
            data_root = getattr(dataset_cfg, "data_root", None)
            if not data_root:
                print(
                    "[Audio3DGSMonoDiff] xyz_init_from_sfm requested but "
                    "dataset.data_root is not set; falling back to random _xyz init."
                )
                return False

            sfm_file = getattr(model_cfg, "sfm_points_file", "cam_imags/sparse/0/points3D.bin")
            sfm_path = str(sfm_file)
            if not os.path.isabs(sfm_path):
                sfm_path = os.path.join(str(data_root), sfm_path)

            # Try common fallbacks if the requested file is missing.
            tried = [sfm_path]
            if not os.path.exists(sfm_path) and sfm_path.endswith(".bin"):
                tried.append(sfm_path[:-4] + ".ply")
                tried.append(sfm_path[:-4] + ".txt")
            for p in tried:
                if os.path.exists(p):
                    sfm_path = p
                    break
            if not os.path.exists(sfm_path):
                print(
                    "[Audio3DGSMonoDiff] xyz_init_from_sfm requested but "
                    f"SfM points file not found (tried: {tried}); falling back to random _xyz init."
                )
                return False

            # Load points3D from COLMAP model.
            xyz_np = None
            if sfm_path.endswith(".bin"):
                from libs.datasets.scene.colmap_loader import read_points3D_binary
                xyz_np, _, _ = read_points3D_binary(sfm_path)
            elif sfm_path.endswith(".txt"):
                from libs.datasets.scene.colmap_loader import read_points3D_text
                xyz_np, _, _ = read_points3D_text(sfm_path)
            else:
                # Avoid adding a hard dependency on plyfile here; users can point to .bin/.txt.
                print(
                    f"[Audio3DGSMonoDiff] xyz_init_from_sfm: unsupported file extension: {sfm_path}. "
                    "Use a COLMAP points3D.bin or points3D.txt."
                )
                return False

            if xyz_np is None:
                return False

            # Filter invalid points
            try:
                import numpy as np
                xyz_np = np.asarray(xyz_np, dtype=np.float32).reshape(-1, 3)
                ok = np.isfinite(xyz_np).all(axis=1)
                xyz_np = xyz_np[ok]
            except Exception:
                pass

            if xyz_np is None or len(xyz_np) == 0:
                print("[Audio3DGSMonoDiff] xyz_init_from_sfm: no valid points; falling back to random init.")
                return False

            xyz_sfm = torch.from_numpy(xyz_np).to(device=self._xyz.device, dtype=self._xyz.dtype)

            sample_mode = str(getattr(model_cfg, "sfm_points_sample", "random") or "random").lower()
            M = int(xyz_sfm.shape[0])
            N = int(self.n_points)
            if sample_mode in ("repeat", "tile"):
                reps = (N + M - 1) // M
                xyz_init = xyz_sfm.repeat(reps, 1)[:N]
            else:
                # Default: random sampling with replacement to fill all points.
                idx = torch.randint(low=0, high=M, size=(N,), device=xyz_sfm.device)
                xyz_init = xyz_sfm.index_select(0, idx)

            jitter_std = float(getattr(model_cfg, "sfm_points_jitter_std", 0.0) or 0.0)
            if jitter_std > 0:
                xyz_init = xyz_init + torch.randn_like(xyz_init) * jitter_std

            if getattr(self, "normalize_world_coords", False):
                xyz_init = xyz_init / float(self.max_norm)

            with torch.no_grad():
                self._xyz.copy_(xyz_init)

            print(f"[Audio3DGSMonoDiff] Initialized _xyz from SfM points: {sfm_path}")
            return True
        except Exception as e:
            print(f"[Audio3DGSMonoDiff] xyz_init_from_sfm failed: {e}")
            return False

    def _maybe_anchor_xyz_to_scene(self, cfg) -> None:
        """
        Optionally re-center _xyz around a scene-derived anchor center.

        This is useful when you want the learned point cloud to live in a more
        physically meaningful region of the ReplayNVAS coordinate system rather
        than near the origin.
        """
        try:
            model_cfg = getattr(cfg, "model", object())
            use_anchor = bool(getattr(model_cfg, "xyz_anchor_to_scene", False))
            if not use_anchor:
                return

            radius = float(getattr(model_cfg, "xyz_anchor_radius", 0.0))
            if radius <= 0.0:
                print(
                    "[Audio3DGSMonoDiff] xyz_anchor_to_scene is True but "
                    "xyz_anchor_radius <= 0; skipping anchor init."
                )
                return
            anchor_mode = str(getattr(model_cfg, "xyz_anchor_mode", "scene_mean") or "scene_mean").lower()
            try:
                anchor_viewpoint = int(getattr(model_cfg, "xyz_anchor_viewpoint", 0) or 0)
            except Exception:
                anchor_viewpoint = 0

            dataset_cfg = getattr(cfg, "dataset", object())
            data_root = getattr(dataset_cfg, "data_root", None)
            scene_scope = getattr(dataset_cfg, "scene_scope", None)
            if not data_root or not scene_scope or not isinstance(scene_scope, str):
                print(
                    "[Audio3DGSMonoDiff] xyz_anchor_to_scene requested but "
                    "dataset.data_root or dataset.scene_scope is not set; "
                    "falling back to random _xyz init."
                )
                return

            pose_source = str(getattr(dataset_cfg, "pose_source", "fixed_rotation") or "fixed_rotation").lower()

            vals = []
            viewpoint_centers = {}

            def _maybe_add_center(view_id, center):
                vals.append(center)
                if view_id is None:
                    return
                try:
                    viewpoint_centers[int(view_id)] = center
                except Exception:
                    return

            def _parse_view_id_from_name(name):
                text = os.path.splitext(os.path.basename(str(name)))[0]
                m = re.search(r"(\d+)$", text)
                if not m:
                    return None
                try:
                    return int(m.group(1))
                except Exception:
                    return None

            def _replay_pose_to_opencv_center(R_blob, T_blob):
                R = torch.tensor(struct.unpack("<9f", R_blob), dtype=torch.float32).reshape(3, 3)
                T = torch.tensor(struct.unpack("<3f", T_blob), dtype=torch.float32).reshape(3)
                T[:2] *= -1.0
                R[:, :2] *= -1.0
                R_w2c = R.T
                center = -(R_w2c.T @ T)
                return center.tolist()

            if pose_source in ("gs", "gs_cameras", "colmap"):
                gs_file = getattr(dataset_cfg, "gs_cameras_file", "cam_imags/gs_cameras.json")
                gs_path = str(gs_file)
                if not os.path.isabs(gs_path):
                    gs_path = os.path.join(str(data_root), gs_path)
                if not os.path.exists(gs_path):
                    print(
                        f"[Audio3DGSMonoDiff] xyz_anchor_to_scene requested but "
                        f"{gs_path} not found; falling back to random _xyz init."
                    )
                    return
                with open(gs_path, "r") as f:
                    cams = json.load(f)
                for cam in cams:
                    name = cam.get("img_name", "")
                    if not isinstance(name, str) or "_" not in name:
                        continue
                    try:
                        sc, vp = name.rsplit("_", 1)
                        _ = int(vp)
                    except Exception:
                        continue
                    if sc != scene_scope:
                        continue
                    pos = cam.get("position", None)
                    if not isinstance(pos, (list, tuple)) or len(pos) != 3:
                        continue
                    try:
                        center = [float(pos[0]), float(pos[1]), float(pos[2])]
                        _maybe_add_center(vp, center)
                    except Exception:
                        continue
            elif pose_source in ("colmap_images", "images_txt", "images_bin"):
                pose_file = getattr(dataset_cfg, "pose_file", "images.txt")
                pose_path = str(pose_file)
                if not os.path.isabs(pose_path):
                    pose_path = os.path.join(str(data_root), pose_path)

                alt_paths = [pose_path]
                if pose_path.endswith(".txt"):
                    alt_paths.append(pose_path[:-4] + ".bin")
                elif pose_path.endswith(".bin"):
                    alt_paths.append(pose_path[:-4] + ".txt")
                else:
                    alt_paths.extend(
                        [
                            os.path.join(str(data_root), "images.txt"),
                            os.path.join(str(data_root), "images.bin"),
                        ]
                    )

                pose_records = None
                chosen_path = None
                for p in alt_paths:
                    if not os.path.exists(p):
                        continue
                    try:
                        if p.endswith(".bin"):
                            pose_records = read_extrinsics_binary(p)
                        else:
                            pose_records = read_extrinsics_text(p)
                        chosen_path = p
                        break
                    except Exception:
                        continue

                if pose_records is None:
                    print(
                        f"[Audio3DGSMonoDiff] xyz_anchor_to_scene requested but "
                        f"could not read COLMAP image poses from {alt_paths}; "
                        "falling back to random _xyz init."
                    )
                    return

                for image in pose_records.values():
                    try:
                        R_w2c = torch.as_tensor(
                            qvec2rotmat(image.qvec),
                            dtype=self._xyz.dtype,
                            device=self._xyz.device,
                        ).reshape(3, 3)
                        t_w2c = torch.as_tensor(
                            image.tvec,
                            dtype=self._xyz.dtype,
                            device=self._xyz.device,
                        ).reshape(3)
                        center = -(R_w2c.transpose(0, 1) @ t_w2c)
                        if torch.isfinite(center).all():
                            _maybe_add_center(_parse_view_id_from_name(getattr(image, "name", "")), center.tolist())
                    except Exception:
                        continue

                if vals:
                    print(
                        f"[Audio3DGSMonoDiff] xyz_anchor_to_scene loaded {len(vals)} "
                        f"camera centers from COLMAP poses: {chosen_path}"
                    )
            elif pose_source in ("replay_metadata", "metadata_sqlite", "replay_sqlite"):
                meta_file = getattr(dataset_cfg, "replay_metadata_file", "data/Replay/metadata.sqlite")
                meta_path = str(meta_file)
                candidates = []
                if os.path.isabs(meta_path):
                    candidates.append(meta_path)
                else:
                    candidates.append(meta_path)
                    candidates.append(os.path.join(str(data_root), meta_path))
                    candidates.append(os.path.join(os.getcwd(), meta_path))
                meta_path = next((p for p in candidates if os.path.exists(p)), "")
                if not meta_path:
                    print(
                        f"[Audio3DGSMonoDiff] xyz_anchor_to_scene requested but "
                        f"Replay metadata sqlite was not found. Tried: {candidates}"
                    )
                    return
                try:
                    con = sqlite3.connect(meta_path)
                    rows = con.execute(
                        """
                        SELECT sensor_name, _viewpoint_R, _viewpoint_T
                        FROM frame_annots
                        WHERE sequence_name = ?
                          AND sensor_name LIKE 'DSLR-%'
                          AND _viewpoint_R IS NOT NULL
                          AND _viewpoint_T IS NOT NULL
                        ORDER BY sensor_name, frame_timestamp, frame_number
                        """,
                        (scene_scope,),
                    ).fetchall()
                    con.close()
                except Exception as e:
                    print(
                        f"[Audio3DGSMonoDiff] xyz_anchor_to_scene could not read "
                        f"Replay metadata sqlite {meta_path}: {e}"
                    )
                    return
                seen_sensors = set()
                for sensor_name, blob_R, blob_T in rows:
                    if sensor_name in seen_sensors:
                        continue
                    seen_sensors.add(sensor_name)
                    try:
                        center = _replay_pose_to_opencv_center(blob_R, blob_T)
                        _maybe_add_center(_parse_view_id_from_name(sensor_name), center)
                    except Exception:
                        continue
            else:
                fixed_file = getattr(dataset_cfg, "fixed_pose_file", "camera_positions_fixed_rotation.json")
                cam_file = str(fixed_file)
                if not os.path.isabs(cam_file):
                    cam_file = os.path.join(str(data_root), cam_file)
                if not os.path.exists(cam_file):
                    print(
                        f"[Audio3DGSMonoDiff] xyz_anchor_to_scene requested but "
                        f"{cam_file} not found; falling back to random _xyz init."
                    )
                    return

                with open(cam_file, "r") as f:
                    cam_data = json.load(f)
                centers = cam_data.get("centers", {})
                scene_centers = centers.get(scene_scope)
                if not isinstance(scene_centers, dict) or len(scene_centers) == 0:
                    print(
                        f"[Audio3DGSMonoDiff] xyz_anchor_to_scene: no centers for "
                        f"scene '{scene_scope}'; falling back to random _xyz init."
                    )
                    return

                for name, c in scene_centers.items():
                    if not isinstance(name, str) or not name.startswith("DSLR-"):
                        continue
                    if not isinstance(c, (list, tuple)) or len(c) != 3:
                        continue
                    try:
                        center = [float(c[0]), float(c[1]), float(c[2])]
                        _maybe_add_center(_parse_view_id_from_name(name), center)
                    except Exception:
                        continue

            if not vals:
                print(
                    f"[Audio3DGSMonoDiff] xyz_anchor_to_scene: no valid DSLR-* "
                    f"centers for scene '{scene_scope}'; keeping random _xyz."
                )
                return

            if anchor_mode == "viewpoint":
                anchor_center = viewpoint_centers.get(int(anchor_viewpoint))
                if anchor_center is None:
                    available = sorted(viewpoint_centers.keys())
                    print(
                        f"[Audio3DGSMonoDiff] xyz_anchor_to_scene requested viewpoint={anchor_viewpoint} "
                        f"but it was not found in pose_source={pose_source}. Available viewpoints: {available}"
                    )
                    return
                center_scene = torch.tensor(
                    anchor_center, dtype=self._xyz.dtype, device=self._xyz.device
                )
                anchor_desc = f"viewpoint {anchor_viewpoint}"
            else:
                center_scene = torch.tensor(
                    vals, dtype=self._xyz.dtype, device=self._xyz.device
                ).mean(dim=0)
                anchor_desc = f"scene '{scene_scope}' camera-center mean"

            with torch.no_grad():
                offsets = torch.randn_like(self._xyz) * radius
                xyz_init = center_scene.view(1, 3) + offsets
                if getattr(self, "normalize_world_coords", False):
                    xyz_init = xyz_init / float(self.max_norm)
                self._xyz.copy_(xyz_init)

            print(
                f"[Audio3DGSMonoDiff] Anchored _xyz around {anchor_desc} "
                f"with radius={radius:.2f}."
            )
        except Exception as e:
            # Fail soft: keep the original random init.
            print(f"[Audio3DGSMonoDiff] xyz_anchor_to_scene failed: {e}")

    # -----------------------------
    # Numeric safety helpers
    # -----------------------------
    def _safe_sh_mono(self) -> torch.Tensor:
        sh = torch.nan_to_num(self._sh_mono, nan=0.0, posinf=0.0, neginf=0.0)
        # abs_mag_output: allow larger dynamic range (do not clamp to small range)
        if getattr(self, "use_abs_mag_output", False):
            return sh
        return torch.clamp(sh, min=-5.0, max=5.0)

    def _safe_sh_diff(self) -> torch.Tensor:
        sh = torch.nan_to_num(self._sh_diff, nan=0.0, posinf=0.0, neginf=0.0)
        if getattr(self, "use_abs_mag_output", False):
            return sh
        return torch.clamp(sh, min=-5.0, max=5.0)

    def _safe_rotation_quaternions(self) -> torch.Tensor:
        q = torch.nan_to_num(self._rotation, nan=0.0, posinf=0.0, neginf=0.0)
        norms = torch.norm(q, dim=-1, keepdim=True)
        bad = (norms < 1e-8) | (~torch.isfinite(norms))
        ident_full = q.new_zeros(q.shape)
        ident_full[:, 0] = 1.0
        q = torch.where(bad, ident_full, q)
        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-8)
        return q

    def _safe_xyz(self) -> torch.Tensor:
        xyz = torch.nan_to_num(self._xyz, nan=0.0, posinf=0.0, neginf=0.0)
        bound = float(self.max_norm) * 50.0
        return torch.clamp(xyz, min=-bound, max=bound)

    def _create_tf_grid(self) -> torch.Tensor:
        """Create time-frequency coordinate grid for spectrogram pixels"""
        f_coords = torch.arange(self.freq_num).float()
        t_coords = torch.arange(self.time_num).float()
        f_grid, t_grid = torch.meshgrid(f_coords, t_coords, indexing="ij")
        tf_grid = torch.stack([f_grid.flatten(), t_grid.flatten()], dim=1)
        return tf_grid

    # -----------------------------
    # Geometry utilities
    # -----------------------------
    def rotation_matrix_to_quaternion(self, R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrices to quaternions

        Args:
            R: [N, 3, 3] rotation matrices
        Returns:
            [N, 4] quaternions (w, x, y, z)
        """
        N = R.shape[0]
        quaternions = torch.zeros((N, 4), device=R.device)

        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

        mask1 = trace > 0
        s = torch.sqrt(trace[mask1] + 1.0) * 2
        quaternions[mask1, 0] = 0.25 * s
        quaternions[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        quaternions[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        quaternions[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

        mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        quaternions[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        quaternions[mask2, 1] = 0.25 * s
        quaternions[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        quaternions[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

        mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        quaternions[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        quaternions[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        quaternions[mask3, 2] = 0.25 * s
        quaternions[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

        mask4 = (~mask1) & (~mask2) & (~mask3)
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        quaternions[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        quaternions[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        quaternions[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        quaternions[mask4, 3] = 0.25 * s

        return quaternions

    def quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert normalized quaternions to rotation matrices.

        Args:
            q: [N, 4] normalized quaternions
        Returns:
            [N, 3, 3] rotation matrices
        """
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-8)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        R = torch.zeros(q.shape[0], 3, 3, device=q.device)
        R[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
        R[:, 0, 1] = 2.0 * (xy - wz)
        R[:, 0, 2] = 2.0 * (xz + wy)

        R[:, 1, 0] = 2.0 * (xy + wz)
        R[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
        R[:, 1, 2] = 2.0 * (yz - wx)

        R[:, 2, 0] = 2.0 * (xz - wy)
        R[:, 2, 1] = 2.0 * (yz + wx)
        R[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
        return R

    def compute_relative_positions(
        self, cam_pose: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute relative vectors from points to microphone in both world/camera frames.

        Returns:
            relative_pos_world_norm: [B, N, 3] world-frame vectors normalized by max_norm
            dir_pp_world:            [B, N, 3] from microphone to point (world, unnormalized)
            relative_pos_cam_norm:   [B, N, 3] camera-frame vectors normalized by max_norm
        """
        B = cam_pose.shape[0]

        microphone_pos = cam_pose[:, :3]
        R_cam = None
        if cam_pose.shape[1] >= 12:
            try:
                R_cam = cam_pose[:, 3:].reshape(B, 3, 3)
            except Exception:
                R_cam = None

        safe_xyz = self._safe_xyz()
        if getattr(self, "normalize_world_coords", False):
            # Use coordinates normalized by max_norm for improved conditioning.
            mic_norm = microphone_pos / float(self.max_norm)  # [B, 3]
            xyz_norm = safe_xyz  # _xyz is already in normalized units
            rel_world_norm = (
                mic_norm.unsqueeze(1).repeat(1, self.n_points, 1)
                - xyz_norm.unsqueeze(0).repeat(B, 1, 1)
            )  # [B, N, 3], already normalized by max_norm
            # World-frame vector from microphone to point (unnormalized, in world units)
            dir_pp_world = -rel_world_norm * float(self.max_norm)

            if R_cam is not None:
                R_t = R_cam.transpose(1, 2)
                # Rotate normalized relative vectors into camera frame; still normalized.
                rel_cam_norm = torch.matmul(rel_world_norm, R_t)
            else:
                rel_cam_norm = rel_world_norm
        else:
            # Original behavior: work in world units then divide by max_norm.
            rel_world = (
                microphone_pos.unsqueeze(1).repeat(1, self.n_points, 1)
                - safe_xyz.unsqueeze(0).repeat(B, 1, 1)
            )

            dir_pp_world = -rel_world
            rel_world_norm = rel_world / self.max_norm

            if R_cam is not None:
                R_t = R_cam.transpose(1, 2)
                rel_cam = torch.matmul(rel_world, R_t)
                rel_cam_norm = rel_cam / self.max_norm
            else:
                rel_cam_norm = rel_world_norm

        return rel_world_norm, dir_pp_world, rel_cam_norm

    # -----------------------------
    # SH evaluation for mono / diff
    # -----------------------------
    def _eval_sh_field(self, sh_params: torch.Tensor, relative_pos: torch.Tensor, apply_softplus: bool = True) -> torch.Tensor:
        """Evaluate a SH field (mono or diff) for given relative positions.

        Args:
            sh_params: [N, 1, C] sanitized SH coefficients
            relative_pos: [B, N, 3] relative positions
        Returns:
            [B, N] non-negative directional magnitudes
        """
        B, N, _ = relative_pos.shape

        # Normalize direction vectors first (robust to zero vectors)
        dir_normalized = F.normalize(relative_pos, dim=-1, eps=1e-8)

        if self.use_point_rotation:
            # Rotate viewing directions into each point's local frame via per-point quaternion.
            rotation_matrices = self.quaternion_to_rotation_matrix(self._safe_rotation_quaternions())
            rotation_matrices_inv = rotation_matrices.transpose(1, 2)
            rotation_matrices_inv_expanded = rotation_matrices_inv.unsqueeze(0).repeat(B, 1, 1, 1)
            dir_normalized_expanded = dir_normalized.unsqueeze(-1)
            dir_local = torch.matmul(rotation_matrices_inv_expanded, dir_normalized_expanded).squeeze(-1)
        else:
            # Directly use world/camera-frame directions (no per-point rotation), like vanilla 3DGS.
            dir_local = dir_normalized

        dir_local = torch.nan_to_num(dir_local, nan=0.0, posinf=0.0, neginf=0.0)

        dir_flat = dir_local.reshape(-1, 3)

        sh_coeffs_expanded = (
            sh_params.unsqueeze(0)
            .repeat(B, 1, 1, 1)
            .reshape(-1, 1, self.num_sh_coeffs)
        )

        sh_values_flat = eval_sh(self.sh_degree, sh_coeffs_expanded, dir_flat)
        sh_values_flat = torch.nan_to_num(sh_values_flat, nan=0.0, posinf=0.0, neginf=0.0)

        sh_values = sh_values_flat.squeeze(-1).reshape(B, N)
        # For mono field we keep non-negativity; for diff field we keep sign.
        if apply_softplus:
            # When abs_mag_output=True, we want direct magnitude output;
            # skip mask shaping and keep raw values (optionally ReLU).
            if getattr(self, "use_abs_mag_output", False):
                sh_values = F.relu(sh_values)
            else:
                act = getattr(self, "mono_mask_activation", "sigmoid")
                if act in ("none", "identity", "linear"):
                    # No activation: raw SH values (can be negative / >1)
                    pass
                elif act in ("relu", "clamp0"):
                    sh_values = F.relu(sh_values)
                else:
                    # Default: symmetric sigmoid gate (0,2), 0 -> 1
                    sh_values = 2.0 * torch.sigmoid(sh_values)
        return sh_values

    def eval_mono_diff_fields(
        self, relative_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate mono and diff SH fields for given relative positions.

        Args:
            relative_pos: [B, N, 3]
        Returns:
            mono_vals: [B, N]
            diff_vals: [B, N]
        """
        # Mono field: non-negative magnitude via softplus
        mono_vals = self._eval_sh_field(self._safe_sh_mono(), relative_pos, apply_softplus=True)
        # Diff field: keep signed responses (no softplus), so diff can enhance either ear.
        diff_vals = self._eval_sh_field(self._safe_sh_diff(), relative_pos, apply_softplus=False)
        return mono_vals, diff_vals

    # -----------------------------
    # Point -> spectrogram features
    # -----------------------------
    def points_to_spectrogram_mono_diff(
        self,
        point_values_mono: torch.Tensor,
        point_values_diff: torch.Tensor,
        relative_pos: torch.Tensor,
        source_magnitude: torch.Tensor,
        ild_spec: torch.Tensor = None,
        side_spec: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare renderer inputs from mono/diff SH fields and geometry.

        Args:
            point_values_mono: [B, N] mono SH-evaluated magnitudes
            point_values_diff: [B, N] diff SH-evaluated magnitudes
            relative_pos: [B, N, 3] normalized (by max_norm) vectors point->mic
            source_magnitude: [B, F, T] magnitude spectrogram of source mono

        Returns:
            mono_features: [B, 3, F, T] = [source_mag, mono_SH, inv_distance]
            diff_features: [B, C_diff, F, T] (first channel = diff_SH, optional extra cues)
        """
        B = point_values_mono.shape[0]

        spec_mono = point_values_mono.reshape(B, self.freq_num, self.time_num)
        spec_diff = point_values_diff.reshape(B, self.freq_num, self.time_num)

        distances_norm = torch.norm(relative_pos, dim=-1).reshape(
            B, self.freq_num, self.time_num
        )
        distances_world = distances_norm * self.max_norm
        inv_distance = 1.0 / (1e-3 + distances_world)

        if source_magnitude.shape[-2:] != spec_mono.shape[-2:]:
            source_magnitude = F.interpolate(
                source_magnitude.unsqueeze(1),
                size=spec_mono.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        mono_features = torch.stack([source_magnitude, spec_mono, inv_distance], dim=1)

        diff_chans = [spec_diff]
        if self.use_stereo_cues and (ild_spec is not None):
            if ild_spec.shape[-2:] != spec_diff.shape[-2:]:
                ild_spec = F.interpolate(
                    ild_spec.unsqueeze(1),
                    size=spec_diff.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            diff_chans.append(ild_spec)
        if self.diff_use_inv_distance:
            diff_chans.append(inv_distance)
        if self.diff_use_side_mag and (side_spec is not None):
            if side_spec.shape[-2:] != spec_diff.shape[-2:]:
                side_spec = F.interpolate(
                    side_spec.unsqueeze(1),
                    size=spec_diff.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            diff_chans.append(side_spec)

        diff_features = torch.stack(diff_chans, dim=1)

        mono_features = torch.nan_to_num(mono_features, nan=0.0, posinf=0.0, neginf=0.0)
        diff_features = torch.nan_to_num(diff_features, nan=0.0, posinf=0.0, neginf=0.0)
        return mono_features, diff_features

    # -----------------------------
    # Forward: (cam_pose, source_audio) -> binaural
    # -----------------------------
    def forward(
        self,
        cam_pose: torch.Tensor,
        source_audio: torch.Tensor,
        is_val: bool = False,
        return_mag: bool = False,
    ):
        """
        Forward pass

        Args:
            cam_pose: [B, 12] Camera/mic poses
            source_audio: [B, C, T] Source audio waveform (near.wav; mono or stereo)
            is_val: unused, kept for compatibility
            return_mag: if True, also return left/right magnitude spectrograms

        Returns:
            binaural_output: [B, 2, T]
        """
        device = cam_pose.device
        B = cam_pose.shape[0]

        # STFT params aligned with existing code
        n_fft = 512
        hop_length = 160
        window_length = 400
        torch_window = torch.hamming_window(window_length).to(device)

        # Convert source audio to spectrogram
        stereo_input = source_audio.dim() > 2 and source_audio.shape[1] > 1
        ild_spec = None
        side_spec = None

        if stereo_input:
            sig_L = source_audio[:, 0]
            sig_R = source_audio[:, 1]
            spec_L = torch.stft(
                sig_L,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=window_length,
                window=torch_window,
                return_complex=True,
            )
            spec_R = torch.stft(
                sig_R,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=window_length,
                window=torch_window,
                return_complex=True,
            )
            mag_L = torch.abs(spec_L)
            mag_R = torch.abs(spec_R)
            phase_L = torch.angle(spec_L)
            phase_R = torch.angle(spec_R)
            source_magnitude = 0.5 * (mag_L + mag_R)
            source_phase_L = torch.nan_to_num(
                phase_L, nan=0.0, posinf=0.0, neginf=0.0
            )
            source_phase_R = torch.nan_to_num(
                phase_R, nan=0.0, posinf=0.0, neginf=0.0
            )
            source_magnitude = torch.nan_to_num(
                source_magnitude, nan=0.0, posinf=0.0, neginf=0.0
            )
            if self.use_stereo_cues:
                eps = 1e-8
                ild_spec = torch.log(torch.clamp(mag_L, min=eps)) - torch.log(
                    torch.clamp(mag_R, min=eps)
                )
                ild_spec = torch.nan_to_num(
                    ild_spec, nan=0.0, posinf=0.0, neginf=0.0
                )
            if self.diff_use_side_mag:
                side_spec = 0.5 * torch.clamp(mag_L - mag_R, min=0.0)
        else:
            # Mono case: regular STFT
            source_mono = (
                source_audio.squeeze() if source_audio.dim() > 1 else source_audio
            )
            source_spec_complex = torch.stft(
                source_mono,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=window_length,
                window=torch_window,
                return_complex=True,
            )
            source_magnitude = torch.abs(source_spec_complex)
            source_phase = torch.angle(source_spec_complex)
            source_magnitude = torch.nan_to_num(
                source_magnitude, nan=0.0, posinf=0.0, neginf=0.0
            )
            source_phase = torch.nan_to_num(
                source_phase, nan=0.0, posinf=0.0, neginf=0.0
            )

        # 1) Compute relative vectors in world/camera frames
        rel_world_norm, dir_pp, rel_cam_norm = self.compute_relative_positions(cam_pose)

        # 2) Evaluate mono & diff SH fields (camera or world frame)
        rel_for_sh = rel_cam_norm if self.use_cam_rotation else rel_world_norm
        if self.use_cam_rotation and self.flip_cam_y_for_sh:
            rel_for_sh = rel_for_sh.clone()
            rel_for_sh[..., 1] *= -1.0
        mono_vals, diff_vals = self.eval_mono_diff_fields(rel_for_sh)

        # 3) Build spectrogram features for the dual-branch renderer
        mono_features, diff_features = self.points_to_spectrogram_mono_diff(
            mono_vals,
            diff_vals,
            rel_world_norm,
            source_magnitude,
            ild_spec=ild_spec,
            side_spec=side_spec,
        )

        # 4) Dual-branch U-Net rendering to predict binaural masks
        mono_mask, diff_mask = self.renderer(mono_features, diff_features)

        # 5) Apply masks to source spectrogram (mono/diff)
        if mono_mask.shape[-2:] != source_magnitude.shape[-2:]:
            mono_mask = F.interpolate(
                mono_mask, size=source_magnitude.shape[-2:], mode="bilinear"
            )
            diff_mask = F.interpolate(
                diff_mask, size=source_magnitude.shape[-2:], mode="bilinear"
            )

        mono_mask = mono_mask.squeeze(1)
        diff_mask = diff_mask.squeeze(1)

        mono_mag = mono_mask * source_magnitude
        mono_mag = torch.nan_to_num(mono_mag, nan=0.0, posinf=0.0, neginf=0.0)
        diff_envelope = diff_mask

        # Cascaded synthesis: diff envelope modulates mono energy to avoid shortcut spatialization.
        left_magnitude = torch.relu(mono_mag * (1.0 + diff_envelope))
        right_magnitude = torch.relu(mono_mag * (1.0 - diff_envelope))

        if stereo_input:
            left_phase = source_phase_L
            right_phase = source_phase_R
        else:
            left_phase = source_phase
            right_phase = source_phase

        left_complex = torch.polar(left_magnitude, left_phase)
        right_complex = torch.polar(right_magnitude, right_phase)

        sig_len = (
            source_audio.shape[-1]
            if isinstance(source_audio, torch.Tensor)
            else None
        )
        left_audio = torch.istft(
            left_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_length,
            window=torch_window,
            length=sig_len,
        )
        right_audio = torch.istft(
            right_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_length,
            window=torch_window,
            length=sig_len,
        )

        binaural_output = torch.stack([left_audio, right_audio], dim=1)
        binaural_output = torch.nan_to_num(
            binaural_output, nan=0.0, posinf=0.0, neginf=0.0
        )

        if return_mag:
            return binaural_output, left_magnitude, right_magnitude
        return binaural_output


def build_model(cfg, gaussian_model=None, scene=None):
    """Build Audio3DGSMonoDiff model."""
    freq_num = cfg.dataset.get("H", 257)
    time_num = cfg.dataset.get("W", 348)
    model = Audio3DGSMonoDiff(cfg, freq_num=freq_num, time_num=time_num)
    return model
