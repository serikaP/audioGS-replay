"""
ReplayNVAS dataset
7 viewpoints for training, 1 for testing per scene
"""

import os
import pickle
import random
import sqlite3
import numpy as np
import torch
import torch.utils.data as data
import librosa
import json
from scipy.signal import butter, sosfiltfilt


class Audio3DGSReplayNVASViewpointDataset(data.Dataset):
    """
    Uses 7 of 8 viewpoints for training, 1 for testing per scene
    """
    
    def __init__(self, cfg, split='train', test_viewpoint=8, selected_scenes=None):
        super(Audio3DGSReplayNVASViewpointDataset, self).__init__()
        
        self.cfg = cfg
        self.split = split
        self.test_viewpoint = test_viewpoint  # Which viewpoint to use for testing (1-8)
        self.audio_len = cfg.dataset.audio_len
        self.sampling_rate = cfg.dataset.sr
        # Optional: choose input source type: 'near' (default) or 'viewpoint'
        self.input_source = str(getattr(cfg.dataset, 'input_source', 'near') or 'near').lower()
        try:
            self.input_viewpoint = int(getattr(cfg.dataset, 'input_viewpoint', 0) or 0)
        except Exception:
            self.input_viewpoint = 0
        # Optional: when using input_source=='viewpoint', randomly pick an input
        # viewpoint from the training set per sample (and not equal to target vp)
        # Only applied on split=='train'.
        self.random_input_viewpoint = bool(getattr(cfg.dataset, 'random_input_viewpoint', False))
        # Optional: enable random input viewpoint for val/test split as well
        # (exclude current target viewpoint)
        self.random_input_viewpoint_eval = bool(getattr(cfg.dataset, 'random_input_viewpoint_eval', False))
        # Optional: use NVAS metadata_v2.json list to select frames (for fair comparison)
        self.use_metadata = bool(getattr(cfg.dataset, 'use_metadata', False))
        self.metadata_file = getattr(cfg.dataset, 'metadata_file', 'metadata_v2.json')
        # Training crop mode: random vs center (match test)
        self.train_random_crop = bool(getattr(cfg.dataset, 'train_random_crop', True))
        # Frame sampling stride: control how densely to sample frame folders per scene
        # Set to 1 to use every frame, larger to subsample
        self.frame_stride_train = int(getattr(cfg.dataset, 'frame_stride_train', 20))
        self.frame_stride_test = int(getattr(cfg.dataset, 'frame_stride_test', 20))
        # Scene-level normalization switch (paper: normalize by max RMS per scene)
        self.scene_level_normalize = getattr(cfg.dataset, 'scene_normalize', True)
        # Bandpass
        bp_cfg = getattr(cfg.dataset, 'bandpass', None)
        self.bp_enable = bool(getattr(bp_cfg, 'enable', False)) if bp_cfg is not None else False
        self.bp_low = float(getattr(bp_cfg, 'low_hz', 150.0)) if bp_cfg is not None else 150.0
        self.bp_high = float(getattr(bp_cfg, 'high_hz', -1.0)) if bp_cfg is not None else -1.0
        
        # Dataset paths
        self.dataset_path = cfg.dataset.data_root
        
        # Selected scenes for evaluation (as mentioned in paper: 6 scenes)
        if selected_scenes is None:
            # Use 6 representative scenes that have good data quality
            self.selected_scenes = ['SC-1027', 'SC-1024', 'SC-1040', 'SC-1042', 'SC-1044', 'SC-1052']
        else:
            # Normalize potential non-ASCII hyphens to ASCII '-'
            dash_variants = ['\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '\u2212']
            def norm_scene(s):
                if not isinstance(s, str):
                    return s
                for dv in dash_variants:
                    s = s.replace(dv, '-')
                return s.strip()
            self.selected_scenes = [norm_scene(s) for s in selected_scenes]
            
        print(f"Using {len(self.selected_scenes)} scenes for evaluation: {self.selected_scenes}")
        print(f"Test viewpoint: {self.test_viewpoint}, Split: {self.split}")
        
        # Load camera poses (centers + rotations)
        self.pose_source = str(getattr(getattr(cfg, 'dataset', object()), 'pose_source', 'fixed_rotation') or 'fixed_rotation').lower()
        self.replay_metadata_frame_lookup = str(
            getattr(
                getattr(cfg, "dataset", object()),
                "replay_metadata_frame_lookup",
                "offset",
            )
            or "offset"
        ).lower()
        self._replay_metadata_pose_sequences = {}
        # Backward-compat: older configs used `dataset.use_gs_cameras` to override rotations only.
        self.use_gs_cameras = bool(getattr(getattr(cfg, 'dataset', object()), 'use_gs_cameras', False))
        # How to interpret fixed_rotation `rotations` vectors.
        self.fixed_rotation_mode = str(
            getattr(getattr(cfg, "dataset", object()), "fixed_rotation_mode", "lookat") or "lookat"
        ).lower()
        try:
            wup = getattr(getattr(cfg, "dataset", object()), "fixed_rotation_world_up", [0.0, 0.0, 1.0])
            wup = np.asarray(wup, dtype=np.float32).reshape(-1)
            if wup.shape[0] != 3 or (not np.isfinite(wup).all()):
                raise ValueError("invalid world_up")
            # Avoid degenerate world_up.
            if float(np.linalg.norm(wup)) < 1e-6:
                raise ValueError("world_up too small")
            self.fixed_rotation_world_up = wup
        except Exception:
            self.fixed_rotation_world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._load_camera_poses()

        # For runtime randomization in __getitem__ (per-sample re-random)
        # store frame path and target viewpoint per sample
        self._sample_frame_paths = []
        self._sample_target_vps = []
        # Store raw (unfiltered) target binaural and environment residual per sample
        self._binaural_audio_raw = []
        self._env_residuals = []

        # Load audio and camera data
        self._load_dataset()
        
        # Init norm-related containers
        self._norm_scene_to_max = {}
        self._norm_factors = [1.0 for _ in range(len(self.binaural_audio))]
        self._base_scene_ids = []
        
        # Apply scene-level normalization if enabled
        if self.scene_level_normalize:
            self._apply_scene_normalization()

    # ------------------------------------------------------------
    # ViGAS bandpass filter helpers 
    # ------------------------------------------------------------
    @staticmethod
    def _butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

    @classmethod
    def _butter_bandpass_filter(cls, data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 5) -> np.ndarray:
        sos = cls._butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfiltfilt(sos, data, axis=-1)
        return y
        
    def _load_camera_positions(self):
        """Load camera positions from camera_positions_fixed_rotation.json"""
        fixed_file = getattr(getattr(self.cfg, 'dataset', object()), 'fixed_pose_file', 'camera_positions_fixed_rotation.json')
        camera_pos_file = os.path.join(self.dataset_path, str(fixed_file))
        
        if not os.path.exists(camera_pos_file):
            raise FileNotFoundError(f"Camera positions file not found: {camera_pos_file}")
            
        with open(camera_pos_file, 'r') as f:
            camera_data = json.load(f)
            
        self.camera_centers = camera_data['centers']
        self.camera_rotations = camera_data['rotations']
        
        print(f"Loaded camera positions for {len(self.camera_centers)} scenes")

    def _load_gs_camera_poses(self):
        """
        Load camera centers and rotations from cam_imags/gs_cameras.json (COLMAP/SfM).

        The file is produced by `libs/utils/camera_utils.py:camera_to_JSON` and stores:
          - position: camera center in world coordinates
          - rotation: camera->world rotation matrix (3x3)
          - img_name: 'SC-XXXX_<vp>'
        """
        gs_file = getattr(getattr(self.cfg, 'dataset', object()), 'gs_cameras_file', 'cam_imags/gs_cameras.json')
        gs_path = os.path.join(self.dataset_path, str(gs_file))
        if not os.path.exists(gs_path):
            raise FileNotFoundError(f"gs_cameras file not found: {gs_path}")

        with open(gs_path, 'r') as f:
            cams = json.load(f)

        centers = {}
        rotations = {}
        for cam in cams:
            name = cam.get('img_name', '')
            if not isinstance(name, str) or '_' not in name:
                continue
            try:
                scene, vp = name.rsplit('_', 1)
                vp_id = int(vp)
            except Exception:
                continue
            dslr_key = f'DSLR-{vp_id}'
            pos = cam.get('position', None)
            rot = cam.get('rotation', None)
            if pos is None or rot is None:
                continue
            try:
                p = np.asarray(pos, dtype=np.float32).reshape(3).tolist()
                # Stored as camera->world (C2W), matching Audio3DGS expectations.
                R_c2w = np.asarray(rot, dtype=np.float32).reshape(3, 3)
            except Exception:
                continue
            centers.setdefault(scene, {})[dslr_key] = p
            rotations.setdefault(scene, {})[dslr_key] = R_c2w.tolist()

        self.camera_centers = centers
        self.camera_rotations = rotations
        print(f"[gs_cameras] Loaded camera poses for {len(self.camera_centers)} scenes from {gs_path}")

    @staticmethod
    def _replay_pose_to_opencv(R_blob, T_blob):
        """
        Convert Replay/PyTorch3D camera fields to OpenCV world-to-camera pose.

        The Replay metadata stores `_viewpoint_R/_viewpoint_T` in the same
        convention used by PyTorch3D. This conversion mirrors
        `tools/replay_sc1107_vggt_novel_view.py:replay_pose_to_opencv`.
        """
        R = np.frombuffer(R_blob, dtype="<f4").reshape(3, 3).astype(np.float32)
        T = np.frombuffer(T_blob, dtype="<f4").reshape(3).astype(np.float32)
        T[:2] *= -1.0
        R[:, :2] *= -1.0
        R_w2c = R.T.astype(np.float32)
        t_w2c = T.astype(np.float32)
        return R_w2c, t_w2c

    def _resolve_replay_metadata_path(self):
        meta_file = str(
            getattr(
                getattr(self.cfg, "dataset", object()),
                "replay_metadata_file",
                "data/Replay/metadata.sqlite",
            )
            or "data/Replay/metadata.sqlite"
        )
        candidates = []
        if os.path.isabs(meta_file):
            candidates.append(meta_file)
        else:
            candidates.append(meta_file)
            candidates.append(os.path.join(self.dataset_path, meta_file))
            candidates.append(os.path.join(os.getcwd(), meta_file))
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Replay metadata sqlite not found. Tried: {candidates}"
        )

    def _load_replay_metadata_camera_poses(self):
        """
        Load DSLR camera centers/rotations from original Replay metadata.sqlite.

        Stored pose format:
          - `frame_annots._viewpoint_R`: 3x3 float32 PyTorch3D rotation blob
          - `frame_annots._viewpoint_T`: 3 float32 PyTorch3D translation blob

        Audio3DGS consumes pose as `[camera_center, R_w2c.flatten()]`, so we
        convert the Replay fields to OpenCV-style world-to-camera rotation and
        derive center as `C = -R_w2c.T @ t_w2c`.
        """
        sqlite_path = self._resolve_replay_metadata_path()
        centers = {}
        rotations = {}
        pose_sequences = {}
        con = sqlite3.connect(sqlite_path)
        try:
            rows = con.execute(
                """
                SELECT sequence_name, sensor_name, frame_number, _viewpoint_R, _viewpoint_T
                FROM frame_annots
                WHERE sensor_name LIKE 'DSLR-%'
                  AND _viewpoint_R IS NOT NULL
                  AND _viewpoint_T IS NOT NULL
                ORDER BY sequence_name, sensor_name, frame_timestamp, frame_number
                """
            ).fetchall()
        finally:
            con.close()

        for scene, sensor_name, frame_number, blob_R, blob_T in rows:
            if blob_R is None or blob_T is None:
                continue
            try:
                vp_id = int(str(sensor_name).split("-")[-1])
                R_w2c, t_w2c = self._replay_pose_to_opencv(blob_R, blob_T)
                center = -(R_w2c.T @ t_w2c.reshape(3))
            except Exception:
                continue
            if not (np.isfinite(center).all() and np.isfinite(R_w2c).all()):
                continue
            dslr_key = f"DSLR-{vp_id}"
            rec = (
                int(frame_number),
                center.astype(np.float32).tolist(),
                R_w2c.astype(np.float32).tolist(),
            )
            pose_sequences.setdefault(str(scene), {}).setdefault(dslr_key, []).append(rec)

        for scene, scene_poses in pose_sequences.items():
            for dslr_key, seq in scene_poses.items():
                if not seq:
                    continue
                _frame_number, center, rotation = seq[0]
                centers.setdefault(scene, {})[dslr_key] = center
                rotations.setdefault(scene, {})[dslr_key] = rotation

        if not pose_sequences:
            raise RuntimeError(f"No DSLR camera poses found in Replay metadata sqlite: {sqlite_path}")

        self.camera_centers = centers
        self.camera_rotations = rotations
        self._replay_metadata_pose_sequences = pose_sequences
        print(
            f"[replay_metadata] Loaded camera poses for {len(self.camera_centers)} "
            f"scenes from {sqlite_path}"
        )

    def _get_camera_pose(self, scene_id, dslr_key, frame_id=None):
        if self.pose_source in ('replay_metadata', 'metadata_sqlite', 'replay_sqlite'):
            seq = (
                self._replay_metadata_pose_sequences
                .get(str(scene_id), {})
                .get(str(dslr_key), [])
            )
            if seq:
                idx = 0
                if frame_id is not None:
                    try:
                        frame_int = int(frame_id)
                    except Exception:
                        frame_int = 0
                    if self.replay_metadata_frame_lookup in ('frame_number', 'number', 'raw'):
                        frame_numbers = [int(item[0]) for item in seq]
                        if frame_int in frame_numbers:
                            idx = frame_numbers.index(frame_int)
                        else:
                            # Fall back to the nearest metadata frame number.
                            idx = min(
                                range(len(frame_numbers)),
                                key=lambda i: abs(frame_numbers[i] - frame_int),
                            )
                    else:
                        idx = max(0, min(int(frame_int), len(seq) - 1))
                _frame_number, center, rotation = seq[idx]
                return center, rotation
        return self.camera_centers[scene_id][dslr_key], self.camera_rotations[scene_id][dslr_key]

    def _load_gs_rotations(self):
        """
        Optional: load rotation matrices from cam_imags/gs_cameras.json.
        Assumes each entry has fields:
          - img_name like 'SC-1022_1' (scene_viewpoint)
          - rotation: 3x3 matrix (camera->world) as produced by camera_to_JSON
        """
        if not getattr(self, 'use_gs_cameras', False):
            return
        gs_file = getattr(getattr(self.cfg, 'dataset', object()), 'gs_cameras_file', 'cam_imags/gs_cameras.json')
        gs_path = os.path.join(self.dataset_path, str(gs_file))
        if not os.path.exists(gs_path):
            return
        try:
            import json
            with open(gs_path, 'r') as f:
                cams = json.load(f)
            rot_updates = {}
            for cam in cams:
                name = cam.get('img_name', '')
                if not isinstance(name, str) or '_' not in name:
                    continue
                try:
                    scene, vp = name.rsplit('_', 1)
                    vp_id = int(vp)
                except Exception:
                    continue
                key = f'DSLR-{vp_id}'
                R = cam.get('rotation', None)
                if R is None:
                    continue
                R_c2w = np.asarray(R, dtype=np.float32).reshape(3, 3)
                rot_updates.setdefault(scene, {})[key] = R_c2w.tolist()
            if rot_updates:
                # Merge: override only those rotations present in gs_cameras.
                for scene, scene_rots in rot_updates.items():
                    if scene not in self.camera_rotations or not isinstance(self.camera_rotations.get(scene), dict):
                        self.camera_rotations[scene] = {}
                    self.camera_rotations[scene].update(scene_rots)
                print(f"[gs_cameras] Overrode rotations for {len(rot_updates)} scenes from {gs_path}")
        except Exception as e:
            print(f"Warning: failed to load gs_cameras rotations ({e}); using default rotations.")

    def _load_camera_poses(self):
        """
        Load camera centers/rotations according to cfg.dataset.pose_source.
        """
        if self.pose_source in ('gs', 'gs_cameras', 'colmap'):
            self._load_gs_camera_poses()
            return
        if self.pose_source in ('replay_metadata', 'metadata_sqlite', 'replay_sqlite'):
            self._load_replay_metadata_camera_poses()
            return

        # Default: fixed_rotation file
        self._load_camera_positions()
        # Backward-compat: allow overriding rotations from gs_cameras.
        if self.use_gs_cameras:
            print(
                "[Audio3DGSReplayNVASViewpointDataset] use_gs_cameras=True: overriding rotations only. "
                "Note: this assumes fixed_pose and gs_cameras share the same world frame."
            )
            self._load_gs_rotations()
        
    def _load_dataset(self):
        """
        Load audio data and camera poses with viewpoint-based splitting.
        When cfg.dataset.use_metadata=True, restrict frame list to metadata_v2.json entries
        """
        
        # Process audio data
        self.binaural_audio = []   # Target binaural audio
        self.source_audio = []     # Source (near or viewpoint) audio
        self.poses = []            # Target camera poses
        self.input_poses = []      # Input camera poses (if input_source == 'viewpoint')
        self.input_vps = []        # Input viewpoint id per sample (0 when 'near')
        self.scene_ids = []        # Scene identifiers for debugging
        
        # Define which viewpoints to use based on split (allow explicit train viewpoint selection)
        train_vps_cfg = getattr(getattr(self.cfg, 'dataset', object()), 'train_viewpoints', None)
        parsed_train_vps = None
        if isinstance(train_vps_cfg, (list, tuple)):
            try:
                parsed_train_vps = [int(v) for v in train_vps_cfg]
            except Exception:
                parsed_train_vps = None
        elif isinstance(train_vps_cfg, str) and train_vps_cfg.strip():
            toks = [t.strip() for part in train_vps_cfg.split(',') for t in part.split() if t.strip()]
            pv = []
            for t in toks:
                try:
                    pv.append(int(t))
                except Exception:
                    pass
            if pv:
                parsed_train_vps = pv

        if self.split == 'train':
            if parsed_train_vps:
                # Remove test viewpoint if accidentally included
                # train_ids = [v for v in parsed_train_vps if v != int(self.test_viewpoint)]
                train_ids = [v for v in parsed_train_vps]
                # Keep in [1..8]
                train_ids = [v for v in train_ids if 1 <= int(v) <= 8]
                available_viewpoints = [str(v) for v in sorted(set(train_ids))]
            else:
                # Default: 7 viewpoints for training (exclude test)
                available_viewpoints = [str(i) for i in range(1, 9) if i != self.test_viewpoint]
        else:  # test or val
            # Use only the test viewpoint
            available_viewpoints = [str(self.test_viewpoint)]
            
        print(f"Using viewpoints {available_viewpoints} for {self.split} split")
        
        # Build per-scene frame lists, either from filesystem or from metadata
        scene_to_frames = {}
        if self.use_metadata:
            # Load metadata_v2.json and filter entries by selected scenes
            meta_path = os.path.join(self.dataset_path, 'v3', self.metadata_file)
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Warning: cannot read {meta_path} ({e}); falling back to filesystem scan.")
                self.use_metadata = False
            if self.use_metadata:
                tmp = {sid: [] for sid in self.selected_scenes}
                for k in metadata.keys():
                    parts = k.strip('/').split('/')
                    if len(parts) < 2:
                        continue
                    scene_id = parts[-2]
                    frame_id = parts[-1]
                    if scene_id in tmp and frame_id.isdigit():
                        tmp[scene_id].append(int(frame_id))
                # Apply stride
                for sid, frames in tmp.items():
                    frames = sorted(frames)
                    stride = max(1, int(self.frame_stride_train)) if self.split == 'train' else max(1, int(self.frame_stride_test))
                    frames = frames[::stride]
                    scene_to_frames[sid] = [str(f) for f in frames]
        if not self.use_metadata:
            # Fallback: scan filesystem
            for scene_id in self.selected_scenes:
                scene_path = os.path.join(self.dataset_path, 'v3', scene_id)
                if not os.path.exists(scene_path):
                    print(f"Skipping scene {scene_id}: directory not found")
                    continue
                frame_dirs = [d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d)) and d.isdigit()]
                frame_dirs = sorted(frame_dirs, key=int)
                stride = max(1, int(self.frame_stride_train)) if self.split == 'train' else max(1, int(self.frame_stride_test))
                scene_to_frames[scene_id] = frame_dirs[::stride]

        # Optional: restrict to a single clip/frame (per-clip training)
        # Priority: cfg.dataset.frame_scope > env:A3DGS_FRAME_ID
        frame_scope = getattr(getattr(self.cfg, 'dataset', object()), 'frame_scope', None)
        if not frame_scope:
            frame_scope = os.environ.get('A3DGS_FRAME_ID', '').strip()
        if frame_scope:
            try:
                # Normalize to string of integer frame id
                frame_scope = str(int(frame_scope))
            except Exception:
                frame_scope = None
        if frame_scope:
            print(f"[Audio3DGSReplayNVASViewpointDataset] Restricting to single frame_id={frame_scope}")
            for sid in list(scene_to_frames.keys()):
                frames = scene_to_frames.get(sid, [])
                scene_to_frames[sid] = [f for f in frames if f == frame_scope]

        for scene_id in self.selected_scenes:
            # Skip scenes with invalid camera positions
            if scene_id not in self.camera_centers:
                print(f"Skipping scene {scene_id}: no camera positions")
                continue

            # Check valid viewpoints for this scene
            valid_viewpoints = []
            for vp in available_viewpoints:
                dslr_key = f'DSLR-{vp}'
                if (dslr_key in self.camera_centers[scene_id] and
                    dslr_key in self.camera_rotations[scene_id] and
                    not np.isnan(np.asarray(self.camera_centers[scene_id][dslr_key])).any() and
                    not np.isnan(np.asarray(self.camera_rotations[scene_id][dslr_key])).any()):
                    valid_viewpoints.append(vp)

            if not valid_viewpoints:
                print(f"Skipping scene {scene_id}: no valid viewpoints")
                continue

            scene_path = os.path.join(self.dataset_path, 'v3', scene_id)
            sampled_frames = scene_to_frames.get(scene_id, [])
            
            print(f"Processing scene {scene_id}: {len(sampled_frames)} frames, viewpoints {valid_viewpoints}")
            
            for frame_id in sampled_frames:
                frame_path = os.path.join(scene_path, frame_id)
                
                # Check if near.wav exists when needed
                near_wav_path = os.path.join(frame_path, 'near.wav')
                if self.input_source == 'near' and not os.path.exists(near_wav_path):
                    continue

                # Input source policy
                # If input_source == 'viewpoint' and input_viewpoint <= 0, enable self-viewpoint mode:
                # use each target viewpoint's audio as the input for that sample.
                src_audio_base = None
                input_cam_pose_base = None
                input_vp_base = 0
                if self.input_source == 'viewpoint':
                    ivp_cfg = int(self.input_viewpoint)
                    if 1 <= ivp_cfg <= 8:
                        # Fixed input viewpoint across all targets (legacy behavior)
                        ivp_key = f'DSLR-{ivp_cfg}'
                        if (ivp_key not in self.camera_centers[scene_id]) or (ivp_key not in self.camera_rotations[scene_id]):
                            continue
                        input_binaural_path = os.path.join(frame_path, f'{ivp_cfg}.wav')
                        if not os.path.exists(input_binaural_path):
                            continue
                        try:
                            src_audio_base, _ = librosa.load(input_binaural_path, sr=self.sampling_rate, mono=False)
                            if src_audio_base.ndim == 1:
                                src_audio_base = np.stack([src_audio_base, src_audio_base])
                            elif src_audio_base.shape[0] == 1:
                                src_audio_base = np.vstack([src_audio_base, src_audio_base])
                        except Exception:
                            continue
                        icenter, irot = self._get_camera_pose(scene_id, ivp_key, frame_id)
                        irot_m = self._to_rotation_matrix(irot)
                        input_cam_pose_base = np.concatenate([icenter, irot_m.flatten()])
                        input_vp_base = ivp_cfg
                    else:
                        # When ivp_cfg<=0, decide policy based on split and random flags
                        # For training with random_input_viewpoint=True: choose ONE fixed random
                        # input viewpoint per frame (from training viewpoints), and reuse it
                        # for all target viewpoints in this frame.
                        if self.split == 'train' and self.random_input_viewpoint:
                            # Build candidate list from training viewpoints for this scene/frame
                            try:
                                cand = [int(v) for v in valid_viewpoints]
                                random.shuffle(cand)
                            except Exception:
                                cand = []
                            chosen = None
                            for c in cand:
                                ivk = f'DSLR-{int(c)}'
                                p = os.path.join(frame_path, f'{int(c)}.wav')
                                if (ivk in self.camera_centers.get(scene_id, {})) and (ivk in self.camera_rotations.get(scene_id, {})) and os.path.exists(p):
                                    chosen = int(c)
                                    break
                            if chosen is not None:
                                try:
                                    input_binaural_path = os.path.join(frame_path, f'{chosen}.wav')
                                    src_audio_base, _ = librosa.load(input_binaural_path, sr=self.sampling_rate, mono=False)
                                    if src_audio_base.ndim == 1:
                                        src_audio_base = np.stack([src_audio_base, src_audio_base])
                                    elif src_audio_base.shape[0] == 1:
                                        src_audio_base = np.vstack([src_audio_base, src_audio_base])
                                except Exception:
                                    src_audio_base = None
                                if src_audio_base is not None:
                                    ivp_key = f'DSLR-{chosen}'
                                    icenter, irot = self._get_camera_pose(scene_id, ivp_key, frame_id)
                                    irot_m = self._to_rotation_matrix(irot)
                                    input_cam_pose_base = np.concatenate([icenter, irot_m.flatten()])
                                    input_vp_base = int(chosen)
                        # Else (val/test or no random), will decide per-target below
                        pass
                else:
                    # Load near.wav as source (default)
                    try:
                        src_audio_base, _ = librosa.load(near_wav_path, sr=self.sampling_rate, mono=False)
                        if src_audio_base.ndim == 1:
                            src_audio_base = np.stack([src_audio_base, src_audio_base])
                    except Exception:
                        continue
                
                # Process each valid viewpoint
                for viewpoint in valid_viewpoints:
                    # Load binaural audio from this viewpoint
                    binaural_wav_path = os.path.join(frame_path, f'{viewpoint}.wav')
                    
                    if not os.path.exists(binaural_wav_path):
                        continue
                        
                    try:
                        binaural_audio, _ = librosa.load(binaural_wav_path, sr=self.sampling_rate, mono=False)
                        if binaural_audio.ndim == 1:
                            binaural_audio = np.stack([binaural_audio, binaural_audio])
                        elif binaural_audio.shape[0] == 1:
                            binaural_audio = np.vstack([binaural_audio, binaural_audio])
                    except Exception as e:
                        continue
                    
                    # Get camera pose
                    dslr_key = f'DSLR-{viewpoint}'
                    camera_center, camera_rotation = self._get_camera_pose(scene_id, dslr_key, frame_id)
                    
                    # Create rotation matrix and pose for TARGET viewpoint
                    rotation_matrix = self._to_rotation_matrix(camera_rotation)
                    cam_pose = np.concatenate([camera_center, rotation_matrix.flatten()])
                    
                    # Decide input source for THIS sample
                    if self.input_source == 'viewpoint':
                        ivp_cfg = int(self.input_viewpoint)
                        if (1 <= ivp_cfg <= 8 and src_audio_base is not None) or \
                           (src_audio_base is not None and input_vp_base > 0 and self.split == 'train' and self.random_input_viewpoint):
                            # fixed input viewpoint across samples
                            src_audio_cur = src_audio_base
                            input_cam_pose = input_cam_pose_base
                            input_vp_used = input_vp_base
                        else:
                            # Per-sample decision: random-from-train (if enabled), else self-viewpoint
                            ivp_cur = int(viewpoint)
                            use_random = (
                                (self.split == 'train' and self.random_input_viewpoint) or
                                (self.split != 'train' and self.random_input_viewpoint_eval)
                            )
                            if use_random:
                                # Build candidate list excluding current target viewpoint
                                cand = []
                                try:
                                    if self.split == 'train':
                                        # valid_viewpoints equals training viewpoints for train split
                                        cand = [int(v) for v in valid_viewpoints if int(v) != int(viewpoint)]
                                    else:
                                        # For val/test, prefer explicit train_viewpoints if provided; else 1..8
                                        tvps = None
                                        if parsed_train_vps:
                                            tvps = [int(v) for v in parsed_train_vps]
                                        else:
                                            tvps = list(range(1, 9))
                                        cand = [int(v) for v in tvps if int(v) != int(viewpoint)]
                                    random.shuffle(cand)
                                except Exception:
                                    cand = []
                                chosen = None
                                for c in cand:
                                    ivk = f'DSLR-{int(c)}'
                                    p = os.path.join(frame_path, f'{int(c)}.wav')
                                    if (ivk in self.camera_centers.get(scene_id, {})) and (ivk in self.camera_rotations.get(scene_id, {})) and os.path.exists(p):
                                        chosen = int(c)
                                        break
                                if chosen is not None:
                                    ivp_cur = int(chosen)
                            ivp_key = f'DSLR-{ivp_cur}'
                            input_binaural_path = os.path.join(frame_path, f'{ivp_cur}.wav')
                            if not os.path.exists(input_binaural_path):
                                continue
                            try:
                                src_audio_cur, _ = librosa.load(input_binaural_path, sr=self.sampling_rate, mono=False)
                                if src_audio_cur.ndim == 1:
                                    src_audio_cur = np.stack([src_audio_cur, src_audio_cur])
                                elif src_audio_cur.shape[0] == 1:
                                    src_audio_cur = np.vstack([src_audio_cur, src_audio_cur])
                            except Exception:
                                continue
                            icenter, irot = self._get_camera_pose(scene_id, ivp_key, frame_id)
                            irot_m = self._to_rotation_matrix(irot)
                            input_cam_pose = np.concatenate([icenter, irot_m.flatten()])
                            input_vp_used = ivp_cur
                    else:
                        # near mode
                        src_audio_cur = src_audio_base
                        input_cam_pose = np.zeros_like(cam_pose, dtype=np.float32)
                        input_vp_used = 0
                    
                    # Ensure audio has target length (e.g., 2-3 seconds as in config)
                    target_samples = int(self.audio_len * self.sampling_rate)
                    
                    # Process both audio tracks to same length
                    def process_audio(audio, target_len):
                        if audio.shape[1] > target_len:
                            if self.split == 'train' and self.train_random_crop:
                                start_idx = random.randint(0, audio.shape[1] - target_len)
                            else:
                                start_idx = (audio.shape[1] - target_len) // 2
                            return audio[:, start_idx:start_idx + target_len]
                        elif audio.shape[1] < target_len:
                            pad_length = target_len - audio.shape[1]
                            return np.pad(audio, ((0, 0), (0, pad_length)))
                        else:
                            return audio
                    
                    binaural_audio = process_audio(binaural_audio, target_samples)
                    src_audio_cur = process_audio(src_audio_cur, target_samples)
                    # Keep a copy of raw (unfiltered) target for env reconstruction
                    binaural_audio_raw = np.array(binaural_audio, copy=True)

                    # Optional bandpass filtering (apply to cropped/padded audio)
                    if self.bp_enable:
                        try:
                            high_hz = self.sampling_rate // 2 - 1 if self.bp_high <= 0 else float(self.bp_high)
                            order = int(getattr(getattr(self.cfg.dataset, 'bandpass', None), 'order', 5))
                            src_audio_cur = self._butter_bandpass_filter(src_audio_cur, self.bp_low, high_hz, self.sampling_rate, order=order)
                            binaural_audio_f = self._butter_bandpass_filter(binaural_audio, self.bp_low, high_hz, self.sampling_rate, order=order)
                            src_audio_cur = np.nan_to_num(src_audio_cur, copy=False).astype(np.float32)
                            binaural_audio_f = np.nan_to_num(binaural_audio_f, copy=False).astype(np.float32)
                        except Exception as e:
                            print(f"Bandpass failed for {scene_id}/{frame_id}/vp{viewpoint}: {e}")
                            binaural_audio_f = binaural_audio.astype(np.float32)
                        # Save filtered as target; compute environment residual = raw - filtered
                        env_res = (binaural_audio_raw - binaural_audio_f).astype(np.float32)
                        binaural_audio = binaural_audio_f
                    else:
                        # No filtering: residual is zero
                        env_res = np.zeros_like(binaural_audio_raw, dtype=np.float32)
                    
                    # Store data
                    self.binaural_audio.append(binaural_audio)
                    self.source_audio.append(src_audio_cur)
                    self.poses.append(cam_pose)
                    # Input pose/meta if using viewpoint input; else zeros
                    if input_cam_pose is None:
                        self.input_poses.append(np.zeros_like(cam_pose, dtype=np.float32))
                        self.input_vps.append(int(0))
                    else:
                        self.input_poses.append(input_cam_pose.astype(np.float32))
                        self.input_vps.append(int(input_vp_used))
                    self.scene_ids.append(f"{scene_id}_{frame_id}_vp{viewpoint}")
                    # Record raw target and environment residual
                    self._binaural_audio_raw.append(binaural_audio_raw.astype(np.float32))
                    self._env_residuals.append(env_res)
                    # Record per-sample metadata for runtime re-randomization
                    self._sample_frame_paths.append(frame_path)
                    try:
                        self._sample_target_vps.append(int(viewpoint))
                    except Exception:
                        self._sample_target_vps.append(int(str(viewpoint)))
        
        print(f"Loaded {len(self.binaural_audio)} audio samples for {self.split} set")
        
        if len(self.binaural_audio) == 0:
            raise RuntimeError(f"No valid samples found for {self.split} split")
            
    def _apply_scene_normalization(self):
        """Apply per-scene RMS max normalization (as described in the paper)."""
        # Build per-sample scene id list (SC-xxxx)
        self._base_scene_ids = [sid.split('_')[0] for sid in self.scene_ids]
        
        # Compute per-scene max RMS on target binaural (use both channels)
        scene_to_max = {}
        for audio, base in zip(self.binaural_audio, self._base_scene_ids):
            if not np.isfinite(audio).all():
                continue
            rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
            if base not in scene_to_max:
                scene_to_max[base] = rms
            else:
                scene_to_max[base] = max(scene_to_max[base], rms)
        
        # Normalize both binaural and source audio by their scene's max RMS (safe)
        new_binaural = []
        new_source = []
        self._norm_factors = []
        eps = 1e-6
        for audio_bi, audio_src, base in zip(self.binaural_audio, self.source_audio, self._base_scene_ids):
            max_rms = scene_to_max.get(base, 1.0)
            if (not np.isfinite(max_rms)) or (max_rms < eps):
                new_binaural.append(audio_bi.astype(np.float32))
                new_source.append(audio_src.astype(np.float32))
                self._norm_factors.append(1.0)
            else:
                new_binaural.append((audio_bi / max_rms).astype(np.float32))
                new_source.append((audio_src / max_rms).astype(np.float32))
                self._norm_factors.append(float(max_rms))
        
        self.binaural_audio = new_binaural
        self.source_audio = new_source
        self._norm_scene_to_max = scene_to_max
        
        # Simple report
        factors = [scene_to_max[b] for b in sorted(scene_to_max.keys())]
        if len(factors) > 0:
            print(f"Applied scene-level normalization for {len(scene_to_max)} scenes. Avg max RMS: {np.mean(factors):.4f}")
    
    @staticmethod
    def _xyz_to_angle(xyz):
        """Match NVAS conversion: 3D direction vector -> angle triplet."""
        x, y, z = xyz
        return np.array([
            np.arctan2(y, x),
            np.arctan2(z, y),
            np.arctan2(z, x)
        ])
            
    def _euler_to_rotation_matrix(self, euler_angles):
        """Convert Euler angles to rotation matrix"""
        rx, ry, rz = euler_angles
        
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        
        R = Rz @ Ry @ Rx
        return R

    @staticmethod
    def _lookat_w2c_from_forward(
        forward_world: np.ndarray,
        world_up: np.ndarray = None,
        yaw_only: bool = False,
    ) -> np.ndarray:
        """
        Build a world->camera rotation (W2C) from a forward direction in world coordinates.

        ReplayNVAS fixed_rotation provides only a forward/look direction vector in world.
        We need a full 3x3 rotation for Audio3DGS, and the model expects poses to carry
        a W2C rotation matrix (consistent with `gs_cameras.json` exported by
        `libs/utils/camera_utils.py:camera_to_JSON`).

        Convention used by Audio3DGS:
          - camera +x : right
          - camera +y : up
          - camera -z : forward

        We first construct a C2W basis with columns [right, up, backward] where
        backward = -forward, then transpose to obtain W2C.
        """
        f = np.asarray(forward_world, dtype=np.float32).reshape(-1)
        if f.shape[0] != 3:
            raise ValueError(f"forward_world must be 3D, got shape {f.shape}")
        fn = float(np.linalg.norm(f))
        if fn < 1e-8 or (not np.isfinite(fn)):
            raise ValueError("forward_world has near-zero norm")
        f = f / fn

        if world_up is None:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            up = np.asarray(world_up, dtype=np.float32).reshape(-1)
            if up.shape[0] != 3 or (not np.isfinite(up).all()):
                up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        upn = float(np.linalg.norm(up))
        if upn < 1e-8 or (not np.isfinite(upn)):
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            upn = float(np.linalg.norm(up))
        up = up / (upn + 1e-8)

        # Optional: ignore pitch/roll by projecting forward onto the horizontal plane.
        if yaw_only:
            f_h = f - float(np.dot(f, up)) * up
            hn = float(np.linalg.norm(f_h))
            if hn >= 1e-6 and np.isfinite(hn):
                f = f_h / hn

        # If nearly colinear, pick an alternate up.
        if abs(float(np.dot(f, up))) > 0.95:
            # Choose the canonical axis least aligned with f.
            cand = [
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
            ]
            cand = sorted(cand, key=lambda a: abs(float(np.dot(f, a))))
            up = cand[0]

        # Build camera +z axis in world (camera forward is -z).
        z_world = -f
        # Right-handed basis: right = up x z
        right = np.cross(up, z_world)
        rn = float(np.linalg.norm(right) + 1e-8)
        right = right / rn
        up2 = np.cross(z_world, right)
        un = float(np.linalg.norm(up2) + 1e-8)
        up2 = up2 / un

        R_c2w = np.stack([right, up2, z_world], axis=1).astype(np.float32)  # [3,3]
        return R_c2w.T  # [3,3] world->camera

    def _to_rotation_matrix(self, rot):
        """Accept either 3x3 matrix, flattened 9, or 3-element direction vector."""
        arr = np.asarray(rot, dtype=np.float32)
        if arr.shape == (3, 3):
            return arr
        arr = arr.flatten()
        if arr.shape[0] == 9:
            return arr.reshape(3, 3)
        if arr.shape[0] == 3:
            # ReplayNVAS fixed_rotation stores a 3D orientation vector used by NVAS/ViGAS.
            # Instead of treating derived angles as Euler rotations, interpret the vector
            # directly as a world-space forward direction and build a look-at W2C matrix.
            mode = str(getattr(self, "fixed_rotation_mode", "lookat") or "lookat").lower()
            yaw_only = mode in ("yaw", "yaw_only", "azimuth", "azimuth_only")
            return self._lookat_w2c_from_forward(
                arr,
                world_up=getattr(self, "fixed_rotation_world_up", None),
                yaw_only=yaw_only,
            )
        raise ValueError(f"Unsupported rotation format: shape {arr.shape}")
    
    def __len__(self):
        return len(self.binaural_audio)
        
    def __getitem__(self, idx):
        """Get a single training sample"""
        
        # Get audio data
        binaural = torch.from_numpy(self.binaural_audio[idx]).float()
        # Raw target and env residual for optional env add-back during eval
        binaural_raw = torch.from_numpy(self._binaural_audio_raw[idx]).float() if idx < len(self._binaural_audio_raw) else binaural.clone()
        env_residual = torch.from_numpy(self._env_residuals[idx]).float() if idx < len(self._env_residuals) else torch.zeros_like(binaural)
        source = torch.from_numpy(self.source_audio[idx]).float()
        pose = torch.from_numpy(self.poses[idx]).float()
        input_pose = torch.from_numpy(self.input_poses[idx]).float() if len(self.input_poses) == len(self.poses) else torch.zeros_like(pose)
        
        # Ensure consistent length for batching
        target_samples = int(self.audio_len * self.sampling_rate)
        
        def ensure_length(audio_tensor):
            if audio_tensor.shape[-1] > target_samples:
                if self.split == 'train' and self.train_random_crop:
                    start_idx = random.randint(0, audio_tensor.shape[-1] - target_samples)
                else:
                    start_idx = (audio_tensor.shape[-1] - target_samples) // 2
                return audio_tensor[..., start_idx:start_idx + target_samples]
            elif audio_tensor.shape[-1] < target_samples:
                pad_length = target_samples - audio_tensor.shape[-1]
                return torch.cat([audio_tensor, torch.zeros(*audio_tensor.shape[:-1], pad_length)], dim=-1)
            else:
                return audio_tensor
        
        # Optional per-sample runtime randomization of input viewpoint
        # Applies when using viewpoint input with input_viewpoint<=0 and the
        # corresponding random flags are enabled.
        input_vp_runtime = int(self.input_vps[idx]) if len(self.input_vps) == len(self.poses) else int(0)
        try:
            need_runtime_rand = False
            if self.input_source == 'viewpoint' and int(getattr(self, 'input_viewpoint', 0) or 0) <= 0:
                if self.split == 'train' and bool(self.random_input_viewpoint):
                    need_runtime_rand = True
                elif self.split != 'train' and bool(self.random_input_viewpoint_eval):
                    need_runtime_rand = True
            if need_runtime_rand:
                frame_path = self._sample_frame_paths[idx] if idx < len(self._sample_frame_paths) else None
                tgt_vp = self._sample_target_vps[idx] if idx < len(self._sample_target_vps) else None
                if frame_path is not None and tgt_vp is not None:
                    # Build candidate list
                    if self.split == 'train':
                        # default training candidate viewpoints: 1..8 except test_viewpoint and target
                        cand = [v for v in range(1, 9) if v != int(self.test_viewpoint) and v != int(tgt_vp)]
                    else:
                        cand = [v for v in range(1, 9) if v != int(tgt_vp)]
                    random.shuffle(cand)
                    chosen = None
                    # parse scene id from frame_path
                    try:
                        scene_id = os.path.basename(os.path.dirname(os.path.dirname(frame_path)))
                    except Exception:
                        scene_id = None
                    try:
                        frame_id = os.path.basename(frame_path)
                    except Exception:
                        frame_id = None
                    for c in cand:
                        ivk = f'DSLR-{int(c)}'
                        p = os.path.join(frame_path, f'{int(c)}.wav')
                        if scene_id and (ivk in self.camera_centers.get(scene_id, {})) and (ivk in self.camera_rotations.get(scene_id, {})) and os.path.exists(p):
                            chosen = int(c)
                            # Load source audio for chosen viewpoint
                            try:
                                src_np, _ = librosa.load(p, sr=self.sampling_rate, mono=False)
                                if src_np.ndim == 1:
                                    src_np = np.stack([src_np, src_np])
                                elif src_np.shape[0] == 1:
                                    src_np = np.vstack([src_np, src_np])
                                source = torch.from_numpy(src_np).float()
                                # Build input pose for chosen
                                icenter, irot = self._get_camera_pose(scene_id, ivk, frame_id)
                                irot_m = self._to_rotation_matrix(irot)
                                input_pose_np = np.concatenate([icenter, irot_m.flatten()]).astype(np.float32)
                                input_pose = torch.from_numpy(input_pose_np).float()
                                input_vp_runtime = int(chosen)
                                break
                            except Exception:
                                continue
        except Exception:
            pass

        binaural = ensure_length(binaural)
        binaural_raw = ensure_length(binaural_raw)
        env_residual = ensure_length(env_residual)
        source = ensure_length(source)
        # Apply bandpass to runtime-randomized source to match dataset preprocessing
        if self.bp_enable:
            try:
                high_hz = self.sampling_rate // 2 - 1 if self.bp_high <= 0 else float(self.bp_high)
                order = int(getattr(getattr(self.cfg.dataset, 'bandpass', None), 'order', 5))
                # Convert to numpy for filtering, then back to torch
                src_np = source.detach().cpu().numpy()
                src_np = self._butter_bandpass_filter(src_np, self.bp_low, high_hz, self.sampling_rate, order=order)
                src_np = np.nan_to_num(src_np, copy=False).astype(np.float32)
                source = torch.from_numpy(src_np)
            except Exception:
                pass

        return {
            'cam_pose': pose,
            'input_cam_pose': input_pose,
            'source_audio': source,
            'target_binaural': binaural,
            'target_binaural_raw': binaural_raw,
            'env_residual': env_residual,
            'idx': idx,
            'scene_id': self.scene_ids[idx],
            'norm_factor': float(self._norm_factors[idx]) if self.scene_level_normalize else 1.0,
            'input_viewpoint': int(input_vp_runtime)
        }


def make_viewpoint_dataset(cfg, split='train', test_viewpoint=8, selected_scenes=None):
    """Create viewpoint-based dataset instance (paper style)"""
    return Audio3DGSReplayNVASViewpointDataset(cfg, split=split, 
                                              test_viewpoint=test_viewpoint,
                                              selected_scenes=selected_scenes)


def make_viewpoint_data_loader(cfg, split='train', test_viewpoint=8, selected_scenes=None, distributed=False):
    """Create viewpoint-based data loader (paper style)"""
    dataset = make_viewpoint_dataset(cfg, split, test_viewpoint, selected_scenes)
    
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = (split == 'train')

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size if split == 'train' else 1,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True,
        drop_last=True if split == 'train' else False
    )

    return data_loader
