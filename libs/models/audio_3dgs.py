"""
Audio 3D Gaussian Splatting Implementation
Based on "Extending Gaussian Splatting to Audio: Optimizing Audio Points for Novel-view Acoustic Synthesis"

Adapting AV-Cloud codebase to implement pure audio-based 3DGS without visual dependency.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.models.networks.encoder import embedding_module_log
from libs.models.networks.mlp import basic_project2
from libs.utils.sh_utils import eval_sh


class DualBranchAudioUNet(nn.Module):
    """Dual-branch Audio U-Net renderer.

    Inputs per branch:
    - Mono branch: [source_mag, SH_value, inv_distance] -> 3 channels
    - Diff branch: [SH_value] -> 1 channel (force directional learning)
    """
    
    def __init__(self, use_groupnorm: bool = True, diff_in_channels: int = 1):
        super().__init__()
        self.use_groupnorm = use_groupnorm
        self.diff_in_channels = int(diff_in_channels)
        
        # Shared encoder layers
        # Mono branch now takes 3-channel input: [source_mag, SH_value, inv_distance]
        self.enc1 = self._make_layer(3, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)
        
        # Separate encoder for diff branch (1 or more channels):
        # default: [SH_value] (1 ch); with stereo cues: [SH_value, ILD] (2 ch)
        self.diff_enc1 = self._make_layer(self.diff_in_channels, 64)
        
        self.maxpool = nn.MaxPool2d(2)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._make_layer(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._make_layer(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._make_layer(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = self._make_layer(128, 64)
        
        # Output heads
        self.out_mono = nn.Conv2d(64, 1, 1)
        self.out_diff = nn.Conv2d(64, 1, 1)
    
    def _norm(self, num_channels):
        if self.use_groupnorm:
            # Use up to 8 groups but not exceeding channels
            num_groups = 8 if num_channels >= 8 else 1
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        else:
            return nn.BatchNorm2d(num_channels)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            self._norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            self._norm(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, mono_features, diff_features):
        # Process mono branch (source_mag + SH + inv_distance)
        e1_mono = self.enc1(mono_features)
        e1_pool_mono = self.maxpool(e1_mono)
        
        # Process diff branch (source_mag + SH)
        e1_diff = self.diff_enc1(diff_features)
        e1_pool_diff = self.maxpool(e1_diff)
        
        # Combine features from both branches
        e1_combined = e1_mono  # Use mono for skip connections
        e1_pool_combined = (e1_pool_mono + e1_pool_diff) / 2
        
        # Shared encoder path
        e2 = self.enc2(e1_pool_combined)
        e2_pool = self.maxpool(e2)
        e3 = self.enc3(e2_pool)
        e3_pool = self.maxpool(e3)
        e4 = self.enc4(e3_pool)
        
        # Decoder path
        d4_up = self.upconv4(e4)
        if d4_up.shape[-2:] != e3.shape[-2:]:
            d4_up = F.interpolate(d4_up, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4_up, e3], dim=1)
        d4 = self.dec4(d4)
        
        d3_up = self.upconv3(d4)
        if d3_up.shape[-2:] != e2.shape[-2:]:
            d3_up = F.interpolate(d3_up, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3_up, e2], dim=1)
        d3 = self.dec3(d3)
        
        d2_up = self.upconv2(d3)
        if d2_up.shape[-2:] != e1_combined.shape[-2:]:
            d2_up = F.interpolate(d2_up, size=e1_combined.shape[-2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2_up, e1_combined], dim=1)
        d2 = self.dec2(d2)
        
        d1_up = self.upconv1(d2)
        if d1_up.shape[-2:] != mono_features.shape[-2:]:
            d1_up = F.interpolate(d1_up, size=mono_features.shape[-2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1_up, e1_combined], dim=1)
        d1 = self.dec1(d1)
        
        # Output predictions
        # Mono mask non-negative; Diff mask can be signed but bounded
        mono_mask = F.softplus(self.out_mono(d1)) + 0.1
        diff_mask = torch.tanh(self.out_diff(d1))
        
        return mono_mask, diff_mask


class AudioUNet(nn.Module):
    """U-Net renderer for binaural audio synthesis"""
    
    def __init__(self, in_channels=2, out_channels=2):
        super(AudioUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._double_conv(in_channels, 64)
        self.enc2 = self._double_conv(64, 128)
        self.enc3 = self._double_conv(128, 256)
        self.enc4 = self._double_conv(256, 512)
        
        # Bottleneck
        self.bottleneck = self._double_conv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._double_conv(128, 64)
        
        # Output layers - predict masks for mono and diff signals
        self.out_mono = nn.Conv2d(64, 1, 1)
        self.out_diff = nn.Conv2d(64, 1, 1)
        
        self.pool = nn.MaxPool2d(2)
        
    def _double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections (handle size mismatches)
        d4_up = self.upconv4(b)
        # Crop or pad to match e4 size
        if d4_up.shape[-2:] != e4.shape[-2:]:
            d4_up = F.interpolate(d4_up, size=e4.shape[-2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4_up, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3_up = self.upconv3(d4)
        if d3_up.shape[-2:] != e3.shape[-2:]:
            d3_up = F.interpolate(d3_up, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3_up, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2_up = self.upconv2(d3)
        if d2_up.shape[-2:] != e2.shape[-2:]:
            d2_up = F.interpolate(d2_up, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2_up, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1_up = self.upconv1(d2)
        if d1_up.shape[-2:] != e1.shape[-2:]:
            d1_up = F.interpolate(d1_up, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1_up, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output masks
        # Use proper activations for audio masking
        # Mono mask: allow amplification, use softplus + small constant
        mono_mask = F.softplus(self.out_mono(d1)) + 0.1
        # Diff mask: constrain to reasonable range to prevent severe imbalance
        diff_mask = torch.tanh(self.out_diff(d1))
        
        return mono_mask, diff_mask


class Audio3DGS(nn.Module):
    """Audio 3D Gaussian Splatting - Pure audio approach without visual dependency"""
    
    def __init__(self, cfg, freq_num=257, time_num=348):
        super(Audio3DGS, self).__init__()
        
        self.freq_num = freq_num
        self.time_num = time_num 
        self.n_points = freq_num * time_num  # Each spectrogram pixel = one 3D point
        
        # SH degree and number of coefficients (default to 2 -> 9 coeffs)
        self.sh_degree = int(getattr(cfg.model, 'sh_degree', 2))
        self.num_sh_coeffs = (self.sh_degree + 1) ** 2
        
        # Audio point parameters (learnable)
        # Position in 3D space for each spectrogram pixel
        self._xyz = nn.Parameter(torch.randn(self.n_points, 3) * 0.1)
        
        # Spherical harmonics coefficients (0..sh_degree)
        # Shape: [n_points, 1, (sh_degree+1)^2] - 1 channel of SH coefficients per point
        # Initialize to zeros; optionally randomize higher orders to break symmetry
        self._sh_coeffs = nn.Parameter(torch.zeros(self.n_points, 1, self.num_sh_coeffs))
        # Optional: random init for higher-order SH to avoid symmetry (configurable)
        sh_rand_std = float(getattr(cfg.model, 'sh_rand_init_std', 0.0))
        if sh_rand_std > 0:
            with torch.no_grad():
                if self.num_sh_coeffs > 1:
                    self._sh_coeffs[:, 0, 1:] = sh_rand_std * torch.randn(self.n_points, self.num_sh_coeffs - 1)
        
        # Rotation quaternion for each point (w, x, y, z).
        # Initialize with identity quaternion [1, 0, 0, 0] plus small random perturbation.
        rotation_init = torch.zeros(self.n_points, 4)
        rotation_init[:, 0] = 1.0  # w component = 1 for identity
        rotation_init[:, 1:] = 0.01 * torch.randn(self.n_points, 3)  # Small random perturbation
        self._rotation = nn.Parameter(rotation_init)
        
        # Time-frequency coordinate mapping (fixed)
        self.register_buffer('tf_coords', self._create_tf_grid())
        
        # Whether to incorporate camera orientation when forming SH directions
        self.use_cam_rotation = bool(getattr(getattr(cfg, 'model', object()), 'use_cam_rotation', False))
        # Whether to rotate viewing directions into each point's local frame before SH eval.
        # When False, SH is evaluated directly in world/camera frame (closer to vanilla 3DGS).
        self.use_point_rotation = bool(getattr(getattr(cfg, 'model', object()), 'use_point_rotation', True))
        # Whether to inject stereo cues (ILD) into diff branch
        self.use_stereo_cues = bool(getattr(getattr(cfg, 'model', object()), 'use_stereo_cues', False))
        # Optional: also feed inv_distance and/or side magnitude into diff branch
        self.diff_use_inv_distance = bool(getattr(getattr(cfg, 'model', object()), 'diff_use_inv_distance', False))
        self.diff_use_side_mag = bool(getattr(getattr(cfg, 'model', object()), 'diff_use_side_mag', False))

        # Dual-branch rendering network
        use_groupnorm = bool(getattr(cfg.model, 'use_groupnorm', True))
        diff_in_channels = 1
        if self.use_stereo_cues:
            diff_in_channels += 1  # ILD
        if self.diff_use_inv_distance:
            diff_in_channels += 1
        if self.diff_use_side_mag:
            diff_in_channels += 1
        self.renderer = DualBranchAudioUNet(use_groupnorm=use_groupnorm, diff_in_channels=diff_in_channels)

        # Normalization factor for relative positions
        self.max_norm = 310.0

    # -----------------------------
    # Numeric safety helpers
    # -----------------------------
    def _safe_sh_coeffs(self):
        """Return a sanitized copy of SH coefficients to guard against NaN/Inf and explosions."""
        sh = torch.nan_to_num(self._sh_coeffs, nan=0.0, posinf=0.0, neginf=0.0)
        # Clamp to a reasonable range to avoid extremely large magnitudes driving instabilities
        return torch.clamp(sh, min=-5.0, max=5.0)

    def _safe_rotation_quaternions(self):
        """Return normalized, finite quaternions; replace invalid rows with identity quaternion."""
        q = torch.nan_to_num(self._rotation, nan=0.0, posinf=0.0, neginf=0.0)
        norms = torch.norm(q, dim=-1, keepdim=True)
        bad = (norms < 1e-8) | (~torch.isfinite(norms))
        # Build identity-quaternion tensor and select per-row
        ident_full = q.new_zeros(q.shape)
        ident_full[:, 0] = 1.0
        q = torch.where(bad, ident_full, q)
        # Normalize safely
        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-8)
        return q

    def _safe_xyz(self):
        """Return a sanitized copy of xyz positions, clamped to a reasonable box and finite."""
        xyz = torch.nan_to_num(self._xyz, nan=0.0, posinf=0.0, neginf=0.0)
        # Clamp positions to within a broad bounding box to prevent numeric overflow in distances
        bound = float(self.max_norm) * 2.0
        return torch.clamp(xyz, min=-bound, max=bound)
        
    def _create_tf_grid(self):
        """Create time-frequency coordinate grid for spectrogram pixels"""
        f_coords = torch.arange(self.freq_num).float()
        t_coords = torch.arange(self.time_num).float()
        
        # Create meshgrid and flatten
        f_grid, t_grid = torch.meshgrid(f_coords, t_coords, indexing='ij')
        tf_grid = torch.stack([f_grid.flatten(), t_grid.flatten()], dim=1)
        
        return tf_grid
        
    def initialize_from_source_audio(self, source_audio, device):
        """Initialize 0th-order SH (DC) from source audio magnitude with safe scaling.
        - Uses same STFT params as forward (n_fft=512, hop=160, win=400, Hamming window)
        - Normalizes to [0, 1] by max, guards NaN/Inf, and clamps extremes
        """
        with torch.no_grad():
            source_audio = source_audio.to(device)
            # Mono mix if stereo
            mono = source_audio.mean(0) if source_audio.dim() > 1 else source_audio
            # STFT params (match model forward)
            n_fft = 512
            hop_length = 160
            window_length = 400
            torch_window = torch.hamming_window(window_length, device=device)
            spec = torch.stft(
                mono, n_fft=n_fft, hop_length=hop_length, win_length=window_length,
                window=torch_window, return_complex=True
            )
            mag = torch.abs(spec)
            # Safe numeric handling
            mag = torch.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
            maxv = torch.max(mag)
            if torch.isfinite(maxv) and maxv > 0:
                mag = mag / maxv
            else:
                mag = torch.zeros_like(mag)
            # Resize to model grid if needed
            if mag.shape[0] != self.freq_num or mag.shape[1] != self.time_num:
                mag = F.interpolate(mag.unsqueeze(0).unsqueeze(0), size=(self.freq_num, self.time_num), mode='bilinear').squeeze()
            # Clamp to safe range
            mag = mag.clamp_(0.0, 1.0)
            # Assign to DC coefficient
            self._sh_coeffs[:, 0, 0] = mag.flatten()
    
    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrices to quaternions
        Args:
            R: [N, 3, 3] rotation matrices
        Returns:
            quaternions: [N, 4] quaternions (w, x, y, z)
        """
        N = R.shape[0]
        quaternions = torch.zeros((N, 4), device=R.device)
        
        # Calculate trace
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        
        # Case 1: trace > 0
        mask1 = trace > 0
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
        quaternions[mask1, 0] = 0.25 * s
        quaternions[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        quaternions[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        quaternions[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s
        
        # Case 2: R[0,0] is max diagonal
        mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        if mask2.any():
            s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2  # s = 4 * x
            quaternions[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
            quaternions[mask2, 1] = 0.25 * s
            quaternions[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
            quaternions[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s
        
        # Case 3: R[1,1] is max diagonal
        mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
        if mask3.any():
            s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2  # s = 4 * y
            quaternions[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
            quaternions[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
            quaternions[mask3, 2] = 0.25 * s
            quaternions[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s
        
        # Case 4: R[2,2] is max diagonal
        mask4 = (~mask1) & (~mask2) & (~mask3)
        if mask4.any():
            s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2  # s = 4 * z
            quaternions[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
            quaternions[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
            quaternions[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
            quaternions[mask4, 3] = 0.25 * s
        
        return F.normalize(quaternions, dim=-1)
    
    def quaternion_to_rotation_matrix(self, quaternions):
        """Convert quaternions to rotation matrices
        Args:
            quaternions: [N, 4] quaternions (w, x, y, z)
        Returns:
            rotation_matrices: [N, 3, 3] rotation matrices
        """
        # Normalize quaternions
        quaternions = F.normalize(quaternions, dim=-1)
        
        w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        
        # Convert to rotation matrices
        R = torch.zeros((quaternions.shape[0], 3, 3), device=quaternions.device)
        
        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x*y - w*z)
        R[:, 0, 2] = 2 * (x*z + w*y)
        
        R[:, 1, 0] = 2 * (x*y + w*z)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y*z - w*x)
        
        R[:, 2, 0] = 2 * (x*z - w*y)
        R[:, 2, 1] = 2 * (y*z + w*x)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
        
        return R
    
    def compute_relative_positions(self, cam_pose):
        """Compute relative vectors from points to microphone in both world/camera frames.

        Returns:
            relative_pos_world_norm: [B, N, 3] world-frame vectors normalized by max_norm
            dir_pp_world:            [B, N, 3] from microphone to point (world, unnormalized)
            relative_pos_cam_norm:   [B, N, 3] camera-frame vectors normalized by max_norm
        """
        B = cam_pose.shape[0]

        # Extract microphone position and, if present, camera rotation
        microphone_pos = cam_pose[:, :3]  # [B, 3]
        R_cam = None
        if cam_pose.shape[1] >= 12:
            try:
                R_cam = cam_pose[:, 3:].reshape(B, 3, 3)
            except Exception:
                R_cam = None

        # Relative vectors (world frame): point -> mic
        safe_xyz = self._safe_xyz()
        rel_world = (microphone_pos.unsqueeze(1).repeat(1, self.n_points, 1) -
                     safe_xyz.unsqueeze(0).repeat(B, 1, 1))  # [B, N, 3]

        dir_pp_world = -rel_world  # mic -> point (unused downstream but kept for compatibility)

        # Normalized world vectors for network inputs (e.g., distance prior)
        rel_world_norm = rel_world / self.max_norm

        # Camera-frame vectors (optional). Use R_cam^T to transform world->camera
        if R_cam is not None:
            R_t = R_cam.transpose(1, 2)  # [B, 3, 3]
            rel_cam = torch.matmul(rel_world, R_t)  # broadcasting [B,N,3]x[B,3,3] -> [B,N,3]
            rel_cam_norm = rel_cam / self.max_norm
        else:
            rel_cam_norm = rel_world_norm

        return rel_world_norm, dir_pp_world, rel_cam_norm
    
    def eval_spherical_harmonics_magnitude(self, relative_pos):
        """Evaluate spherical harmonics to get directional magnitude
        
        Args:
            relative_pos: [B, N, 3] relative positions from camera to points
        Returns:
            sh_values: [B, N] directional magnitudes
        """
        B, N, _ = relative_pos.shape

        # Normalize direction vectors (robust to zero vectors)
        dir_normalized = F.normalize(relative_pos, dim=-1, eps=1e-8)  # [B, N, 3]

        if self.use_point_rotation:
            # Rotate viewing directions into each point's local frame via per-point quaternion.
            rotation_matrices = self.quaternion_to_rotation_matrix(self._safe_rotation_quaternions())  # [N, 3, 3]
            rotation_matrices_inv = rotation_matrices.transpose(1, 2)  # [N, 3, 3]
            rotation_matrices_inv_expanded = rotation_matrices_inv.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, N, 3, 3]
            dir_normalized_expanded = dir_normalized.unsqueeze(-1)  # [B, N, 3, 1]
            dir_local = torch.matmul(rotation_matrices_inv_expanded, dir_normalized_expanded).squeeze(-1)  # [B, N, 3]
        else:
            # Directly use world/camera-frame directions (no per-point rotation), like vanilla 3DGS.
            dir_local = dir_normalized

        dir_local = torch.nan_to_num(dir_local, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reshape for SH evaluation: [B, N, 3] -> [B*N, 3]
        dir_flat = dir_local.reshape(-1, 3)
        
        # Expand SH coefficients for all batches: [N, 1, C] -> [B*N, 1, C]
        # Use sanitized SH coefficients to prevent NaN/Inf propagation
        sh_coeffs_expanded = self._safe_sh_coeffs().unsqueeze(0).repeat(B, 1, 1, 1).reshape(-1, 1, self.num_sh_coeffs)
        
        # Evaluate spherical harmonics up to configured degree
        # eval_sh returns [..., C], where C=1 in our case
        # Evaluate SH and sanitize
        sh_values_flat = eval_sh(self.sh_degree, sh_coeffs_expanded, dir_flat)  # [B*N, 1]
        sh_values_flat = torch.nan_to_num(sh_values_flat, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reshape back to batch format: [B*N, 1] -> [B, N]
        sh_values = sh_values_flat.squeeze(-1).reshape(B, N)
        
        sh_values = F.softplus(sh_values)  # Use ReLU instead of sigmoid for better gradient flow
        
        return sh_values  # [B, N]
    
    def points_to_spectrogram(self, point_values, relative_pos, source_magnitude,
                               ild_spec: torch.Tensor = None,
                               side_spec: torch.Tensor = None):
        """Prepare renderer inputs from point values and geometry.

        Args:
            point_values: [B, N] SH-evaluated magnitudes per point
            relative_pos: [B, N, 3] normalized (by max_norm) vectors from point->mic
            source_magnitude: [B, F, T] magnitude spectrogram of source mono

        Returns:
            mono_features: [B, 3, F, T] = [source_mag, SH_value, inv_distance]
            diff_features: [B, 1, F, T] = [SH_value]
        """
        B = point_values.shape[0]

        # Reshape SH directional magnitude to spectrogram layout
        spec_values = point_values.reshape(B, self.freq_num, self.time_num)

        # Distances in world units: relative_pos is normalized by max_norm
        distances_norm = torch.norm(relative_pos, dim=-1).reshape(B, self.freq_num, self.time_num)
        distances_world = distances_norm * self.max_norm

        # Use inverse distance as physically motivated attenuation cue
        inv_distance = 1.0 / (1e-3 + distances_world)

        # Ensure source magnitude matches (F, T)
        if source_magnitude.shape[-2:] != spec_values.shape[-2:]:
            source_magnitude = F.interpolate(
                source_magnitude.unsqueeze(1),
                size=spec_values.shape[-2:],
                mode='bilinear', align_corners=False
            ).squeeze(1)

        # Build features
        mono_features = torch.stack([source_magnitude, spec_values, inv_distance], dim=1)

        # For diff branch, start from SH-based directional magnitude
        diff_chans = [spec_values]
        # Optionally inject ILD
        if self.use_stereo_cues and (ild_spec is not None):
            if ild_spec.shape[-2:] != spec_values.shape[-2:]:
                ild_spec = F.interpolate(ild_spec.unsqueeze(1), size=spec_values.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
            diff_chans.append(ild_spec)
        # Optionally inject inv_distance
        if self.diff_use_inv_distance:
            diff_chans.append(inv_distance)
        # Optionally inject side magnitude (from input stereo)
        if self.diff_use_side_mag and (side_spec is not None):
            if side_spec.shape[-2:] != spec_values.shape[-2:]:
                side_spec = F.interpolate(side_spec.unsqueeze(1), size=spec_values.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
            diff_chans.append(side_spec)
        diff_features = torch.stack(diff_chans, dim=1)

        # Final numeric safety on features before feeding the renderer
        mono_features = torch.nan_to_num(mono_features, nan=0.0, posinf=0.0, neginf=0.0)
        diff_features = torch.nan_to_num(diff_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return mono_features, diff_features
    
    def forward(self, cam_pose, source_audio, is_val=False, return_mag: bool=False):
        """
        Forward pass for Audio 3DGS
        
        Args:
            cam_pose: [B, 12] Camera poses
            source_audio: [B, 2, T] Source audio waveform
            is_val: Whether in validation mode
            
        Returns:
            [B, 2, T] Predicted binaural audio
        """
        device = cam_pose.device
        B = cam_pose.shape[0]
        
        # STFT parameters (aligned with losses/metrics): 512/160/400, Hamming
        n_fft = 512
        hop_length = 160
        window_length = 400
        torch_window = torch.hamming_window(window_length).to(device)
        
        # Convert source audio to spectrogram
        # Stereo case: DO NOT average before STFT. Compute STFT per channel.
        stereo_input = (source_audio.dim() > 1 and source_audio.shape[1] > 1)
        ild_spec = None
        side_spec = None
        if stereo_input:
            sig_L = source_audio[:, 0]
            sig_R = source_audio[:, 1]
            spec_L = torch.stft(
                sig_L, n_fft=n_fft, hop_length=hop_length,
                win_length=window_length, window=torch_window, return_complex=True
            )
            spec_R = torch.stft(
                sig_R, n_fft=n_fft, hop_length=hop_length,
                win_length=window_length, window=torch_window, return_complex=True
            )
            mag_L = torch.abs(spec_L)
            mag_R = torch.abs(spec_R)
            phase_L = torch.angle(spec_L)
            phase_R = torch.angle(spec_R)
            # Use the average magnitude as content carrier for mask prediction
            source_magnitude = 0.5 * (mag_L + mag_R)
            source_phase_L = torch.nan_to_num(phase_L, nan=0.0, posinf=0.0, neginf=0.0)
            source_phase_R = torch.nan_to_num(phase_R, nan=0.0, posinf=0.0, neginf=0.0)
            source_magnitude = torch.nan_to_num(source_magnitude, nan=0.0, posinf=0.0, neginf=0.0)
            # Optional ILD stereo cue: log-ratio of magnitudes
            if self.use_stereo_cues:
                eps = 1e-8
                ild_spec = torch.log(torch.clamp(mag_L, min=eps)) - torch.log(torch.clamp(mag_R, min=eps))
                ild_spec = torch.nan_to_num(ild_spec, nan=0.0, posinf=0.0, neginf=0.0)
            # Optional side magnitude cue (Mid/Side transform): side = (L - R)/2 in magnitude domain
            if self.diff_use_side_mag:
                # Note: We use magnitude-domain side. Keep scale consistent with mid/side synthesis (÷2)
                side_spec = 0.5 * torch.clamp(mag_L - mag_R, min=0.0)  # non-negative magnitude proxy
        else:
            # Mono case: regular STFT
            source_mono = source_audio.squeeze() if source_audio.dim() > 1 else source_audio
            source_spec_complex = torch.stft(
                source_mono, n_fft=n_fft, hop_length=hop_length,
                win_length=window_length, window=torch_window, return_complex=True
            )
            source_magnitude = torch.abs(source_spec_complex)
            source_phase = torch.angle(source_spec_complex)
            source_magnitude = torch.nan_to_num(source_magnitude, nan=0.0, posinf=0.0, neginf=0.0)
            source_phase = torch.nan_to_num(source_phase, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 1. Compute relative vectors in world/camera frames
        rel_world_norm, dir_pp, rel_cam_norm = self.compute_relative_positions(cam_pose)

        # 2. Evaluate spherical harmonics for directional magnitude
        #    Use camera-frame directions when enabled to capture head/camera orientation
        rel_for_sh = rel_cam_norm if self.use_cam_rotation else rel_world_norm
        directional_magnitude = self.eval_spherical_harmonics_magnitude(rel_for_sh)
        
        # 3. Convert points back to features for dual-branch renderer (content-aware)
        #    Feed source magnitude so masks can depend on input audio content
        # For distance prior, always use world-frame distances
        mono_features, diff_features = self.points_to_spectrogram(
            directional_magnitude, rel_world_norm, source_magnitude,
            ild_spec=ild_spec, side_spec=side_spec
        )
        
        # 4. Dual-branch U-Net rendering to predict binaural masks
        mono_mask, diff_mask = self.renderer(mono_features, diff_features)
        
        # 5. Apply masks to source spectrogram
        # Ensure masks match source spectrogram size
        if mono_mask.shape[-2:] != source_magnitude.shape[-2:]:
            mono_mask = F.interpolate(mono_mask, size=source_magnitude.shape[-2:], mode='bilinear')
            diff_mask = F.interpolate(diff_mask, size=source_magnitude.shape[-2:], mode='bilinear')
        
        mono_mask = mono_mask.squeeze(1)
        diff_mask = diff_mask.squeeze(1)
        
        # Generate mono and diff spectrograms
        mono_mag = mono_mask * source_magnitude
        mono_mag = torch.nan_to_num(mono_mag, nan=0.0, posinf=0.0, neginf=0.0)
        diff_envelope = diff_mask

        # Cascaded synthesis: difference envelope scales mono energy (avoid shortcut spatialization).
        left_magnitude = torch.relu(mono_mag * (1.0 + diff_envelope))
        right_magnitude = torch.relu(mono_mag * (1.0 - diff_envelope))
        
        # Reconstruct complex spectrograms using source phase for both ears
        if stereo_input:
            left_phase = source_phase_L
            right_phase = source_phase_R
        else:
            left_phase = source_phase
            right_phase = source_phase

        left_complex = torch.polar(left_magnitude, left_phase)
        right_complex = torch.polar(right_magnitude, right_phase)
        
        # Convert back to time domain
        sig_len = source_audio.shape[-1] if isinstance(source_audio, torch.Tensor) else None
        left_audio = torch.istft(
            left_complex, n_fft=n_fft, hop_length=hop_length, 
            win_length=window_length, window=torch_window, length=sig_len
        )
        right_audio = torch.istft(
            right_complex, n_fft=n_fft, hop_length=hop_length, 
            win_length=window_length, window=torch_window, length=sig_len
        )
        
        # Stack to binaural output
        binaural_output = torch.stack([left_audio, right_audio], dim=1)
        # Numeric safety: always sanitize NaN/Inf; no amplitude clamp here
        binaural_output = torch.nan_to_num(binaural_output, nan=0.0, posinf=0.0, neginf=0.0)

        if return_mag:
            return binaural_output, left_magnitude, right_magnitude
        return binaural_output
    
    def load_state_dict(self, state_dict, strict=True):
        """Custom load_state_dict to handle rotation format conversion"""
        # Work on a shallow copy to avoid mutating caller's dict
        sd = dict(state_dict)

        # 1) Backward-compat: convert rotation matrices -> quaternions
        if '_rotation' in sd:
            rotation_param = sd['_rotation']
            if len(rotation_param.shape) == 3 and rotation_param.shape[1] == 3 and rotation_param.shape[2] == 3:
                print(f"Converting rotation matrices to quaternions: {rotation_param.shape} -> [{rotation_param.shape[0]}, 4]")
                quaternions = self.rotation_matrix_to_quaternion(rotation_param)
                sd['_rotation'] = quaternions

        # 2) Backward-compat: adapt diff branch first conv from 2->1 input channels
        key = 'renderer.diff_enc1.0.weight'
        if key in sd:
            w = sd[key]
            try:
                current_w = self.renderer.diff_enc1[0].weight
                if w.dim() == 4 and current_w.dim() == 4 and w.shape != current_w.shape:
                    in_ckpt = w.shape[1]
                    in_need = current_w.shape[1]
                    if in_ckpt == in_need:
                        pass
                    elif in_ckpt > in_need:
                        # Slice first in_need channels
                        new_w = w[:, :in_need, :, :].contiguous()
                        sd[key] = new_w
                        print(f"Adapted checkpoint {key} from {tuple(w.shape)} to {tuple(new_w.shape)} (sliced channels)")
                    else:
                        # Replicate last channel to match required in_need
                        reps = in_need - in_ckpt
                        rep_w = torch.cat([w, w[:, -1:, :, :].repeat(1, reps, 1, 1)], dim=1).contiguous()
                        sd[key] = rep_w
                        print(f"Adapted checkpoint {key} from {tuple(w.shape)} to {tuple(rep_w.shape)} (replicated channels)")
            except Exception:
                pass

        # 3) Backward/forward-compat: adapt SH coeffs size if degree changed
        try:
            key = '_sh_coeffs'
            if key in sd and hasattr(self, '_sh_coeffs'):
                old_sh = sd[key]
                new_shape = self._sh_coeffs.shape
                if old_sh.shape != new_shape:
                    # Create new tensor and copy overlapping coefficients
                    new_sh = torch.zeros(new_shape, dtype=old_sh.dtype, device=old_sh.device)
                    min_c = min(old_sh.shape[-1], new_shape[-1])
                    new_sh[..., :min_c] = old_sh[..., :min_c]
                    sd[key] = new_sh
                    print(f"Adapted SH coeffs from {tuple(old_sh.shape)} to {tuple(new_shape)} (copied first {min_c})")
        except Exception:
            pass

        # Load the modified state dict
        return super().load_state_dict(sd, strict=strict)


def build_model(cfg, gaussian_model=None, scene=None):
    """Build Audio 3DGS model"""
    # Use config parameters for dimensions
    freq_num = cfg.dataset.get('H', 257)
    time_num = cfg.dataset.get('W', 348)
    model = Audio3DGS(cfg, freq_num=freq_num, time_num=time_num)
    return model
