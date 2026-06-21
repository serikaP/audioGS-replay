"""
Enhanced Criterion for Audio 3DGS with spatial audio emphasis
Modified from original Criterion to add spatial audio penalties
"""

from typing import Any

import auraloss
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.criterions.Criterion_1 import (
    stft,
    SpectralConvergenceLoss,
    LogMagSTFTLoss,
    MagSTFTLoss,
    MultiResolutionSTFTLoss,
)


class SpatialAudioLoss(nn.Module):
    """Spatial audio loss component to emphasize stereo effects"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_wav, gt_wav):
        """
        Calculate spatial audio loss emphasizing stereo characteristics
        
        Args:
            pred_wav: [B, 2, T] predicted binaural audio
            gt_wav: [B, 2, T] ground truth binaural audio
        
        Returns:
            dict: spatial loss components
        """
        losses = {}
        
        # 先做数值清理，确保无 NaN/Inf 且幅值有界
        pred_wav = torch.nan_to_num(pred_wav, nan=0.0, posinf=0.0, neginf=0.0)
        gt_wav = torch.nan_to_num(gt_wav, nan=0.0, posinf=0.0, neginf=0.0)
        pred_wav = torch.clamp(pred_wav, min=-2.0, max=2.0)
        gt_wav = torch.clamp(gt_wav, min=-2.0, max=2.0)

        # 1) 左右能量比（LRE）
        pred_left_energy = pred_wav[:, 0, :].pow(2).sum(dim=-1)  # [B]
        pred_right_energy = pred_wav[:, 1, :].pow(2).sum(dim=-1)  # [B]
        gt_left_energy = gt_wav[:, 0, :].pow(2).sum(dim=-1)     # [B]
        gt_right_energy = gt_wav[:, 1, :].pow(2).sum(dim=-1)     # [B]
        
        # 避免除零
        eps = 1e-8
        pred_lr_ratio = torch.log10((pred_left_energy + eps) / (pred_right_energy + eps))
        gt_lr_ratio = torch.log10((gt_left_energy + eps) / (gt_right_energy + eps))
        pred_lr_ratio = torch.nan_to_num(pred_lr_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        gt_lr_ratio = torch.nan_to_num(gt_lr_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        
        losses['lre_loss'] = F.l1_loss(pred_lr_ratio, gt_lr_ratio)
        
        # 2) 双声道相关性（相干性）
        pred_correlation = F.cosine_similarity(pred_wav[:, 0, :], pred_wav[:, 1, :], dim=-1)
        gt_correlation = F.cosine_similarity(gt_wav[:, 0, :], gt_wav[:, 1, :], dim=-1)
        pred_correlation = torch.nan_to_num(pred_correlation, nan=0.0, posinf=0.0, neginf=0.0)
        gt_correlation = torch.nan_to_num(gt_correlation, nan=0.0, posinf=0.0, neginf=0.0)
        losses['coherence_loss'] = F.l1_loss(pred_correlation, gt_correlation)
        
        # 3) 双声道相位差
        win = torch.hann_window(400, device=pred_wav.device)
        pred_stft_left = torch.stft(pred_wav[:, 0, :], n_fft=512, hop_length=160, win_length=400,
                                    window=win, pad_mode='constant', return_complex=True)
        pred_stft_right = torch.stft(pred_wav[:, 1, :], n_fft=512, hop_length=160, win_length=400,
                                     window=win, pad_mode='constant', return_complex=True)
        win2 = torch.hann_window(400, device=gt_wav.device)
        gt_stft_left = torch.stft(gt_wav[:, 0, :], n_fft=512, hop_length=160, win_length=400,
                                   window=win2, pad_mode='constant', return_complex=True)
        gt_stft_right = torch.stft(gt_wav[:, 1, :], n_fft=512, hop_length=160, win_length=400,
                                    window=win2, pad_mode='constant', return_complex=True)
        
        # Stack to create multi-channel STFT
        pred_stft = torch.stack([pred_stft_left, pred_stft_right], dim=1)  # [B, 2, F, T]
        gt_stft = torch.stack([gt_stft_left, gt_stft_right], dim=1)  # [B, 2, F, T]
        
        pred_phase = torch.angle(pred_stft)
        gt_phase = torch.angle(gt_stft)
        pred_phase = torch.nan_to_num(pred_phase, nan=0.0, posinf=0.0, neginf=0.0)
        gt_phase = torch.nan_to_num(gt_phase, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Phase difference between channels
        pred_phase_diff = pred_phase[:, 0, :, :] - pred_phase[:, 1, :, :]
        gt_phase_diff = gt_phase[:, 0, :, :] - gt_phase[:, 1, :, :]
        
        losses['phase_diff_loss'] = F.l1_loss(
            torch.cos(pred_phase_diff), torch.cos(gt_phase_diff)
        )
        
        # 4) 总能量平衡
        pred_total_energy = pred_wav.pow(2).sum()
        gt_total_energy = gt_wav.pow(2).sum()
        
        # Encourage similar overall energy
        losses['energy_balance_loss'] = F.l1_loss(
            torch.log10(pred_total_energy + eps), 
            torch.log10(gt_total_energy + eps)
        )
        
        return losses


class EnhancedCriterion(nn.Module):
    """Enhanced Criterion for Audio 3DGS with spatial audio emphasis"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Original losses
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.l1_loss = torch.nn.L1Loss(reduction='none')
        self.stft_loss = LogMagSTFTLoss(fft_size=512, shift_size=128, win_length=512,
                                     window="hamming_window")
        self.spec_loss = auraloss.freq.STFTLoss(reduction='none')
        self.random_stft_loss = auraloss.freq.RandomResolutionSTFTLoss(reduction='none', window="hamming_window")
        self.mag_loss = MagSTFTLoss(fft_size=512, shift_size=160, win_length=400,
                                        window="hamming_window")
        self.sum_diff = auraloss.freq.SumAndDifferenceSTFTLoss(fft_sizes=[512], hop_sizes=[128], win_lengths=[512])
        self.sdrloss = auraloss.time.SISDRLoss()
        
        # Spatial audio losses
        self.spatial_loss = SpatialAudioLoss()
        
        # Loss weights (更合理的权重)
        self.spatial_weight = 0.2   # 进一步降低空间分量权重，减小梯度波动
        self.energy_weight = 0.1    # 能量项权重再降，避免对数在极小能量时放大
        
        # 配置项：是否启用 MR-STFT 损失（multi_res_stft_loss）
        # - 若关闭，则完全不计算 MR-STFT 项，仅使用固定单分辨率 STFT 相关损失
        train_cfg = getattr(cfg, 'train', {})
        self.use_mr_stft_loss: bool = bool(getattr(train_cfg, 'use_mr_stft_loss', True))

        # Fallback MR-STFT suitable for short clips (e.g., 1s @ 22.05kHz)
        # Avoids auraloss RandomResolutionSTFTLoss max n_fft (e.g., 32768) constraint
        self._mrstft_fallback = MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024, 2048, 4096],
            hop_sizes=[128, 256, 512, 1024],
            win_lengths=[512, 1024, 2048, 4096],
            factor_sc=1.0,
            factor_mag=1.0,
        )
        # 配置：是否使用随机分辨率 STFT 以及触发随机 STFT 的最小长度阈值
        # - 默认保持现状：use_random_stft=True, randstft_max_fft=32768
        # - 若设置 use_random_stft=False 或 randstft_max_fft<=0，则始终使用固定 MR-STFT（可复现实验）
        self.use_random_stft: bool = bool(getattr(train_cfg, 'use_random_stft', True))
        self.randstft_max_fft: int = int(getattr(train_cfg, 'randstft_max_fft', 32768))

        # Largest n_fft used by auraloss RandomResolutionSTFTLoss by default（保留旧字段名做兼容）
        self._randstft_max_fft = self.randstft_max_fft

        # Optional absolute phase supervision weight (per-channel)
        # 0.0 by default to keep backward-compatible behavior
        train_cfg = getattr(cfg, 'train')
        try:
            self.abs_phase_weight: float = float(getattr(train_cfg, 'abs_phase_weight', 0.0))
        except Exception:
            self.abs_phase_weight = 0.0
        
    def forward(self, pred_wav, gt_wav, out_pred_wav_0=None):
        scalar_stats = {}
        
        if pred_wav is not None:
            # Ensure correct shape
            if len(pred_wav.shape) == 2:
                pred_wav = pred_wav.unsqueeze(0)
            if len(gt_wav.shape) == 2:
                gt_wav = gt_wav.unsqueeze(0)
            
            # Original spectral losses（加入数值清理）
            x0 = pred_wav.reshape(-1, pred_wav.shape[-1]).contiguous().float()
            y0 = gt_wav.reshape(-1, pred_wav.shape[-1]).contiguous().float()
            x0 = torch.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0)
            y0 = torch.nan_to_num(y0, nan=0.0, posinf=0.0, neginf=0.0)
            scalar_stats['wav_mag_loss'] = 1.0 * self.stft_loss(
                x0, y0
            ).mean()
            scalar_stats['wav_mag_loss'] = torch.nan_to_num(scalar_stats['wav_mag_loss'], nan=0.0, posinf=0.0, neginf=0.0)
            
            # 可选：多分辨率 STFT 损失（MR-STFT）。若关闭，则跳过该项。
            if self.use_mr_stft_loss:
                # Add multi-resolution STFT loss (降低权重)
                # 可配置：固定 MR-STFT 或随机分辨率 STFT（默认随机，与现有行为一致）
                T = pred_wav.shape[-1]
                use_random = self.use_random_stft and (self.randstft_max_fft <= 0 or T >= self.randstft_max_fft)
                if use_random:
                    try:
                        mr_val = self.random_stft_loss(
                            pred_wav.contiguous().float(),
                            gt_wav.contiguous().float(),
                        ).mean()
                        mr_val = torch.nan_to_num(mr_val, nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception:
                        mr_val = x0.new_tensor(0.0)
                else:
                    x = pred_wav.reshape(-1, T).contiguous().float()
                    y = gt_wav.reshape(-1, T).contiguous().float()
                    sc_l, mag_l = self._mrstft_fallback(x, y)
                    # Combine spectral convergence and log-magnitude losses
                    mr_val = (sc_l + mag_l)
                    try:
                        mr_val = torch.nan_to_num(mr_val.mean(), nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception:
                        mr_val = x.new_tensor(0.0)
                scalar_stats['multi_res_stft_loss'] = 0.1 * mr_val
            
            # Add magnitude STFT loss (降低权重)
            # Handle multi-channel audio by averaging channels
            if pred_wav.dim() == 3:  # [B, 2, T]
                pred_mono = pred_wav.mean(1)  # Convert to mono
                gt_mono = gt_wav.mean(1)
            else:
                pred_mono = pred_wav
                gt_mono = gt_wav
                
            mval = self.mag_loss(
                pred_mono.contiguous().float(), 
                gt_mono.contiguous().float()
            ).mean()
            scalar_stats['mag_stft_loss'] = 0.25 * torch.nan_to_num(mval, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Add sum and difference loss (降低权重)
            sdl = self.sum_diff(
                pred_wav.contiguous().float(), 
                gt_wav.contiguous().float()
            ).mean()
            scalar_stats['sum_diff_loss'] = 0.5 * torch.nan_to_num(sdl, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Spatial audio losses
            spatial_losses = self.spatial_loss(pred_wav, gt_wav)

            # Optional: absolute phase loss per channel to encourage phase alignment
            abs_phase_term = pred_wav.new_tensor(0.0)
            if self.abs_phase_weight > 0.0:
                try:
                    win = torch.hann_window(400, device=pred_wav.device)
                    # Left channel
                    pL = torch.stft(pred_wav[:, 0, :], n_fft=512, hop_length=160, win_length=400,
                                     window=win, pad_mode='constant', return_complex=True)
                    gL = torch.stft(gt_wav[:, 0, :], n_fft=512, hop_length=160, win_length=400,
                                     window=win, pad_mode='constant', return_complex=True)
                    # Right channel
                    pR = torch.stft(pred_wav[:, 1, :], n_fft=512, hop_length=160, win_length=400,
                                     window=win, pad_mode='constant', return_complex=True)
                    gR = torch.stft(gt_wav[:, 1, :], n_fft=512, hop_length=160, win_length=400,
                                     window=win, pad_mode='constant', return_complex=True)

                    # Phase difference (wrap-invariant) using cosine distance
                    def phase_abs_loss(px, gx):
                        dphi = torch.angle(px) - torch.angle(gx)
                        dphi = torch.nan_to_num(dphi, nan=0.0, posinf=0.0, neginf=0.0)
                        # 1 - cos(dphi) in [0,2], average over F,T
                        return (1.0 - torch.cos(dphi)).mean()
                    absL = phase_abs_loss(pL, gL)
                    absR = phase_abs_loss(pR, gR)
                    abs_phase_term = 0.5 * (absL + absR)
                except Exception:
                    abs_phase_term = pred_wav.new_tensor(0.0)
            
            # Add spatial losses with increased weights
            # 对每个空间项单独做数值清理与降权
            scalar_stats['lre_loss'] = self.spatial_weight * torch.nan_to_num(spatial_losses['lre_loss'], nan=0.0, posinf=0.0, neginf=0.0)
            scalar_stats['coherence_loss'] = self.spatial_weight * torch.nan_to_num(spatial_losses['coherence_loss'], nan=0.0, posinf=0.0, neginf=0.0)
            scalar_stats['phase_diff_loss'] = 0.1 * self.spatial_weight * torch.nan_to_num(spatial_losses['phase_diff_loss'], nan=0.0, posinf=0.0, neginf=0.0)
            scalar_stats['energy_balance_loss'] = 0.01 * self.energy_weight * torch.nan_to_num(spatial_losses['energy_balance_loss'], nan=0.0, posinf=0.0, neginf=0.0)
            
            # Combine all losses
            # 汇总总损失（根据是否启用 MR-STFT 决定是否加上该项）
            total_loss = (
                scalar_stats['wav_mag_loss'] +
                (scalar_stats.get('multi_res_stft_loss', 0.0)) +
                scalar_stats['mag_stft_loss'] +
                scalar_stats['sum_diff_loss'] +
                scalar_stats['lre_loss'] +
                scalar_stats['coherence_loss'] +
                scalar_stats['phase_diff_loss'] +
                scalar_stats['energy_balance_loss'] +
                self.abs_phase_weight * abs_phase_term
            )
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
            scalar_stats['total_loss'] = total_loss

            if self.abs_phase_weight > 0.0:
                scalar_stats['abs_phase_loss'] = self.abs_phase_weight * abs_phase_term
            
            # Auxiliary losses if available
            if out_pred_wav_0 is not None:
                if out_pred_wav_0.shape[1] == 1:
                    gt_wav_aux = gt_wav.mean(1).unsqueeze(1)
                else:
                    gt_wav_aux = gt_wav
                    
                scalar_stats['mse_loss_aux'] = 20. * self.mse_loss(
                    out_pred_wav_0.contiguous().float(), 
                    gt_wav_aux.float()
                ).mean()
                
                scalar_stats['wav_mag_loss_aux'] = 20. * self.stft_loss(
                    out_pred_wav_0.reshape(-1, out_pred_wav_0.shape[-1]).contiguous().float(), 
                    gt_wav_aux.reshape(-1, gt_wav_aux.shape[-1]).contiguous().float()
                ).mean()
                
                # 可选：辅助分支的 MR-STFT（若关闭 MR-STFT 则跳过）
                if self.use_mr_stft_loss:
                    Taux = out_pred_wav_0.shape[-1]
                    use_random_aux = self.use_random_stft and (self.randstft_max_fft <= 0 or Taux >= self.randstft_max_fft)
                    if use_random_aux:
                        aux_val = self.random_stft_loss(
                            out_pred_wav_0.contiguous().float(),
                            gt_wav_aux.contiguous().float(),
                        ).mean()
                    else:
                        xaux = out_pred_wav_0.reshape(-1, Taux).contiguous().float()
                        yaux = gt_wav_aux.reshape(-1, Taux).contiguous().float()
                        sc_a, mag_a = self._mrstft_fallback(xaux, yaux)
                        aux_val = (sc_a + mag_a)
                        try:
                            aux_val = aux_val.mean()
                        except Exception:
                            pass
                    scalar_stats['wav_spec_loss_aux'] = 0.25 * aux_val
        
        return scalar_stats
