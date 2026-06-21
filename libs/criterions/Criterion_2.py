import torch
import torch.nn as nn
import torch.nn.functional as F

def stft(x, fft_size, hop_size, win_length, window):
    """
    对输入信号执行 STFT 并转换为幅度谱。
    Args:
        x (Tensor): 输入信号张量 (B, T)。
        fft_size (int): FFT 点数。
        hop_size (int): 帧移。
        win_length (int): 窗长。
        window (Tensor): 窗函数。
    Returns:
        Tensor: 幅度谱 (B, #frames, fft_size // 2 + 1)。
    """
    window = window.to(x.device)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_stft = torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window, return_complex=True)
    
    # 使用 clamp 避免 nan 或 inf
    return torch.sqrt(torch.clamp(x_stft.abs()**2, min=1e-7))

class Criterion(nn.Module):
    """
    根据论文 "Extending Gaussian Splatting to Audio" 设计的增强版损失函数。
    该损失函数分别计算单声道 (mono) 和差分 (diff) 信号谱的 L2 距离。
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.l2_loss = nn.MSELoss()

        # STFT 参数应与模型和数据处理流程保持一致
        self.fft_size = 512
        self.hop_size = 160
        self.win_length = 400
        # Use Hamming window to align with model forward/metrics
        self.register_buffer("window", torch.hamming_window(self.win_length), persistent=False)
        # Optional controls
        self.use_log_mag_loss = bool(getattr(cfg.train, 'use_log_mag_loss', True))
        # Optional absolute phase supervision weight for mono/diff STFTs
        try:
            self.abs_phase_weight = float(getattr(cfg.train, 'abs_phase_weight', 0.0))
        except Exception:
            self.abs_phase_weight = 0.0
        # Optional PSNR image loss on spectrograms
        try:
            self.use_image_psnr_loss = bool(getattr(cfg.train, 'use_image_psnr_loss', False))
        except Exception:
            self.use_image_psnr_loss = False
        try:
            self.image_psnr_weight = float(getattr(cfg.train, 'image_psnr_weight', 0.0))
        except Exception:
            self.image_psnr_weight = 0.0

    def forward(self, pred_wav, gt_wav, out_pred_wav_0=None):
        """
        计算损失。
        Args:
            pred_wav (Tensor): 预测的双耳音频 (B, 2, T)。
            gt_wav (Tensor): 真实双耳音频 (B, 2, T)。
            out_pred_wav_0: 辅助输出，此处未使用。
        Returns:
            dict: 包含总损失和各分量损失的字典。
        """
        # 1. 将预测和真实的双耳音频分解为 mono 和 diff 分量
        pred_wav = torch.nan_to_num(pred_wav, nan=0.0, posinf=0.0, neginf=0.0)
        gt_wav = torch.nan_to_num(gt_wav, nan=0.0, posinf=0.0, neginf=0.0)
        pred_mono = pred_wav[:, 0, :] + pred_wav[:, 1, :]
        pred_diff = pred_wav[:, 0, :] - pred_wav[:, 1, :]
        
        gt_mono = gt_wav[:, 0, :] + gt_wav[:, 1, :]
        gt_diff = gt_wav[:, 0, :] - gt_wav[:, 1, :]

        # 2. 计算每个分量的 STFT 幅度谱
        pred_mono_spec = stft(pred_mono, self.fft_size, self.hop_size, self.win_length, self.window)
        gt_mono_spec = stft(gt_mono, self.fft_size, self.hop_size, self.win_length, self.window)
        
        pred_diff_spec = stft(pred_diff, self.fft_size, self.hop_size, self.win_length, self.window)
        gt_diff_spec = stft(gt_diff, self.fft_size, self.hop_size, self.win_length, self.window)
        
        if self.use_log_mag_loss:
            pred_mono_spec = torch.log1p(torch.clamp(pred_mono_spec, min=0.0))
            gt_mono_spec = torch.log1p(torch.clamp(gt_mono_spec, min=0.0))
            pred_diff_spec = torch.log1p(torch.clamp(pred_diff_spec, min=0.0))
            gt_diff_spec = torch.log1p(torch.clamp(gt_diff_spec, min=0.0))

        # 3. 计算 mono 和 diff 谱的 L2 损失
        loss_mono = self.l2_loss(pred_mono_spec, gt_mono_spec)
        loss_diff = self.l2_loss(pred_diff_spec, gt_diff_spec)

        # 3.1 可选：绝对相位损失（对齐 mono/diff 的相位，使用 cos 差避免 2π 模糊）
        phase_loss = pred_wav.new_tensor(0.0)
        if self.abs_phase_weight > 0.0:
            try:
                # STFT（复数）
                pred_mono_c = torch.stft(pred_mono, self.fft_size, self.hop_size, self.win_length, self.window, return_complex=True, pad_mode='constant')
                gt_mono_c = torch.stft(gt_mono, self.fft_size, self.hop_size, self.win_length, self.window, return_complex=True, pad_mode='constant')
                pred_diff_c = torch.stft(pred_diff, self.fft_size, self.hop_size, self.win_length, self.window, return_complex=True, pad_mode='constant')
                gt_diff_c = torch.stft(gt_diff, self.fft_size, self.hop_size, self.win_length, self.window, return_complex=True, pad_mode='constant')
                def abs_phase(px, gx):
                    dphi = torch.angle(px) - torch.angle(gx)
                    dphi = torch.nan_to_num(dphi, nan=0.0, posinf=0.0, neginf=0.0)
                    return (1.0 - torch.cos(dphi)).mean()
                phase_loss = 0.5 * (abs_phase(pred_mono_c, gt_mono_c) + abs_phase(pred_diff_c, gt_diff_c))
            except Exception:
                phase_loss = pred_wav.new_tensor(0.0)

        # 3.2 可选：在谱“图像”上计算 PSNR 作为额外监督（负 PSNR 作为损失）
        psnr_img_loss = pred_wav.new_tensor(0.0)
        if self.use_image_psnr_loss and self.image_psnr_weight > 0.0:
            eps = 1e-8
            def _psnr_loss(pred_img: torch.Tensor, tgt_img: torch.Tensor) -> torch.Tensor:
                # pred_img, tgt_img: [B, F, T]
                # 逐样本计算 MSE 与动态范围，输出为负 PSNR（越小越好）
                # 计算每个样本的 MSE
                # 为避免数值不稳，先清理
                pred_img = torch.nan_to_num(pred_img, nan=0.0, posinf=0.0, neginf=0.0)
                tgt_img = torch.nan_to_num(tgt_img, nan=0.0, posinf=0.0, neginf=0.0)
                # 展平到 [B, N]
                B = pred_img.shape[0]
                diff = (pred_img - tgt_img).reshape(B, -1)
                mse = (diff.pow(2).mean(dim=1) + eps)
                # 动态范围按样本计算，使用两者的联合范围
                max_val = torch.maximum(pred_img.amax(dim=(1,2)), tgt_img.amax(dim=(1,2)))
                min_val = torch.minimum(pred_img.amin(dim=(1,2)), tgt_img.amin(dim=(1,2)))
                data_range = torch.clamp(max_val - min_val, min=eps)
                psnr = 10.0 * torch.log10((data_range.pow(2)) / mse)
                # 负 PSNR 作为损失
                return (-psnr).mean()

            try:
                psnr_mono = _psnr_loss(pred_mono_spec, gt_mono_spec)
                psnr_diff = _psnr_loss(pred_diff_spec, gt_diff_spec)
                psnr_img_loss = 0.5 * (psnr_mono + psnr_diff)
            except Exception:
                psnr_img_loss = pred_wav.new_tensor(0.0)

        # 4. 组合损失
        # 引入可配置的差分分量权重以更好优化方向性（影响LRE）
        diff_w = float(getattr(self.cfg.train, 'diff_weight', 1.0))
        total_loss = loss_mono + diff_w * loss_diff + self.abs_phase_weight * phase_loss
        if self.use_image_psnr_loss and self.image_psnr_weight > 0.0:
            total_loss = total_loss + self.image_psnr_weight * psnr_img_loss
        
        scalar_stats = {
            'total_loss': total_loss,
            'mono_loss': loss_mono,
            'diff_loss': loss_diff,
            'abs_phase_loss': self.abs_phase_weight * phase_loss if self.abs_phase_weight > 0.0 else pred_wav.new_tensor(0.0)
        }
        if self.use_image_psnr_loss and self.image_psnr_weight > 0.0:
            scalar_stats['psnr_img_loss'] = self.image_psnr_weight * psnr_img_loss

        return scalar_stats
