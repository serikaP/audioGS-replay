import torch
import torch.nn as nn

from libs.criterions.Criterion_2 import stft


class MonoOnlyMSECriterion(nn.Module):
    """
    Mono-only STFT MSE loss.

    - Converts predicted/GT binaural waveforms to mono by averaging L/R.
    - Computes STFT magnitude for mono signals.
    - Uses MSE (optionally on log1p magnitudes).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.l2_loss = nn.MSELoss()

        # STFT params aligned with Audio3DGS forward / metrics
        self.fft_size = 512
        self.hop_size = 160
        self.win_length = 400
        self.register_buffer("window", torch.hamming_window(self.win_length), persistent=False)

        try:
            self.use_log_mag_loss = bool(getattr(cfg.train, "use_log_mag_loss", True))
        except Exception:
            self.use_log_mag_loss = True

    def forward(self, pred_wav, gt_wav, out_pred_wav_0=None):
        """
        Args:
            pred_wav: [B, 2, T] predicted (mono duplicated) audio
            gt_wav:   [B, 2, T] ground-truth binaural audio
        Returns:
            dict: {'total_loss', 'mono_loss'}
        """
        pred_wav = torch.nan_to_num(pred_wav, nan=0.0, posinf=0.0, neginf=0.0)
        gt_wav = torch.nan_to_num(gt_wav, nan=0.0, posinf=0.0, neginf=0.0)

        # Collapse to mono by averaging L/R
        pred_mono = pred_wav.mean(dim=1)  # [B, T]
        gt_mono = gt_wav.mean(dim=1)      # [B, T]

        pred_spec = stft(pred_mono, self.fft_size, self.hop_size, self.win_length, self.window)
        gt_spec = stft(gt_mono, self.fft_size, self.hop_size, self.win_length, self.window)

        if self.use_log_mag_loss:
            pred_spec = torch.log1p(torch.clamp(pred_spec, min=0.0))
            gt_spec = torch.log1p(torch.clamp(gt_spec, min=0.0))

        loss_mono = self.l2_loss(pred_spec, gt_spec)
        return {
            "total_loss": loss_mono,
            "mono_loss": loss_mono,
        }

