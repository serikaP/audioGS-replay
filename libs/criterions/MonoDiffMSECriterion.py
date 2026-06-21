import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.criterions.Criterion_2 import stft


class MonoDiffMSECriterion(nn.Module):
    """
    Simple mono/diff STFT MSE loss.

    - Decomposes binaural waveforms into mono/diff.
    - Computes STFT magnitude for each.
    - Uses MSE on (optionally log1p) magnitudes.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.l2_loss = nn.MSELoss()

        # STFT params aligned with Audio3DGS forward / metrics
        self.fft_size = 512
        self.hop_size = 160
        self.win_length = 400
        self.register_buffer(
            "window", torch.hamming_window(self.win_length), persistent=False
        )

        # Whether to apply log1p before MSE (consistent with existing Criterion_2)
        try:
            self.use_log_mag_loss = bool(getattr(cfg.train, "use_log_mag_loss", True))
        except Exception:
            self.use_log_mag_loss = True

        # Optional phase-difference loss (only used when geometry-guided phase
        # is enabled). When use_geom_phase is True, we allow an additional
        # supervision term on STFT phase differences between prediction and GT.
        model_cfg = getattr(cfg, "model", object())
        self.use_geom_phase = bool(getattr(model_cfg, "use_geom_phase", False))
        try:
            self.phase_loss_weight = float(
                getattr(cfg.train, "phase_loss_weight", 0.0)
            )
        except Exception:
            self.phase_loss_weight = 0.0
        # Backward-compat: if geometry-guided phase is enabled but the config
        # does not define `phase_loss_weight`, use a small default to gently
        # regularize phase.
        if self.use_geom_phase and (not hasattr(getattr(cfg, "train", object()), "phase_loss_weight")):
            self.phase_loss_weight = 0.05

        # Which signals to apply phase supervision on:
        #   - "mono_diff" (default): phase loss on mono/diff STFTs (legacy)
        #   - "lr": phase loss on left/right STFTs
        try:
            self.phase_loss_target = str(
                getattr(cfg.train, "phase_loss_target", "mono_diff")
            ).strip().lower()
        except Exception:
            self.phase_loss_target = "mono_diff"
        if self.phase_loss_target not in ("mono_diff", "lr"):
            self.phase_loss_target = "mono_diff"

    def forward(self, pred_wav, gt_wav, out_pred_wav_0=None):
        """
        Args:
            pred_wav: [B, 2, T] predicted binaural audio
            gt_wav:   [B, 2, T] ground-truth binaural audio
        Returns:
            dict with keys: total_loss, mono_loss, diff_loss
        """
        pred_wav = torch.nan_to_num(pred_wav, nan=0.0, posinf=0.0, neginf=0.0)
        gt_wav = torch.nan_to_num(gt_wav, nan=0.0, posinf=0.0, neginf=0.0)

        pred_mono = pred_wav[:, 0, :] + pred_wav[:, 1, :]
        pred_diff = pred_wav[:, 0, :] - pred_wav[:, 1, :]

        gt_mono = gt_wav[:, 0, :] + gt_wav[:, 1, :]
        gt_diff = gt_wav[:, 0, :] - gt_wav[:, 1, :]

        pred_mono_spec = stft(
            pred_mono, self.fft_size, self.hop_size, self.win_length, self.window
        )
        gt_mono_spec = stft(
            gt_mono, self.fft_size, self.hop_size, self.win_length, self.window
        )
        pred_diff_spec = stft(
            pred_diff, self.fft_size, self.hop_size, self.win_length, self.window
        )
        gt_diff_spec = stft(
            gt_diff, self.fft_size, self.hop_size, self.win_length, self.window
        )

        # if self.use_log_mag_loss:
        #     pred_mono_spec = torch.log1p(torch.clamp(pred_mono_spec, min=0.0))
        #     gt_mono_spec = torch.log1p(torch.clamp(gt_mono_spec, min=0.0))
        #     pred_diff_spec = torch.log1p(torch.clamp(pred_diff_spec, min=0.0))
        #     gt_diff_spec = torch.log1p(torch.clamp(gt_diff_spec, min=0.0))
        if self.use_log_mag_loss:
            pred_mono_spec = torch.log(torch.clamp(pred_mono_spec, min=1e-7))
            gt_mono_spec   = torch.log(torch.clamp(gt_mono_spec,   min=1e-7))
            pred_diff_spec = torch.log(torch.clamp(pred_diff_spec, min=1e-7))
            gt_diff_spec = torch.log(torch.clamp(gt_diff_spec, min=1e-7))

        loss_mono = self.l2_loss(pred_mono_spec, gt_mono_spec)
        loss_diff = self.l2_loss(pred_diff_spec, gt_diff_spec)

        diff_w = float(getattr(self.cfg.train, "diff_weight", 1.0))
        total_loss = loss_mono + diff_w * loss_diff

        # Optional phase MSE: when geometry-guided phase is used, we add
        # supervision on STFT phase differences. We wrap phase
        # differences to (-pi, pi] to avoid 2π ambiguity and then apply MSE.
        phase_loss = pred_wav.new_tensor(0.0)
        if self.use_geom_phase and self.phase_loss_weight > 0.0:
            try:
                window = self.window.to(pred_wav.device)

                def _phase_mse(px: torch.Tensor, gx: torch.Tensor) -> torch.Tensor:
                    # Phase difference wrapped to (-pi, pi]
                    dphi = torch.angle(px) - torch.angle(gx)
                    dphi = torch.nan_to_num(dphi, nan=0.0, posinf=0.0, neginf=0.0)
                    dphi_wrapped = torch.atan2(torch.sin(dphi), torch.cos(dphi))
                    return (dphi_wrapped ** 2).mean()

                if getattr(self, "phase_loss_target", "mono_diff") == "lr":
                    pred_L = pred_wav[:, 0, :]
                    pred_R = pred_wav[:, 1, :]
                    gt_L = gt_wav[:, 0, :]
                    gt_R = gt_wav[:, 1, :]

                    pred_L_c = torch.stft(
                        pred_L,
                        n_fft=self.fft_size,
                        hop_length=self.hop_size,
                        win_length=self.win_length,
                        window=window,
                        return_complex=True,
                        pad_mode="constant",
                    )
                    gt_L_c = torch.stft(
                        gt_L,
                        n_fft=self.fft_size,
                        hop_length=self.hop_size,
                        win_length=self.win_length,
                        window=window,
                        return_complex=True,
                        pad_mode="constant",
                    )
                    pred_R_c = torch.stft(
                        pred_R,
                        n_fft=self.fft_size,
                        hop_length=self.hop_size,
                        win_length=self.win_length,
                        window=window,
                        return_complex=True,
                        pad_mode="constant",
                    )
                    gt_R_c = torch.stft(
                        gt_R,
                        n_fft=self.fft_size,
                        hop_length=self.hop_size,
                        win_length=self.win_length,
                        window=window,
                        return_complex=True,
                        pad_mode="constant",
                    )

                    phase_loss = 0.5 * (
                        _phase_mse(pred_L_c, gt_L_c) + _phase_mse(pred_R_c, gt_R_c)
                    )
                else:
                    # Complex STFTs on mono/diff (legacy behavior)
                    pred_mono_c = torch.stft(
                        pred_mono,
                        n_fft=self.fft_size,
                        hop_length=self.hop_size,
                        win_length=self.win_length,
                        window=window,
                        return_complex=True,
                        pad_mode="constant",
                    )
                    gt_mono_c = torch.stft(
                        gt_mono,
                        n_fft=self.fft_size,
                        hop_length=self.hop_size,
                        win_length=self.win_length,
                        window=window,
                        return_complex=True,
                        pad_mode="constant",
                    )
                    pred_diff_c = torch.stft(
                        pred_diff,
                        n_fft=self.fft_size,
                        hop_length=self.hop_size,
                        win_length=self.win_length,
                        window=window,
                        return_complex=True,
                        pad_mode="constant",
                    )
                    gt_diff_c = torch.stft(
                        gt_diff,
                        n_fft=self.fft_size,
                        hop_length=self.hop_size,
                        win_length=self.win_length,
                        window=window,
                        return_complex=True,
                        pad_mode="constant",
                    )

                    phase_loss = 0.5 * (
                        _phase_mse(pred_mono_c, gt_mono_c)
                        + _phase_mse(pred_diff_c, gt_diff_c)
                    )
                total_loss = total_loss + self.phase_loss_weight * phase_loss
            except Exception:
                phase_loss = pred_wav.new_tensor(0.0)

        return {
            "total_loss": total_loss,
            "mono_loss": loss_mono,
            "diff_loss": loss_diff,
            "phase_loss": phase_loss * self.phase_loss_weight
        }
