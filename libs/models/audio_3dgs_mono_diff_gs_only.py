"""
Mono/Diff Audio 3DGS without U-Net (GS-only)

- Based on Audio3DGSMonoDiff (two SH fields: mono and diff).
- Removes the DualBranchAudioUNet; SH fields directly modulate the source magnitude
  to form mono/diff spectrograms, which are then combined into L/R and inverted
  via iSTFT.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.models.audio_3dgs_mono_diff import Audio3DGSMonoDiff
from libs.utils.sh_utils import eval_sh


class Audio3DGSMonoDiffGSOnly(Audio3DGSMonoDiff):
    """
    Two-field GS-only model:
      - Uses _sh_mono / _sh_diff (inherited from Audio3DGSMonoDiff).
      - Does NOT use the U-Net renderer; instead:
          * mono_mag    = SH_mono(f,t,dir) * |STFT(near.wav)|
          * diff_envelope = SH_diff(f,t,dir)                 (signed)
          * L/R mag     = ReLU( mono_mag * (1 ± diff_envelope) )
          * phase     = from source STFT (L/R or mono)
          * iSTFT → binaural waveform
    """

    def __init__(self, cfg, freq_num: int = 257, time_num: int = 348):
        super().__init__(cfg, freq_num=freq_num, time_num=time_num)

        # When True, SH mono/diff fields directly output absolute mono/diff
        # magnitudes on the STFT grid, instead of predicting a multiplicative
        # mask on |STFT(source)|.
        #
        # In this mode we will:
        #   1) Initialize the mono SH DC term from the source STFT magnitude
        #      of the first training batch (initialize_from_batch).
        #   2) Keep higher-order SH coefficients at zero so that directionality
        #      is learned from scratch.
        #   3) In forward(), interpret the SH fields (after distance
        #      attenuation g_ft) as absolute mono/diff magnitudes and
        #      drop the final multiplication by |STFT(source)|.
        model_cfg = getattr(cfg, "model", object())
        self.use_abs_mag_output = bool(
            getattr(model_cfg, "abs_mag_output", False)
        )
        # Optional per-frequency distance attenuation (mask-style)
        self.use_freq_atten = bool(
            getattr(getattr(cfg, "model", object()), "use_freq_atten", False)
        )
        # Whether to use energy-weighted per-frequency distance (True) or
        # simple time-average distance per frequency band (False).
        # NOTE: originally only used for per-frequency attenuation; after we
        # switch to per-point attenuation (F,T), this flag is kept for
        # backward compatibility but is no longer used in the new scheme.
        self.use_energy_weighted_freq_dist = bool(
            getattr(
                getattr(cfg, "model", object()),
                "use_energy_weighted_freq_dist",
                True,
            )
        )
        try:
            self.freq_atten_alpha = float(
                getattr(getattr(cfg, "model", object()), "freq_atten_alpha", 1.0)
            )
        except Exception:
            self.freq_atten_alpha = 1.0

        # Optional per-point distance attenuation exponent alpha_i.
        # When enabled (and Hopkins attenuation is disabled), each TF-bin / point
        # has its own learnable exponent alpha_i in:
        #   g_i(r) ∝ 1 / (r + eps)^{alpha_i}
        self.use_pointwise_alpha = bool(
            getattr(getattr(cfg, "model", object()), "use_pointwise_alpha", False)
        )
        if self.use_pointwise_alpha:
            # One alpha parameter per point; we map raw values through tanh+1 so
            # that alpha starts at 1 (raw=0) and stays in (0, 2).
            self._alpha_param = nn.Parameter(torch.zeros(self.n_points))
        else:
            self._alpha_param = None

        # Optional: geometry-guided explicit phase shift (ITD) based on 3D point
        # positions and a rigid-sphere head model. When enabled, we compute a
        # per-point reference azimuth (in the listener/head frame) once, then
        # use a frequency-dependent ITD model to derive relative phase offsets
        # at novel viewpoints.
        self.use_geom_phase = bool(
            getattr(getattr(cfg, "model", object()), "use_geom_phase", False)
        )
        # Trainer can pass a per-sample reference pose (e.g., input_cam_pose) for
        # viewpoint-source audio so that phase is corrected relative to the
        # actual source viewpoint rather than an arbitrary first batch.
        self.supports_ref_cam_pose = True
        try:
            self.sound_speed = float(
                getattr(getattr(cfg, "model", object()), "sound_speed", 343.0)
            )
        except Exception:
            self.sound_speed = 343.0
        try:
            self.head_width = float(getattr(getattr(cfg, "model", object()), "head_width", 0.18))
        except Exception:
            self.head_width = 0.18
        # Rigid-sphere head radius a (meters). If not provided, fall back to
        # half the inter-aural distance.
        try:
            self.head_radius = float(
                getattr(getattr(cfg, "model", object()), "head_radius", 0.5 * self.head_width)
            )
        except Exception:
            self.head_radius = 0.5 * self.head_width
        # Frequency thresholds for the hybrid ITD model (Hz).
        try:
            self.geom_phase_f_low_hz = float(
                getattr(getattr(cfg, "model", object()), "geom_phase_f_low_hz", 500.0)
            )
        except Exception:
            self.geom_phase_f_low_hz = 500.0
        try:
            self.geom_phase_f_high_hz = float(
                getattr(getattr(cfg, "model", object()), "geom_phase_f_high_hz", 1500.0)
            )
        except Exception:
            self.geom_phase_f_high_hz = 1500.0
        # Reference per-point signed azimuth angles (radians) for geometry-guided phase.
        self.register_buffer("ref_theta", torch.zeros(self.n_points))
        self._geom_phase_initialized = False

        # Optional: multiplicative residual on geometry-guided ITD, conditioned on
        # viewing direction via a small SH field (deg=1), shared across time per
        # frequency bin.
        #
        # For each frequency f we learn a tiny SH coefficient vector c_f (deg=1,
        # 4 coeffs) and evaluate a direction-conditioned scalar:
        #   s(f,t, d) = SH(c_f, d),
        # where d is the signed view direction in the head frame.
        #
        # We then scale the rigid-sphere ITD magnitudes:
        #   ΔT_total = ΔT_phys * (1 + tanh(alpha*s) * scale_limit)
        # and compute the phase correction using the *difference* between the
        # target and reference viewpoints:
        #   (ΔT_total - ΔT_total_ref) = ΔT_phys*(1+s_tgt) - ΔT_ref*(1+s_ref),
        # which does not cancel and still equals 0 when target==reference.
        model_cfg = getattr(cfg, "model", object())
        self.use_geom_phase_residual = bool(
            getattr(model_cfg, "use_geom_phase_residual", True)
        ) and bool(self.use_geom_phase)
        try:
            self.geom_phase_residual_alpha = float(
                getattr(model_cfg, "geom_phase_residual_alpha", 1.0)
            )
        except Exception:
            self.geom_phase_residual_alpha = 1.0
        try:
            self.geom_phase_residual_scale_limit = float(
                getattr(model_cfg, "geom_phase_residual_scale_limit", 0.2)
            )
        except Exception:
            self.geom_phase_residual_scale_limit = 0.2
        # Degree for the residual SH field; keep it small for stability.
        self.geom_phase_residual_sh_degree = 1
        self.geom_phase_residual_sh_coeffs = (self.geom_phase_residual_sh_degree + 1) ** 2
        if self.use_geom_phase_residual and self.geom_phase_residual_scale_limit > 0.0:
            # SH coefficients per frequency bin (shared across time).
            # Shape: [F, 1, 4] when degree=1. Initialized to 0 => no correction.
            self._sh_phase = nn.Parameter(
                torch.zeros(self.freq_num, 1, self.geom_phase_residual_sh_coeffs)
            )
        else:
            self._sh_phase = None

        # Optional: Hopkins–Stryker style attenuation (direct + reverberant field)
        #   L_p = L_w + 10 log10(Q/(4 pi r^2) + 4/R_c)
        # Here we only use the relative energy term E(r) = Q/(4 pi r^2) + 4/R_c.
        self.use_hopkins_atten = bool(
            getattr(getattr(cfg, "model", object()), "use_hopkins_atten", False)
        )
        self.hopkins_learn_params = bool(
            getattr(getattr(cfg, "model", object()), "hopkins_learn_params", True)
        )

        if self.use_hopkins_atten:
            # Q: directivity factor. Initialize around 2 (e.g., source near floor).
            if self.hopkins_learn_params:
                self.Q_param = nn.Parameter(torch.tensor(2.0))
                # log_Rc_param stores log(R_c) so that R_c = exp(log_Rc_param) > 0.
                self.log_Rc_param = nn.Parameter(torch.zeros(1))
            else:
                self.register_buffer("Q_param", torch.tensor(2.0))
                self.register_buffer("log_Rc_param", torch.zeros(1))
        else:
            self.Q_param = None
            self.log_Rc_param = None

        # Cache for RT60 estimator (loaded lazily) and init flag for Hopkins params
        self._rt60_estimator = None
        self._hopkins_initialized = False

        # Cache dataset sampling rate for RT60 estimation; fall back to 16 kHz
        try:
            self.sr = int(getattr(getattr(cfg, "dataset", object()), "sr", 16000))
        except Exception:
            self.sr = 16000

        # Reference per-point attenuation g_ft(ref) for normalization
        # Shape: [F, T]; filled in initialize_from_batch().
        ref_g = torch.ones(freq_num, time_num)
        self.register_buffer("ref_freq_g", ref_g)

        # Optional static source STFT (magnitude + phase) cached per model.
        # When use_static_source=True and forward() is called with
        # source_audio=None, these buffers will be used instead of computing
        # STFT(source_audio) on the fly.
        self.use_static_source = False
        # Placeholders; real values are set in initialize_from_batch().
        self.register_buffer("static_source_mag", torch.zeros(1, freq_num, 1))
        self.register_buffer("static_phase_L", torch.zeros(1, freq_num, 1))
        self.register_buffer("static_phase_R", torch.zeros(1, freq_num, 1))

    # ------------------------------------------------------------------
    # 可选保存 source audio 的静态 STFT 以供推理时复用
    # ------------------------------------------------------------------
    def initialize_from_batch(
        self,
        batch: dict,
        device: torch.device,
        update_ref_g: bool = True,
    ):
        """
        Initialize model from the first training batch:
          1) Cache a static source STFT (magnitude + phase) that can be reused
             at inference time when source_audio is omitted.
          2) Optionally compute reference per-frequency attenuation g_f(ref)
             when freq_atten is enabled, so that g_f_norm ≈ 1 at reference.

        Args:
            batch: dict with keys "cam_pose" and "source_audio".
            device: target torch.device.
            update_ref_g: when True (training-time init), also update
                self.ref_freq_g using the batch cam_pose; when False
                (e.g., test-time static-source init), only refresh the
                cached static STFT and keep ref_freq_g as loaded from
                checkpoint.
        """
        try:
            cam_pose = batch["cam_pose"][0:1].to(device)         # [1, 12] (target)
            input_cam_pose = None
            try:
                input_cam_pose = batch.get("input_cam_pose", None)
                if isinstance(input_cam_pose, torch.Tensor):
                    input_cam_pose = input_cam_pose[0:1].to(device)
            except Exception:
                input_cam_pose = None
            source_audio = batch["source_audio"][0:1].to(device) # [1, C, T]
        except Exception as e:
            print(f"[Audio3DGSMonoDiffGSOnly] initialize_from_batch failed: {e}")
            return

        with torch.no_grad():
            # 1) Cache static source STFT (mag + phase) from the first sample
            n_fft = 512
            hop_length = 160
            window_length = 400
            torch_window = torch.hamming_window(window_length).to(device)

            if source_audio.dim() > 2 and source_audio.shape[1] > 1:
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
                mag_L = spec_L.abs()
                mag_R = spec_R.abs()
                source_mag = 0.5 * (mag_L + mag_R)  # [1, F, T_spec]
                phase_L = torch.angle(spec_L)
                phase_R = torch.angle(spec_R)
            else:
                source_mono = source_audio.squeeze(0)
                source_spec = torch.stft(
                    source_mono,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=window_length,
                    window=torch_window,
                    return_complex=True,
                )
                source_mag = source_spec.abs().unsqueeze(0)   # [1, F, T_spec]
                phase = torch.angle(source_spec).unsqueeze(0)  # [1, F, T_spec]
                phase_L = phase
                phase_R = phase

            source_mag = torch.nan_to_num(
                source_mag, nan=0.0, posinf=0.0, neginf=0.0
            )
            phase_L = torch.nan_to_num(
                phase_L, nan=0.0, posinf=0.0, neginf=0.0
            )
            phase_R = torch.nan_to_num(
                phase_R, nan=0.0, posinf=0.0, neginf=0.0
            )

            # Overwrite static STFT buffers (used when source_audio is None).
            self.static_source_mag = source_mag.to(device)
            self.static_phase_L = phase_L.to(device)
            self.static_phase_R = phase_R.to(device)

            # ------------------------------------------------------------------
            # Optional: initialize mono / diff SH fields from source magnitude
            #           when using absolute-magnitude output mode.
            # ------------------------------------------------------------------
            if getattr(self, "use_abs_mag_output", False):
                try:
                    import math as _math
                    import torch.nn.functional as _F

                    # Resize |STFT(source)| to the model's (freq_num, time_num) grid.
                    # source_mag: [1, F_spec, T_spec]
                    mag_1ft = source_mag[0]  # [F_spec, T_spec]
                    if mag_1ft.shape[-2:] != (self.freq_num, self.time_num):
                        mag_resized = _F.interpolate(
                            mag_1ft.unsqueeze(0).unsqueeze(0),
                            size=(self.freq_num, self.time_num),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0).squeeze(0)
                    else:
                        mag_resized = mag_1ft

                    mag_resized = torch.nan_to_num(
                        mag_resized, nan=0.0, posinf=0.0, neginf=0.0
                    )

                    # Target mono magnitude per (f,t) on the model grid.
                    # In abs_mag_output mode we interpret the SH mono field
                    # directly as absolute magnitude (after distance
                    # attenuation). Therefore we no longer normalize the
                    # reference |STFT(source)| into [0, 1]; instead we keep
                    # its natural scale and invert the softplus mapping. Note
                    # eval_sh multiplies the DC term by C0=0.28209479..., so
                    # we first divide by C0 to avoid a baked-in amplitude
                    # shrink.
                    A_ft = mag_resized.clamp(min=0.0)  # [F_model, T_model]
                    C0 = 0.28209479177387814  # SH Y00 constant
                    A_target = A_ft / C0

                    # Mapping depends on activation used in _eval_sh_field:
                    #   - abs_mag_output=True uses ReLU (no softplus/scale)
                    #   - otherwise uses softplus / log(2)
                    if getattr(self, "use_abs_mag_output", False):
                        # eval_sh (DC only) gives: sh_values ≈ C0 * c_ft.
                        # ReLU leaves positive values unchanged, so set
                        # c_ft = A_ft / C0 to match target magnitude.
                        c_ft = A_target  # already A_ft / C0
                    else:
                        # Original mask mode: invert softplus mapping.
                        log2 = _math.log(2.0)
                        eps = 1e-6
                        c_ft = torch.log(torch.expm1(A_target * log2) + eps)

                    # Flatten to [N_points] and assign to mono DC term.
                    c_flat = c_ft.reshape(-1)
                    if hasattr(self, "_sh_mono") and self._sh_mono.shape[0] == c_flat.shape[0]:
                        self._sh_mono[:, 0, 0] = c_flat.to(self._sh_mono.device, dtype=self._sh_mono.dtype)
                        # Zero-out higher-order mono coefficients.
                        if self._sh_mono.shape[-1] > 1:
                            self._sh_mono[:, 0, 1:] = 0.0
                    # Diff field starts from zero in abs-mag mode.
                    if hasattr(self, "_sh_diff"):
                        self._sh_diff.zero_()

                    print(
                        "[Audio3DGSMonoDiffGSOnly] Initialized SH mono DC from "
                        "source STFT magnitude for abs_mag_output mode."
                    )
                except Exception as _e:
                    # Fail soft: keep random/zero SH init.
                    print(
                        f"[Audio3DGSMonoDiffGSOnly] SH init from source mag failed: {_e}"
                    )

            # Use input_cam_pose as reference when source_audio is taken from a
            # fixed input viewpoint; otherwise fall back to target cam_pose.
            ref_pose = input_cam_pose if self._pose_is_valid(input_cam_pose) else cam_pose

            # Initialize reference azimuth for geometry-guided phase (once).
            if self.use_geom_phase and (not self._geom_phase_initialized):
                try:
                    self._init_geom_phase_reference(ref_pose, device)
                    print(
                        "[Audio3DGSMonoDiffGSOnly] Initialized reference azimuth "
                        "for geometry-guided phase."
                    )
                except Exception as _e:
                    print(
                        f"[Audio3DGSMonoDiffGSOnly] Failed to init geom-phase reference: {_e}"
                    )

            # 2) Optionally compute reference per-point attenuation g_ft(ref)
            #    when freq_atten is enabled, so that g_ft_norm ≈ 1 at reference.
            #    During training we want to set this from the source viewpoint,
            #    but at test-time (static-source mode) we keep the checkpoint
            #    value so that distance attenuation stays tied to the source
            #    audio viewpoint instead of the target/test viewpoint.
            if (not update_ref_g) or (not self.use_freq_atten):
                return

            # If using Hopkins-style attenuation, try to initialize R_c from RT60
            if self.use_hopkins_atten:
                self._maybe_init_hopkins_from_rt60(source_audio, device)

            # Geometry: point→mic distances in world units (normalize at source viewpoint)
            rel_world_norm, _, _ = self.compute_relative_positions(ref_pose)  # [1, N, 3]
            distances = (rel_world_norm * float(self.max_norm)).norm(dim=-1)  # [1, N]
            distances_ft = distances.view(1, self.freq_num, self.time_num)    # [1, F, T_model]

            eps = 1e-6
            # Per-point distance attenuation on the model grid [F_model, T_model]
            if self.use_hopkins_atten and (self.Q_param is not None) and (self.log_Rc_param is not None):
                # Hopkins–Stryker style reference energy:
                #   E_ref(r) = Q/(4 pi r^2) + 4/R_c
                Q = torch.clamp(self.Q_param, min=0.1)
                Rc = torch.exp(self.log_Rc_param)

                r2 = distances_ft ** 2 + eps
                direct_ref = Q / (4.0 * math.pi * r2)
                rever_ref = 4.0 / (Rc + eps)
                E_ref = direct_ref + rever_ref  # [1, F, T]
                # Store sqrt(E_ref) so that later g(r) = sqrt(E_curr / E_ref)
                g_ft_ref = torch.sqrt(torch.clamp(E_ref, min=0.0))
            else:
                if self.use_pointwise_alpha and (self._alpha_param is not None):
                    # Per-point exponent alpha_i mapped via tanh+1 (range 0~2).
                    alpha_ft = (torch.tanh(self._alpha_param) + 1.0).view(
                        1, self.freq_num, self.time_num
                    )
                    alpha_ft = torch.clamp(alpha_ft, min=0.0, max=2.0)
                    g_ft_ref = (1.0 / (distances_ft + eps)) ** alpha_ft  # [1, F, T_model]
                else:
                    g_ft_ref = (1.0 / (distances_ft + eps)) ** float(self.freq_atten_alpha)  # [1, F, T_model]
            # Store reference attenuation on model grid (drop batch dim)
            self.ref_freq_g.data = g_ft_ref[0].to(self.ref_freq_g.device)
            print(
                "[Audio3DGSMonoDiffGSOnly] Initialized per-point ref_freq_g from reference batch."
            )

    # ------------------------------------------------------------------
    # RT60 estimator helpers (used to initialize Hopkins parameters)
    # ------------------------------------------------------------------
    def _load_rt60_estimator(self, device: torch.device):
        """Load the RT60 estimator (Audio-only VisualNet) once and cache it."""
        if self._rt60_estimator is not None:
            return self._rt60_estimator
        try:
            from libs.models.vigas.visual_net import VisualNet
            estimator = VisualNet(use_rgb=False, use_depth=False, use_audio=True)
            weights = "data/avcloud_data/models/rt60_estimator.pth"
            if os.path.exists(weights):
                checkpoint = torch.load(weights, map_location="cpu")
                state = checkpoint.get("predictor", checkpoint)
                estimator.load_state_dict(state)
                estimator.to(device=device).eval()
                self._rt60_estimator = estimator
                print(f"[Audio3DGSMonoDiffGSOnly] Loaded RT60 estimator from {weights}")
            else:
                print(
                    f"[Audio3DGSMonoDiffGSOnly] RT60 estimator not found at {weights}; "
                    "using default Rc init."
                )
                self._rt60_estimator = None
        except Exception as e:
            print(f"[Audio3DGSMonoDiffGSOnly] Failed to load RT60 estimator: {e}")
            self._rt60_estimator = None
        return self._rt60_estimator

    def _estimate_rt60_from_source(
        self, source_audio: torch.Tensor, device: torch.device
    ) -> float:
        """
        Estimate RT60 of the current source audio using the AV-Cloud RT60 estimator.

        Args:
            source_audio: [1, C, T] tensor on `device`.
        Returns:
            Scalar RT60 estimate in seconds (rough), with basic clamping.
        """
        estimator = self._load_rt60_estimator(device)
        if estimator is None:
            # Reasonable fallback; will be refined by training anyway.
            return 0.5

        try:
            x = source_audio[0]  # [C, T] or [T]
            if x.dim() > 1:
                x = x.mean(dim=0)  # mono
            x = x.unsqueeze(0).to(device)  # [1, T]

            n_fft = 512
            hop_length = 160
            window_length = 400
            window = torch.hamming_window(window_length, device=device)
            stft = torch.stft(
                x,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=window_length,
                window=window,
                pad_mode="constant",
                return_complex=True,
            )
            spec = torch.log1p(stft.abs()).unsqueeze(1)  # [1, 1, F, T]
            with torch.no_grad():
                rt60_pred = estimator(spec.float())
            # Estimator typically outputs [B, 1]; take mean and clamp to [0.1, 3.0] s
            rt60_val = float(rt60_pred.mean().clamp(0.1, 3.0).item())
            return rt60_val
        except Exception as e:
            print(f"[Audio3DGSMonoDiffGSOnly] RT60 estimation failed: {e}")
            return 0.5

    def _maybe_init_hopkins_from_rt60(
        self, source_audio: torch.Tensor, device: torch.device
    ) -> None:
        """
        Initialize Hopkins parameters (mainly R_c) from an RT60 estimate of source_audio.

        We use the empirical relation R_c ∝ 1 / T60 to set an initial magnitude for R_c,
        then let training refine Q and R_c jointly.
        """
        if (
            (not self.use_hopkins_atten)
            or (self.log_Rc_param is None)
            or self._hopkins_initialized
        ):
            return

        rt60 = self._estimate_rt60_from_source(source_audio, device)

        # Rough mapping: longer RT60 -> smaller room constant (stronger reverberant field).
        # Scale factor is heuristic; training will refine log_Rc_param.
        # We keep Rc in a reasonable range [10, 500] m^2.
        Rc_init = 100.0 / max(rt60, 0.1)
        Rc_init = float(max(10.0, min(Rc_init, 500.0)))

        Rc_tensor = torch.tensor(Rc_init, device=self.log_Rc_param.device, dtype=self.log_Rc_param.dtype)
        if isinstance(self.log_Rc_param, nn.Parameter):
            self.log_Rc_param.data = torch.log(Rc_tensor)
        else:
            self.log_Rc_param = torch.log(Rc_tensor)

        self._hopkins_initialized = True
        print(
            f"[Audio3DGSMonoDiffGSOnly] Initialized Hopkins params from RT60≈{rt60:.2f}s, "
            f"Rc_init≈{Rc_init:.1f}"
        )

    # ------------------------------------------------------------------
    # Geometry-guided phase helpers (ITD based on 3D point cloud)
    # ------------------------------------------------------------------
    @staticmethod
    def _pose_is_valid(pose: torch.Tensor) -> bool:
        try:
            if not isinstance(pose, torch.Tensor):
                return False
            if pose.numel() == 0:
                return False
            # In our datasets, "no pose" is encoded as all-zeros.
            return bool(torch.isfinite(pose).all().item()) and bool(pose.abs().sum().item() > 1e-6)
        except Exception:
            return False

    @staticmethod
    def _resize_angle(angle: torch.Tensor, size) -> torch.Tensor:
        """
        Resize an angle field by interpolating sin/cos to avoid wrap-around artifacts.
        angle: [B, F, T]
        size: (F_out, T_out)
        """
        if angle.shape[-2:] == tuple(size):
            return angle
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        sin_r = F.interpolate(sin.unsqueeze(1), size=size, mode="bilinear", align_corners=False).squeeze(1)
        cos_r = F.interpolate(cos.unsqueeze(1), size=size, mode="bilinear", align_corners=False).squeeze(1)
        return torch.atan2(sin_r, cos_r)

    def _compute_signed_azimuth(self, cam_pose: torch.Tensor) -> torch.Tensor:
        """
        Compute per-point signed azimuth angles (radians) in the listener/head frame.

        We use the camera frame as the head frame:
          - +x: right
          - +y: up
          - forward: -z (NeRF/OpenGL convention for camera_to_world poses)

        Returns:
            theta: [B, N] signed azimuth in [-pi, pi], where theta>0 means source on the right.
        """
        # mic->point in camera frame: dir_cam = -(point->mic) = -rel_cam_norm
        _, _, rel_cam_norm = self.compute_relative_positions(cam_pose)
        dir_cam = -rel_cam_norm  # [B, N, 3]
        theta = torch.atan2(dir_cam[..., 0], -dir_cam[..., 2])
        theta = torch.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
        return theta

    def _itd_delay(self, freqs_hz: torch.Tensor, theta_abs: torch.Tensor) -> torch.Tensor:
        """
        Frequency-dependent ITD magnitude ΔT(f, θ) (seconds) for a rigid sphere.

        Args:
            freqs_hz:  [F] or [1,F,1] frequency in Hz
            theta_abs: [B,F,T] absolute azimuth in radians
        Returns:
            dt: [B,F,T] non-negative ITD magnitude in seconds
        """
        c = float(getattr(self, "sound_speed", 343.0))
        a = float(getattr(self, "head_radius", 0.5 * float(getattr(self, "head_width", 0.18))))
        f_low = float(getattr(self, "geom_phase_f_low_hz", 500.0))
        f_high = float(getattr(self, "geom_phase_f_high_hz", 1500.0))

        if freqs_hz.dim() == 1:
            f = freqs_hz.view(1, -1, 1)
        else:
            f = freqs_hz

        # Ensure azimuth is within [0, pi] for the rigid-sphere model.
        theta_abs = torch.clamp(theta_abs, min=0.0, max=float(math.pi))

        # Low/high frequency models (Aaronson & Hartmann-style hybrid).
        dt_low = (3.0 * a / c) * torch.sin(theta_abs)
        # High-frequency approximation is piecewise: for sources behind the head
        # (theta > pi/2), ITD should decrease back to 0 at theta=pi.
        dt_high_front = (a / c) * (theta_abs + torch.sin(theta_abs))
        dt_high_back = (a / c) * ((float(math.pi) - theta_abs) + torch.sin(theta_abs))
        dt_high = torch.where(theta_abs <= (0.5 * float(math.pi)), dt_high_front, dt_high_back)

        # Linear interpolation in the mid band.
        denom = max(f_high - f_low, 1.0)
        w = ((f - f_low) / denom).clamp(0.0, 1.0)  # [1,F,1]
        dt_mid = (1.0 - w) * dt_low + w * dt_high

        dt = torch.where(f < f_low, dt_low, torch.where(f > f_high, dt_high, dt_mid))
        dt = torch.nan_to_num(dt, nan=0.0, posinf=0.0, neginf=0.0)
        return dt

    def _init_geom_phase_reference(self, cam_pose: torch.Tensor, device: torch.device):
        """
        Initialize reference per-point azimuth angles (ref_theta) for geometry-guided
        phase, using the first training batch (reference viewpoint).
        """
        if self._geom_phase_initialized:
            return

        theta_ref = self._compute_signed_azimuth(cam_pose.to(device))  # [1, N]
        self.ref_theta.data = theta_ref[0].to(self.ref_theta.device)
        self._geom_phase_initialized = True

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Backward-compat: older checkpoints stored ear-distance based geom-phase
        buffers (ref_dL/ref_dR/omega). Newer checkpoints store ref_theta.
        """
        sd = dict(state_dict)
        # Drop deprecated buffers if present.
        for k in ("ref_dL", "ref_dR", "omega"):
            if k in sd:
                sd.pop(k)
        # Ensure new buffer exists when loading older checkpoints with strict=True.
        if "ref_theta" not in sd and hasattr(self, "ref_theta"):
            sd["ref_theta"] = self.ref_theta
        # Drop older residual parameter if present (legacy multiplicative residual).
        if "_geom_phase_scale_param" in sd:
            sd.pop("_geom_phase_scale_param")
        # Ensure SH-phase residual exists when loading older checkpoints with strict=True.
        if "_sh_phase" not in sd and hasattr(self, "_sh_phase") and isinstance(getattr(self, "_sh_phase"), torch.Tensor):
            sd["_sh_phase"] = self._sh_phase
        return super().load_state_dict(sd, strict=strict)

    def forward(
        self,
        cam_pose: torch.Tensor,
        source_audio: torch.Tensor = None,
        is_val: bool = False,
        return_mag: bool = False,
        return_masks: bool = False,
        ref_cam_pose: torch.Tensor = None,
    ):
        """
        Args:
            cam_pose: [B, 12] camera/mic poses
            source_audio: [B, C, T] source waveform (near.wav; mono or stereo).
                          When None and use_static_source=True, the model will
                          use cached static_source_mag / static_phase_L/R
                          instead of recomputing STFT.
        Returns:
            binaural_output: [B, 2, T]
            If return_mag=True, also returns (left_mag, right_mag).
            If return_masks=True, also returns
              (mono_mask, diff_mask, source_magnitude, g_ft),
            where:
              - mono_mask/diff_mask: GS-only modulation fields on the STFT grid
                after distance attenuation (spec_mono_resized/spec_diff_resized).
              - source_magnitude: |STFT(source)| on the STFT grid.
              - g_ft: per-(F,T) distance attenuation on the STFT grid used to
                modulate both mono/diff fields.
        """
        device = cam_pose.device

        # STFT params (aligned with other models)
        n_fft = 512
        hop_length = 160
        window_length = 400
        torch_window = torch.hamming_window(window_length).to(device)

        if source_audio is None:
            if not self.use_static_source:
                raise RuntimeError(
                    "source_audio is None but use_static_source=False; "
                    "either pass a valid source_audio tensor or enable "
                    "static-source mode after initializing the model from a batch."
                )
            # Use cached static STFT, broadcast to batch size if needed.
            B = cam_pose.shape[0]
            source_magnitude = self.static_source_mag.to(device)
            phase_L = self.static_phase_L.to(device)
            phase_R = self.static_phase_R.to(device)
            if source_magnitude.shape[0] == 1 and B > 1:
                source_magnitude = source_magnitude.expand(B, -1, -1)
                phase_L = phase_L.expand(B, -1, -1)
                phase_R = phase_R.expand(B, -1, -1)
            source_phase_L = phase_L
            source_phase_R = phase_R
        else:
            stereo_input = source_audio.dim() > 2 and source_audio.shape[1] > 1

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

                source_magnitude = 0.5 * (mag_L + mag_R)  # content carrier
                source_magnitude = torch.nan_to_num(
                    source_magnitude, nan=0.0, posinf=0.0, neginf=0.0
                )
                source_phase_L = torch.nan_to_num(
                    phase_L, nan=0.0, posinf=0.0, neginf=0.0
                )
                source_phase_R = torch.nan_to_num(
                    phase_R, nan=0.0, posinf=0.0, neginf=0.0
                )
            else:
                # Mono case: STFT once and share phase to both ears
                source_mono = (
                    source_audio.squeeze() if source_audio.dim() > 1 else source_audio
                )
                source_spec = torch.stft(
                    source_mono,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=window_length,
                    window=torch_window,
                    return_complex=True,
                )
                source_magnitude = torch.abs(source_spec)
                source_phase = torch.angle(source_spec)
                source_magnitude = torch.nan_to_num(
                    source_magnitude, nan=0.0, posinf=0.0, neginf=0.0
                )
                source_phase = torch.nan_to_num(
                    source_phase, nan=0.0, posinf=0.0, neginf=0.0
                )
                source_phase_L = source_phase
                source_phase_R = source_phase

        # 1) Relative positions
        rel_world_norm, dir_pp, rel_cam_norm = self.compute_relative_positions(cam_pose)

        # 2) Evaluate mono & diff SH fields
        rel_for_sh = rel_cam_norm if self.use_cam_rotation else rel_world_norm
        mono_vals, diff_vals = self.eval_mono_diff_fields(rel_for_sh)  # [B, N], [B, N]

        # 3) Map SH fields to spectrogram grid and align with STFT grid
        B = source_magnitude.shape[0]
        spec_mono = mono_vals.reshape(B, self.freq_num, self.time_num)
        spec_diff = diff_vals.reshape(B, self.freq_num, self.time_num)

        # Resize SH fields to match STFT resolution (keep STFT grid as primary)
        if spec_mono.shape[-2:] != source_magnitude.shape[-2:]:
            spec_mono_resized = F.interpolate(
                spec_mono.unsqueeze(1),
                size=source_magnitude.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            spec_diff_resized = F.interpolate(
                spec_diff.unsqueeze(1),
                size=source_magnitude.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        else:
            spec_mono_resized = spec_mono
            spec_diff_resized = spec_diff

        # Optional: frequency-dependent distance attenuation g_f(r),
        # normalized by reference viewpoint so that g_ft_norm ≈ 1 at ref.
        B = source_magnitude.shape[0]
        eps = 1e-6
        if self.use_freq_atten:
            # rel_world_norm is normalized by max_norm; recover distances in world units
            distances = (rel_world_norm * float(self.max_norm)).norm(dim=-1)  # [B, N]
            distances_ft = distances.view(B, self.freq_num, self.time_num)    # [B, F_model, T_model]

            if self.use_hopkins_atten and (self.Q_param is not None) and (self.log_Rc_param is not None):
                # Hopkins–Stryker-style per-point attenuation:
                #   E(r) = Q/(4 pi r^2) + 4/R_c
                #   g(r) = sqrt(E(r) / E_ref(r))
                Q = torch.clamp(self.Q_param, min=0.1)
                Rc = torch.exp(self.log_Rc_param)

                r2 = distances_ft ** 2 + eps
                direct_curr = Q / (4.0 * math.pi * r2)
                rever_curr = 4.0 / (Rc + eps)
                E_curr = direct_curr + rever_curr  # [B, F, T]

                # ref_freq_g ≈ sqrt(E_ref) from initialize_from_batch
                g_ft_ref_model = self.ref_freq_g.view(1, self.freq_num, self.time_num).to(
                    E_curr.device
                )  # [1, F, T]
                E_ref = g_ft_ref_model ** 2
                g_ft_model = torch.sqrt(torch.clamp(E_curr / (E_ref + eps), min=0.0))
            else:
                # Simple inverse-distance attenuation as fallback
                if self.use_pointwise_alpha and (self._alpha_param is not None):
                    # Per-point exponent alpha_i mapped via tanh+1 (range 0~2).
                    alpha_ft = (torch.tanh(self._alpha_param) + 1.0).view(
                        1, self.freq_num, self.time_num
                    )
                    alpha_ft = torch.clamp(alpha_ft, min=0.0, max=2.0)
                    alpha_ft = alpha_ft.expand(B, -1, -1)  # [B, F, T]
                    g_ft_curr_model = (1.0 / (distances_ft + eps)) ** alpha_ft  # [B, F_model, T_model]
                else:
                    g_ft_curr_model = (1.0 / (distances_ft + eps)) ** float(self.freq_atten_alpha)  # [B, F_model, T_model]
                g_ft_ref_model = self.ref_freq_g.view(1, self.freq_num, self.time_num).to(
                    g_ft_curr_model.device
                )  # [1, F_model, T_model]
                g_ft_model = g_ft_curr_model / (g_ft_ref_model + eps)  # [B, F_model, T_model]
        else:
            g_ft_model = torch.ones(
                (B, self.freq_num, self.time_num),
                device=source_magnitude.device,
                dtype=source_magnitude.dtype,
            )

        # Resize per-point attenuation to STFT grid [B, F_spec, T_spec] if needed
        if g_ft_model.shape[-2:] != source_magnitude.shape[-2:]:
            g_ft = F.interpolate(
                g_ft_model.unsqueeze(1),
                size=source_magnitude.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        else:
            g_ft = g_ft_model

        # 4) GS-only modulation.
        #
        # Two modes:
        #   - Default (mask mode): SH fields produce masks that modulate
        #     |STFT(source)| (with optional distance attenuation g_f).
        #   - abs_mag_output=True: SH fields (after g_f) are interpreted as
        #     absolute mono/diff magnitudes on the STFT grid; no additional
        #     multiplication by |STFT(source)|.
        spec_mono_mod = spec_mono_resized * g_ft
        spec_diff_mod = spec_diff_resized * g_ft

        if getattr(self, "use_abs_mag_output", False):
            mono_mag = torch.nan_to_num(
                spec_mono_mod, nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            mono_mag = torch.nan_to_num(
                spec_mono_mod * source_magnitude, nan=0.0, posinf=0.0, neginf=0.0
            )

        diff_envelope = torch.nan_to_num(
            spec_diff_mod, nan=0.0, posinf=0.0, neginf=0.0
        )

        # 5) Cascaded mono→diff: diff envelope modulates mono energy (avoid shortcut learning).
        left_mag = torch.relu(mono_mag * (1.0 + diff_envelope))
        right_mag = torch.relu(mono_mag * (1.0 - diff_envelope))

        # 6) Geometry-guided explicit phase shift (optional).
        phase_L = source_phase_L
        phase_R = source_phase_R
        if self.use_geom_phase and self._geom_phase_initialized:
            try:
                # Signed azimuth angles in head frame for each audio point: [B, N]
                theta = self._compute_signed_azimuth(cam_pose.to(device))
                theta_ft = theta.view(B, self.freq_num, self.time_num)

                # Reference azimuth: prefer per-sample ref_cam_pose (input viewpoint),
                # otherwise fall back to cached ref_theta from initialize_from_batch().
                if self._pose_is_valid(ref_cam_pose):
                    theta_ref = self._compute_signed_azimuth(ref_cam_pose.to(device))
                    theta_ref_ft = theta_ref.view(B, self.freq_num, self.time_num)
                else:
                    theta_ref_ft = self.ref_theta.view(1, self.freq_num, self.time_num).to(device).expand(B, -1, -1)

                # Resize azimuth grids to match STFT grid (interpolate sin/cos to avoid wrap).
                target_size = source_phase_L.shape[-2:]  # (F_spec, T_spec)
                theta_ft = self._resize_angle(theta_ft, target_size)
                theta_ref_ft = self._resize_angle(theta_ref_ft, target_size)

                # Frequency bins in Hz for STFT.
                F_spec = int(target_size[0])
                freqs_hz = (
                    torch.arange(F_spec, device=device, dtype=source_phase_L.dtype)
                    * (float(getattr(self, "sr", 16000)) / float(n_fft))
                )  # [F_spec]
                omega = (2.0 * math.pi) * freqs_hz.view(1, -1, 1)  # [1, F_spec, 1]

                # ITD magnitude at current & reference viewpoints (seconds).
                dt = self._itd_delay(freqs_hz, theta_ft.abs())
                dt_ref = self._itd_delay(freqs_hz, theta_ref_ft.abs())

                # Optional: direction-conditioned multiplicative residual on dt/dt_ref.
                #
                #   dt_total     = dt     * factor(theta)
                #   dt_total_ref = dt_ref * factor(theta_ref)
                #   delta_dt     = dt_total - dt_total_ref
                if self.use_geom_phase_residual and isinstance(self._sh_phase, torch.Tensor):
                    try:
                        alpha = float(getattr(self, "geom_phase_residual_alpha", 1.0))
                        limit = float(getattr(self, "geom_phase_residual_scale_limit", 0.2))
                        if limit > 0.0:
                            T_spec = int(target_size[1])
                            # Build unit directions from signed azimuth in the x–z plane.
                            # Convention matches _compute_signed_azimuth:
                            #   theta = atan2(x, -z)  =>  d = [sin(theta), 0, -cos(theta)]
                            zero = torch.zeros_like(theta_ft)
                            dirs = torch.stack(
                                [torch.sin(theta_ft), zero, -torch.cos(theta_ft)], dim=-1
                            )
                            dirs_ref = torch.stack(
                                [torch.sin(theta_ref_ft), zero, -torch.cos(theta_ref_ft)], dim=-1
                            )
                            dirs = F.normalize(dirs, dim=-1, eps=1e-8)
                            dirs_ref = F.normalize(dirs_ref, dim=-1, eps=1e-8)

                            # SH coeffs per frequency (shared across time).
                            sh = torch.nan_to_num(
                                self._sh_phase, nan=0.0, posinf=0.0, neginf=0.0
                            ).to(device=device, dtype=source_phase_L.dtype)
                            # In case freq dims ever diverge, truncate to STFT bins.
                            sh = sh[:F_spec, :, : self.geom_phase_residual_sh_coeffs]
                            sh_bft = sh.view(1, F_spec, 1, 1, -1).expand(B, F_spec, T_spec, 1, -1)
                            sh_flat = sh_bft.reshape(-1, 1, sh.shape[-1])

                            s = eval_sh(
                                self.geom_phase_residual_sh_degree,
                                sh_flat,
                                dirs.reshape(-1, 3),
                            ).reshape(B, F_spec, T_spec, 1).squeeze(-1)
                            s_ref = eval_sh(
                                self.geom_phase_residual_sh_degree,
                                sh_flat,
                                dirs_ref.reshape(-1, 3),
                            ).reshape(B, F_spec, T_spec, 1).squeeze(-1)

                            s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
                            s_ref = torch.nan_to_num(s_ref, nan=0.0, posinf=0.0, neginf=0.0)

                            scale = torch.tanh(alpha * s) * limit
                            scale_ref = torch.tanh(alpha * s_ref) * limit
                            factor = torch.clamp(1.0 + scale, min=1e-3)
                            factor_ref = torch.clamp(1.0 + scale_ref, min=1e-3)
                            dt = dt * factor
                            dt_ref = dt_ref * factor_ref
                    except Exception:
                        pass

                delta_dt = dt - dt_ref  # [B, F_spec, T_spec]

                # Ear sign: + for ipsilateral, - for contralateral.
                # theta>0 => source on right (R ipsi, L contra); theta<0 => source on left.
                sign_L = torch.where(theta_ft < 0.0, 1.0, -1.0).to(source_phase_L.dtype)
                sign_R = -sign_L

                # Geometry-guided phase correction:
                #   Δφ_k = sign_k * (ω/2) * (ΔT - ΔT_ref)
                delta_phi_L = sign_L * 0.5 * omega * delta_dt
                delta_phi_R = sign_R * 0.5 * omega * delta_dt

                phase_L = source_phase_L + delta_phi_L
                phase_R = source_phase_R + delta_phi_R
                # Keep phases bounded for numerical stability in polar/istft.
                phase_L = torch.atan2(torch.sin(phase_L), torch.cos(phase_L))
                phase_R = torch.atan2(torch.sin(phase_R), torch.cos(phase_R))
            except Exception as _e:
                print(f"[Audio3DGSMonoDiffGSOnly] Geometry-guided phase failed: {_e}")
                phase_L = source_phase_L
                phase_R = source_phase_R

        # 7) Reconstruct complex spectra and iSTFT
        left_complex = torch.polar(left_mag, phase_L)
        right_complex = torch.polar(right_mag, phase_R)

        # When source_audio is None (static-source mode), let iSTFT infer length.
        sig_len = None
        if isinstance(source_audio, torch.Tensor):
            sig_len = source_audio.shape[-1]

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

        binaural = torch.stack([left_audio, right_audio], dim=1)
        binaural = torch.nan_to_num(
            binaural, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Optional debug outputs
        if return_mag and return_masks:
            return (
                binaural,
                left_mag,
                right_mag,
                spec_mono_resized,
                spec_diff_resized,
                source_magnitude,
                g_ft,
            )
        if return_mag:
            return binaural, left_mag, right_mag
        if return_masks:
            return binaural, spec_mono_resized, spec_diff_resized, source_magnitude, g_ft
        return binaural


def build_model(cfg, gaussian_model=None, scene=None):
    """
    Build Audio3DGSMonoDiffGSOnly model.
    Uses the same freq/time dims as Audio3DGSMonoDiff.
    """
    # We can reuse dataset-driven H/W from cfg.dataset (like other builders)
    freq_num = cfg.dataset.get("H", 257)
    time_num = cfg.dataset.get("W", 348)
    model = Audio3DGSMonoDiffGSOnly(cfg, freq_num=freq_num, time_num=time_num)
    return model
