#!/usr/bin/env python3
"""
Visualize Audio 3DGS point cloud as a 3D scatter plot.

Example:
  CUDA_VISIBLE_DEVICES=0 \\
  python tools/visualize_audio_3dgs_cloud.py \\
    --cfg configs/audio_3dgs_replaynvas_viewpoint.yaml \\
    --checkpoint 3dgs_result/replayNVAS/SC-1044/viewpoint_8/frame_123/best_model.pth \\
    --output picture/audio3dgs_SC-1044_frame123.png
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Add project root & tools for imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "tools"))

import _init_paths  # noqa: F401,E402
from configs import cfg, update_config  # noqa: E402
from importlib import import_module as impm  # noqa: E402

plt.rcParams['font.family'] = 'serif'
# 尝试按顺序寻找字体：先找 Times New Roman，找不到就找 DejaVu Serif (Linux标配)，再找不到就用通用 serif
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'serif']
plt.rcParams['axes.unicode_minus'] = False

def load_model(cfg_path: str, checkpoint_path: str, keep_static_source: bool = False):
    class Args:
        def __init__(self, yaml_file):
            self.yaml_file = yaml_file
            self.opts = []
            self.distributed = False
            self.local_rank = 0
            self.resume = None

    config_args = Args(cfg_path)
    update_config(cfg, config_args)

    # Try to infer scene id (e.g., "SC-1044") from checkpoint path so that
    # xyz_anchor_to_scene (if enabled) can use a meaningful scene_scope.
    try:
        parts = os.path.normpath(checkpoint_path).split(os.sep)
        scene_id = next((p for p in parts if p.startswith("SC-")), None)
        if scene_id:
            cfg.defrost()
            cfg.dataset.scene_scope = scene_id
            cfg.freeze()
    except Exception:
        # Best-effort only; fall back silently if anything goes wrong.
        pass

    model_module = impm(f"libs.models.{cfg.model.file}")
    model = model_module.build_model(cfg)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    static_source_mag = None
    # Optionally keep a copy of static-source STFT magnitude for visualization.
    # We still drop it from the state dict before loading to avoid size
    # mismatch warnings/errors with newer model definitions.
    if isinstance(state, dict):
        if keep_static_source and "static_source_mag" in state:
            try:
                static_source_mag = state["static_source_mag"].detach().cpu()
            except Exception:
                static_source_mag = torch.as_tensor(state["static_source_mag"]).cpu()
        # Drop static-source STFT buffers before loading.
        for k in ["static_source_mag", "static_phase_L", "static_phase_R"]:
            if k in state:
                state.pop(k, None)
        # Handle potential shape mismatches for newly introduced Hopkins parameters
        # so that older checkpoints remain loadable.
        try:
            import torch as _torch
            for hk in ["Q_param", "log_Rc_param"]:
                if hk in state and hasattr(model, hk):
                    ckpt_tensor = state[hk]
                    model_tensor = getattr(model, hk)
                    if isinstance(model_tensor, _torch.Tensor):
                        if ckpt_tensor.shape != model_tensor.shape:
                            if ckpt_tensor.numel() == model_tensor.numel():
                                state[hk] = ckpt_tensor.view_as(model_tensor)
                                print(
                                    f"[visualize_audio_3dgs_cloud] Reshaped checkpoint {hk} "
                                    f"from {tuple(ckpt_tensor.shape)} to {tuple(model_tensor.shape)}"
                                )
                            else:
                                state.pop(hk, None)
                                print(
                                    f"[visualize_audio_3dgs_cloud] Dropped incompatible "
                                    f"checkpoint key {hk} with shape {tuple(ckpt_tensor.shape)}"
                                )
        except Exception as _e:
            print(
                f"[visualize_audio_3dgs_cloud] Warning: could not sanitize Hopkins params "
                f"in state_dict: {_e}"
            )
    model.load_state_dict(state, strict=False)
    return model, static_source_mag


def plot_xyz(
    xyz: np.ndarray,
    out_path: str,
    title: str = "",
    colors=None,
    cmap: str = "viridis",
    add_colorbar: bool = False,
    colorbar_label: str = "",
    axis_limits=None,
    font_size=None,
    title_font_size=None,
    colorbar_pad=None,
    z_labelpad=None,
):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    xs = xyz[:, 0]
    ys = xyz[:, 1]
    zs = xyz[:, 2]

    # If colors is None, matplotlib will use a default single color.
    if colors is None:
        sc = ax.scatter(xs, ys, zs, s=1, alpha=0.4)
    else:
        colors_arr = np.asarray(colors)
        if np.issubdtype(colors_arr.dtype, np.number):
            sc = ax.scatter(xs, ys, zs, s=1, alpha=0.4, c=colors, cmap=cmap)
        else:
            # String or RGBA colors; do not pass a colormap.
            sc = ax.scatter(xs, ys, zs, s=1, alpha=0.4, c=colors)
    fs = None
    if font_size is not None:
        try:
            fs = float(font_size)
            if not np.isfinite(fs) or fs <= 0:
                fs = None
        except Exception:
            fs = None

    title_fs = fs
    if title_font_size is not None:
        try:
            title_fs = float(title_font_size)
            if not np.isfinite(title_fs) or title_fs <= 0:
                title_fs = fs
        except Exception:
            title_fs = fs

    label_kwargs = {"fontsize": fs} if fs is not None else {}
    title_kwargs = {"fontsize": title_fs} if title_fs is not None else {}

    ax.set_xlabel("X", **label_kwargs)
    ax.set_ylabel("Y", **label_kwargs)
    z_label_kwargs = dict(label_kwargs)
    if z_labelpad is not None:
        try:
            zlp = float(z_labelpad)
            if np.isfinite(zlp):
                z_label_kwargs["labelpad"] = zlp
        except Exception:
            pass
    ax.set_zlabel("Z", **z_label_kwargs)
    if title:
        ax.set_title(title, **title_kwargs)
    if fs is not None:
        ax.xaxis.set_tick_params(labelsize=fs)
        ax.yaxis.set_tick_params(labelsize=fs)
        ax.zaxis.set_tick_params(labelsize=fs)

    # Equal aspect ratio. If axis_limits is provided, keep a fixed coordinate scale.
    if axis_limits is not None:
        (x0, x1), (y0, y1), (z0, z1) = axis_limits
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_zlim(z0, z1)
    else:
        max_range = np.array(
            [xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]
        ).max()
        mid_x = (xs.max() + xs.min()) * 0.5
        mid_y = (ys.max() + ys.min()) * 0.5
        mid_z = (zs.max() + zs.min()) * 0.5
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if add_colorbar and colors is not None:
        # Only add colorbar when colors is numeric; string colors (e.g., "red"/"blue")
        # do not benefit from a colorbar.
        colors_arr = np.asarray(colors)
        if np.issubdtype(colors_arr.dtype, np.number):
            pad = None
            if colorbar_pad is not None:
                try:
                    pad = float(colorbar_pad)
                    if not np.isfinite(pad) or pad < 0.0:
                        pad = None
                except Exception:
                    pad = None
            if pad is None:
                pad = 0.05
            cbar = fig.colorbar(sc, ax=ax, pad=pad)
            if fs is not None:
                cbar.set_label(colorbar_label or "Color value", fontsize=fs)
                cbar.ax.tick_params(labelsize=fs)
            else:
                cbar.set_label(colorbar_label or "Color value")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def compute_equal_aspect_limits(xyz: np.ndarray):
    """Compute shared axis limits (xlim, ylim, zlim) with equal aspect ratio."""
    if xyz is None:
        return ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.size == 0:
        return ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    finite_mask = np.isfinite(xyz).all(axis=1)
    xyz_finite = xyz[finite_mask]
    if xyz_finite.size == 0:
        return ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    mins = xyz_finite.min(axis=0)
    maxs = xyz_finite.max(axis=0)
    spans = maxs - mins
    max_range = float(np.max(spans))
    if not np.isfinite(max_range) or max_range <= 0.0:
        max_range = 1.0
    mids = (maxs + mins) * 0.5
    half = max_range * 0.5
    return (
        (float(mids[0] - half), float(mids[0] + half)),
        (float(mids[1] - half), float(mids[1] + half)),
        (float(mids[2] - half), float(mids[2] + half)),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Audio 3DGS (mono/diff) point cloud as 3D scatter."
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Config file (e.g., configs/audio_3dgs_replaynvas_viewpoint.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (best_model.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PNG path for scatter plot",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help=(
            "Custom base title for the plots. "
            "If set to an empty string, no base title is shown."
        ),
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=None,
        help=(
            "Font size (points) for title/axis labels/ticks/colorbar. "
            "If omitted, uses matplotlib defaults."
        ),
    )
    parser.add_argument(
        "--title-font-size",
        type=float,
        default=None,
        help=(
            "Override font size (points) for the plot title only. "
            "If omitted, falls back to --font-size (or matplotlib defaults)."
        ),
    )
    parser.add_argument(
        "--colorbar-pad",
        type=float,
        default=None,
        help=(
            "Padding between plot axes and the colorbar (matplotlib `pad`, "
            "in fraction of the parent axes). Larger values move the colorbar "
            "further right to avoid covering the Z label."
        ),
    )
    parser.add_argument(
        "--z-labelpad",
        type=float,
        default=None,
        help=(
            "Padding for the Z-axis label (matplotlib `labelpad`). "
            "Smaller (even negative) values move the label inward."
        ),
    )
    parser.add_argument(
        "--color-by-amplitude",
        action="store_true",
        help="Color points by per-point mono SH amplitude (DC term).",
    )
    parser.add_argument(
        "--color-by-source-mag",
        action="store_true",
        help=(
            "Color points by static source STFT magnitude (L/R-averaged), "
            "resampled to the Audio3DGS point grid if needed. Requires "
            "a checkpoint that stores 'static_source_mag'."
        ),
    )
    parser.add_argument(
        "--color-by-frequency",
        action="store_true",
        help="Color points by frequency index (low→high).",
    )
    parser.add_argument(
        "--high-freq-top",
        type=float,
        default=0.0,
        help=(
            "When used with --color-by-frequency, also save a separate figure "
            "containing only the top X fraction of highest-frequency points "
            "(e.g., 0.3 for top 30%%)."
        ),
    )
    parser.add_argument(
        "--binary-color",
        action="store_true",
        help=(
            "If set with --color-by-amplitude/--color-by-source-mag, use two "
            "colors (high/low amplitude) instead of a continuous colormap."
        ),
    )
    parser.add_argument(
        "--amp-quantile",
        type=float,
        default=0.8,
        help=(
            "Quantile used as threshold between high/low amplitude when "
            "using --binary-color (default: 0.8, i.e., top 20%% points are high)."
        ),
    )
    parser.add_argument(
        "--amp-split-plots",
        action="store_true",
        help=(
            "When used with --color-by-amplitude/--color-by-source-mag, also save "
            "two extra figures: one containing only the top-X%% amplitude points, "
            "and one containing only the bottom (1-X)%% points. "
            "See --amp-split-frac for X."
        ),
    )
    parser.add_argument(
        "--share-axis-limits",
        action="store_true",
        help=(
            "Use axis limits computed from the full point cloud for subset plots "
            "(e.g., amp split / high-freq subset) so coordinate scales align."
        ),
    )
    parser.add_argument(
        "--amp-split-frac",
        type=float,
        default=0.5,
        help=(
            "When using --amp-split-plots, fraction of points to include in the "
            "high-amplitude subset (0–1, default: 0.5 for top 50%% / bottom 50%%)."
        ),
    )
    parser.add_argument(
        "--amp-log-scale",
        action="store_true",
        help=(
            "Apply log scaling (log(x + eps)) to amplitude values before "
            "normalizing for visualization. Useful when most amplitudes are small."
        ),
    )
    parser.add_argument(
        "--amp-log-eps",
        type=float,
        default=1e-6,
        help="Epsilon added before log when using --amp-log-scale (default: 1e-6).",
    )
    args = parser.parse_args()

    if args.color_by_amplitude and args.color_by_source_mag:
        parser.error(
            "Cannot use --color-by-amplitude and --color-by-source-mag together; "
            "please choose one."
        )

    def normalize_amplitude(arr_np: np.ndarray) -> np.ndarray:
        """Optionally log-scale, then normalize amplitude array to [0, 1]."""
        arr_np = np.nan_to_num(arr_np, nan=0.0, posinf=0.0, neginf=0.0)
        if args.amp_log_scale:
            eps = float(getattr(args, "amp_log_eps", 1e-6))
            if not np.isfinite(eps) or eps < 0.0:
                eps = 1e-6
            arr_np = np.log(arr_np + eps)
        a_min, a_max = float(arr_np.min()), float(arr_np.max())
        if a_max > a_min:
            arr_norm = (arr_np - a_min) / (a_max - a_min)
        else:
            arr_norm = np.zeros_like(arr_np)
        return arr_norm

    need_static_source = bool(getattr(args, "color_by_source_mag", False))
    model, static_source_mag = load_model(
        args.cfg, args.checkpoint, keep_static_source=need_static_source
    )
    if not hasattr(model, "_xyz"):
        raise RuntimeError("Loaded model does not have _xyz attribute.")

    with torch.no_grad():
        xyz = model._xyz.detach().cpu().numpy()
        shared_axis_limits = (
            compute_equal_aspect_limits(xyz) if args.share_axis_limits else None
        )

        amp = None
        amp_label = ""
        freq_colors = None
        freq_idx_arr = None

        # Option 1: color by static source STFT magnitude (L/R-averaged).
        if args.color_by_source_mag:
            if static_source_mag is None:
                print(
                    "[visualize_audio_3dgs_cloud] Warning: checkpoint has no "
                    "'static_source_mag'; cannot color by source magnitude."
                )
            else:
                src_mag = static_source_mag
                if not isinstance(src_mag, torch.Tensor):
                    src_mag = torch.as_tensor(src_mag)
                # Expect shape [B, F_src, T_src] or [F_src, T_src]
                if src_mag.dim() == 3:
                    src_mag = src_mag[0:1]  # [1, F_src, T_src]
                elif src_mag.dim() == 2:
                    src_mag = src_mag.unsqueeze(0)  # [1, F_src, T_src]
                else:
                    print(
                        "[visualize_audio_3dgs_cloud] Warning: static_source_mag "
                        f"has unexpected shape {tuple(src_mag.shape)}; "
                        "cannot color by source magnitude."
                    )
                    src_mag = None

                if src_mag is not None:
                    # Resize static STFT to model's (freq_num, time_num) grid if needed.
                    freq_num = int(getattr(model, "freq_num", src_mag.shape[-2]))
                    time_num = int(getattr(model, "time_num", src_mag.shape[-1]))
                    n_points = xyz.shape[0]
                    if freq_num * time_num != n_points:
                        print(
                            "[visualize_audio_3dgs_cloud] Warning: freq_num*time_num "
                            f"({freq_num}*{time_num}) != number of points ({n_points}); "
                            "cannot reliably map source magnitude to points."
                        )
                    else:
                        F_src, T_src = src_mag.shape[-2:]
                        if (F_src != freq_num) or (T_src != time_num):
                            src_mag_resized = F.interpolate(
                                src_mag.unsqueeze(1),
                                size=(freq_num, time_num),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(1)
                        else:
                            src_mag_resized = src_mag

                        amp_t = src_mag_resized.reshape(-1)
                        amp_t = torch.nan_to_num(
                            amp_t, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        amp = normalize_amplitude(amp_t.cpu().numpy())
                        amp_label = "Normalized STFT magnitude"

        # Option 2: color by mono SH amplitude (DC term).
        if (amp is None) and args.color_by_amplitude and hasattr(model, "_sh_mono"):
            sh_mono = model._sh_mono.detach().cpu().numpy()  # [N, 1, C]
            amp_raw = np.abs(sh_mono[:, 0, 0])
            amp = normalize_amplitude(amp_raw)
            amp_label = "Normalized mono SH amplitude (DC)"

        # Option 3: color by frequency index (low → high).
        if args.color_by_frequency:
            if hasattr(model, "freq_num") and hasattr(model, "time_num"):
                freq_num = int(getattr(model, "freq_num"))
                time_num = int(getattr(model, "time_num"))
                n_points = xyz.shape[0]
                if n_points == freq_num * time_num and freq_num > 0 and time_num > 0:
                    # Points are ordered as [f, t] flattened with freq-major order.
                    freq_idx_arr = np.repeat(np.arange(freq_num), time_num)
                    if freq_num > 1:
                        freq_colors = freq_idx_arr.astype(np.float32) / float(freq_num - 1)
                    else:
                        freq_colors = np.zeros_like(freq_idx_arr, dtype=np.float32)
                else:
                    print(
                        "[visualize_audio_3dgs_cloud] Warning: "
                        "freq_num * time_num != number of points; cannot color by frequency."
                    )
            else:
                print(
                    "[visualize_audio_3dgs_cloud] Warning: model has no freq_num/time_num; "
                    "cannot color by frequency."
                )

    # Base title: by default, use parent directory name of checkpoint
    # (e.g., "frame_7"). Can be overridden via --title; if --title is set
    # to an empty string, no base title is used.
    if args.title is not None:
        title = args.title
    else:
        title = os.path.basename(os.path.dirname(args.checkpoint))

    # Priority: amplitude coloring (if requested and available) > frequency coloring.
    if (args.color_by_amplitude or args.color_by_source_mag) and amp is not None:
        if args.amp_log_scale and amp_label:
            amp_label = f"{amp_label} (log-scaled)"
        if args.binary_color:
            # High / low amplitude as two colors.
            q = np.clip(args.amp_quantile, 0.0, 1.0)
            thresh = np.quantile(amp, q)
            high_mask = amp >= thresh
            colors = np.where(high_mask, "red", "blue")
            plot_xyz(
                xyz,
                args.output,
                title=title,
                colors=colors,
                add_colorbar=False,
                axis_limits=shared_axis_limits,
                font_size=args.font_size,
                title_font_size=args.title_font_size,
                colorbar_pad=args.colorbar_pad,
                z_labelpad=args.z_labelpad,
            )
        else:
            # Continuous amplitude colormap.
            plot_xyz(
                xyz,
                args.output,
                title=title,
                colors=amp,
                cmap="viridis",
                add_colorbar=True,
                colorbar_label=amp_label or "Normalized magnitude",
                axis_limits=shared_axis_limits,
                font_size=args.font_size,
                title_font_size=args.title_font_size,
                colorbar_pad=args.colorbar_pad,
                z_labelpad=args.z_labelpad,
            )
            # Optionally, also save separate figures for top-X% and bottom-(1-X)% amplitude subsets.
            if args.amp_split_plots:
                try:
                    frac = float(getattr(args, "amp_split_frac", 0.5))
                    # Clamp to a reasonable open interval (0, 1)
                    if not np.isfinite(frac):
                        frac = 0.5
                    frac = max(1e-4, min(1.0 - 1e-4, frac))
                    # Threshold so that approximately top-`frac` fraction are high.
                    thresh = float(np.quantile(amp, 1.0 - frac))
                    high_mask = amp >= thresh
                    low_mask = amp < thresh

                    base, ext = os.path.splitext(args.output)
                    top_pct = int(frac * 100 + 0.5)
                    bottom_pct = int((1.0 - frac) * 100 + 0.5)

                    if high_mask.any():
                        xyz_high = xyz[high_mask]
                        amp_high = amp[high_mask]
                        out_high = base + f"_top{top_pct}pct" + ext
                        if title:
                            high_title = f"{title} (top {top_pct}% magnitude)"
                        else:
                            high_title = f"top {top_pct}% magnitude"
                        plot_xyz(
                            xyz_high,
                            out_high,
                            title=high_title,
                            colors=amp_high,
                            cmap="viridis",
                            add_colorbar=True,
                            colorbar_label=amp_label or "Normalized magnitude",
                            axis_limits=shared_axis_limits,
                            font_size=args.font_size,
                            title_font_size=args.title_font_size,
                            colorbar_pad=args.colorbar_pad,
                            z_labelpad=args.z_labelpad,
                        )
                    else:
                        print(
                            "[visualize_audio_3dgs_cloud] Warning: top magnitude mask is empty; "
                            "skipping top-subset plot."
                        )

                    if low_mask.any():
                        xyz_low = xyz[low_mask]
                        amp_low = amp[low_mask]
                        out_low = base + f"_bottom{bottom_pct}pct" + ext
                        if title:
                            low_title = f"{title} (bottom {bottom_pct}% magnitude)"
                        else:
                            low_title = f"bottom {bottom_pct}% magnitude"
                        plot_xyz(
                            xyz_low,
                            out_low,
                            title=low_title,
                            colors=amp_low,
                            cmap="viridis",
                            add_colorbar=True,
                            colorbar_label=amp_label or "Normalized magnitude",
                            axis_limits=shared_axis_limits,
                            font_size=args.font_size,
                            title_font_size=args.title_font_size,
                            colorbar_pad=args.colorbar_pad,
                            z_labelpad=args.z_labelpad,
                        )
                    else:
                        print(
                            "[visualize_audio_3dgs_cloud] Warning: bottom magnitude mask is empty; "
                            "skipping bottom-subset plot."
                        )
                except Exception as _e:
                    print(
                        f"[visualize_audio_3dgs_cloud] Warning: failed to generate amp-split plots: {_e}"
                    )
    elif args.color_by_frequency and freq_colors is not None:
        # Continuous frequency colormap: low → high frequency (all points).
        plot_xyz(
            xyz,
            args.output,
            title=title,
            colors=freq_colors,
            # Use reversed plasma so that low frequency is bright yellow
            # and high frequency is dark purple.
            cmap="plasma_r",
            add_colorbar=True,
            colorbar_label="Normalized frequency",
            axis_limits=shared_axis_limits,
            font_size=args.font_size,
            title_font_size=args.title_font_size,
            colorbar_pad=args.colorbar_pad,
            z_labelpad=args.z_labelpad,
        )

        # Optionally, also save a separate figure for top-X% high-frequency points.
        if freq_idx_arr is not None and args.high_freq_top > 0.0:
            frac = float(args.high_freq_top)
            # Clamp to [0,1]
            frac = max(0.0, min(1.0, frac))
            if frac > 0.0:
                try:
                    freq_num = int(freq_idx_arr.max() + 1)
                    # Threshold frequency band index: keep top frac fraction.
                    band_thresh = int(np.floor(freq_num * (1.0 - frac)))
                    mask = freq_idx_arr >= band_thresh
                    if mask.any():
                        xyz_high = xyz[mask]
                        freq_high = freq_colors[mask]
                        base, ext = os.path.splitext(args.output)
                        suffix = f"_top{int(frac * 100 + 0.5)}pct"
                        out_high = base + suffix + ext
                        pct = int(frac * 100 + 0.5)
                        if title:
                            high_freq_title = f"{title} (top {pct}% freq)"
                        else:
                            high_freq_title = f"top {pct}% freq"
                        plot_xyz(
                            xyz_high,
                            out_high,
                            title=high_freq_title,
                            colors=freq_high,
                            cmap="plasma_r",
                            add_colorbar=True,
                            colorbar_label="Normalized frequency",
                            axis_limits=shared_axis_limits,
                            font_size=args.font_size,
                            title_font_size=args.title_font_size,
                            colorbar_pad=args.colorbar_pad,
                            z_labelpad=args.z_labelpad,
                        )
                    else:
                        print(
                            "[visualize_audio_3dgs_cloud] Warning: high-freq mask is empty; "
                            "skipping high-frequency-only plot."
                        )
                except Exception as _e:
                    print(
                        f"[visualize_audio_3dgs_cloud] Warning: failed to generate high-frequency-only plot: {_e}"
                    )
    else:
        # Fallback: XYZ-only scatter (single color)
        plot_xyz(
            xyz,
            args.output,
            title=title,
            axis_limits=shared_axis_limits,
            font_size=args.font_size,
            title_font_size=args.title_font_size,
            colorbar_pad=args.colorbar_pad,
            z_labelpad=args.z_labelpad,
        )


if __name__ == "__main__":
    main()
