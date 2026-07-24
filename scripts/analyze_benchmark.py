#!/usr/bin/env python3
"""
UI-LIC Metric Analysis, BD-Rate, and Plotting Suite
Aggregates completed evaluation JSON metrics (*_metrics.json), calculates BD-Rate & BD-PSNR,
analyzes per-image statistical variance across dataset images (e.g. Kodak 24),
renders 300 DPI publication-quality figures, and exports paper LaTeX & README markdown.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import BD-Rate module
sys.path.append(os.path.dirname(__file__))
from bd_rate import bd_rate, bd_metric

# Codec type classification.
# - "predictive": signal-reconstruction codecs — BD-rate comparisons are valid.
# - "generative": synthesis/hallucination codecs (e.g. diffusion-based) —
#   pixel-fidelity metrics (PSNR/SSIM) are meaningless as a distortion axis;
#   BD-rate against predictive anchors is conceptually invalid.
CODEC_TYPE = {
    "DCVC-RT":          "predictive",
    "ELIC":             "predictive",
    "HPCM":             "predictive",
    "HPCM_Base":        "predictive",
    "HPCM_Base_SSIM":   "predictive",
    "HPCM_Large":       "predictive",
    "HPCM_Large_SSIM":  "predictive",
    "RwkvCompress":     "predictive",
    "LIC-TCM":          "predictive",
    "StableCodec":      "generative",   # Stable Diffusion-based; ultra-low bpp regime
    "AV1":              "predictive",
    "HEVC":             "predictive",
    "AVC":              "predictive",
}

COLOR_MAP = {
    "DCVC-RT": "#1f77b4",       # Blue
    "ELIC": "#ff7f0e",          # Orange
    "HPCM": "#2ca02c",          # Green
    "HPCM_Base": "#2ca02c",
    "HPCM_Large": "#006400",
    "RwkvCompress": "#d62728",  # Red
    "LIC-TCM": "#9467bd",       # Purple
    "StableCodec": "#8c564b",   # Brown
    "AV1": "#e377c2",           # Pink
    "HEVC": "#7f7f7f",          # Gray
    "AVC": "#bcbd22",           # Olive
}

MARKER_MAP = {
    "DCVC-RT": "o",
    "ELIC": "s",
    "HPCM": "^",
    "HPCM_Base": "^",
    "HPCM_Base_SSIM": "^",
    "HPCM_Large": "v",
    "HPCM_Large_SSIM": "v",
    "RwkvCompress": "D",
    "LIC-TCM": "P",
    "StableCodec": "*",
    "AV1": "X",
    "HEVC": "h",
    "AVC": "p"
}

# Within model families that have multiple capacity tiers, the lower-capacity
# variant is superseded by the higher-capacity one for "best-variant" plots.
# Key = model to suppress; Value = model that replaces it.
FAMILY_BEST = {
    "HPCM_Base":      "HPCM_Large",       # Large supersedes Base (MSE-optimised)
    "HPCM_Base_SSIM": "HPCM_Large_SSIM",  # Large supersedes Base (SSIM-optimised)
}

def parse_model_name(task_name):
    # SSIM variants must appear before their MSE counterparts so they match first.
    known_models = [
        "DCVC-RT", "ELIC",
        "HPCM_Base_SSIM", "HPCM_Base",
        "HPCM_Large_SSIM", "HPCM_Large",
        "HPCM",
        "RwkvCompress", "LALIC", "LIC-TCM", "TCM",
        "StableCodec", "AV1", "HEVC", "AVC",
    ]
    for m in known_models:
        if task_name.startswith(m):
            qual = task_name[len(m):].lstrip("_- ")
            return m, qual if qual else "default"
    parts = task_name.split("_")
    return parts[0], "_".join(parts[1:]) if len(parts) > 1 else "default"

def load_all_metrics(results_dir):
    results_dir = os.path.abspath(os.path.expanduser(results_dir))
    model_data = {}

    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith("_metrics.json"):
                fpath = os.path.join(root, f)
                try:
                    with open(fpath, "r") as json_file:
                        data = json.load(json_file)
                        
                    task_name = data.get("task_name", f.replace("_metrics.json", ""))
                    model, qual = parse_model_name(task_name)
                    
                    averages = data.get("averages", {})
                    per_img = data.get("per_image_metrics", [])
                    
                    if model not in model_data:
                        model_data[model] = []
                        
                    model_data[model].append({
                        "task_name": task_name,
                        "quality": qual,
                        "bpp": averages.get("bpp", 0.0),
                        "psnr": averages.get("psnr", 0.0),
                        "psnr_y": averages.get("psnr_y", 0.0),
                        "ssim": averages.get("ssim", 0.0),
                        "lpips": averages.get("lpips", 0.0),
                        "vmaf": averages.get("vmaf", 0.0),
                        "per_image": per_img
                    })
                except Exception as e:
                    print(f"Warning: Failed to parse metric file {fpath}: {e}")

    for model in model_data:
        model_data[model] = sorted(model_data[model], key=lambda x: x["bpp"])

    return model_data

def compute_per_image_variance(model_data):
    variance_stats = {}

    for model, q_list in model_data.items():
        variance_stats[model] = []
        for point in q_list:
            per_img = point["per_image"]
            if not per_img: continue
                
            bpps = [p.get("bpp", 0.0) for p in per_img]
            psnrs = [p.get("psnr", 0.0) for p in per_img]
            ssims = [p.get("ssim", 0.0) for p in per_img]
            lpipss = [p.get("lpips", 0.0) for p in per_img]

            variance_stats[model].append({
                "task_name": point["task_name"],
                "quality": point["quality"],
                "mean_bpp": float(np.mean(bpps)),
                "std_bpp": float(np.std(bpps)),
                "var_bpp": float(np.var(bpps)),
                "min_bpp": float(np.min(bpps)),
                "max_bpp": float(np.max(bpps)),
                "mean_psnr": float(np.mean(psnrs)),
                "std_psnr": float(np.std(psnrs)),
                "var_psnr": float(np.var(psnrs)),
                "min_psnr": float(np.min(psnrs)),
                "max_psnr": float(np.max(psnrs)),
                "mean_ssim": float(np.mean(ssims)),
                "std_ssim": float(np.std(ssims)),
                "mean_lpips": float(np.mean(lpipss)),
                "std_lpips": float(np.std(lpipss))
            })

    return variance_stats

def is_generative(model_name):
    """Returns True if the model is a generative codec (e.g. diffusion-based).
    Unknown models default to predictive so they are always included in BD."""
    return CODEC_TYPE.get(model_name, "predictive") == "generative"


def compute_all_bd_rates(model_data, anchors=["AV1", "HEVC", "ELIC"]):
    bd_results = {}

    for anchor in anchors:
        if anchor not in model_data or len(model_data[anchor]) < 2:
            continue

        # Generative codecs should never serve as a BD anchor against predictive
        # codecs — their quality axis has a different meaning.
        if is_generative(anchor):
            print(f"INFO: Skipping BD anchor '{anchor}' — classified as generative.")
            continue

        anc_pts = model_data[anchor]
        anc_bpp = [p["bpp"] for p in anc_pts]
        anc_psnr = [p["psnr"] for p in anc_pts]
        anc_ssim = [p["ssim"] for p in anc_pts]
        anc_lpips = [p["lpips"] for p in anc_pts]

        bd_results[anchor] = {}

        for model, tgt_pts in model_data.items():
            if model == anchor or len(tgt_pts) < 2:
                continue

            # Skip BD computation when target is a generative codec.
            # BD-rate is conceptually invalid across codec paradigms: generative
            # codecs synthesise plausible images rather than reconstructing pixels,
            # so PSNR/SSIM do not measure the same thing on both curves.
            if is_generative(model):
                bd_results[anchor][model] = {
                    "bd_rate_psnr": None,
                    "bd_rate_ssim": None,
                    "bd_rate_lpips": None,
                    "bd_psnr_db": None,
                    "skipped_reason": "generative_codec"
                }
                continue

            tgt_bpp = [p["bpp"] for p in tgt_pts]
            tgt_psnr = [p["psnr"] for p in tgt_pts]
            tgt_ssim = [p["ssim"] for p in tgt_pts]
            tgt_lpips = [p["lpips"] for p in tgt_pts]

            # NOTE: LPIPS is negated so that "higher = better" holds for the
            # BD integration, keeping the sign convention consistent with PSNR/SSIM.
            # A negative BD-rate LPIPS means the target is perceptually better.
            r_psnr = bd_rate(anc_bpp, anc_psnr, tgt_bpp, tgt_psnr)
            r_ssim = bd_rate(anc_bpp, anc_ssim, tgt_bpp, tgt_ssim)
            r_lpips = bd_rate(anc_bpp, [-l for l in anc_lpips], tgt_bpp, [-l for l in tgt_lpips])
            bd_psnr_val = bd_metric(anc_bpp, anc_psnr, tgt_bpp, tgt_psnr)

            bd_results[anchor][model] = {
                "bd_rate_psnr": r_psnr,
                "bd_rate_ssim": r_ssim,
                "bd_rate_lpips": r_lpips,
                "bd_psnr_db": bd_psnr_val,
                "skipped_reason": None
            }

    return bd_results

def get_best_variant_models(model_data):
    """
    Returns model_data filtered to remove lower-capacity variants when a
    higher-capacity sibling exists (as defined in FAMILY_BEST).
    E.g. HPCM_Base is excluded when HPCM_Large is present.
    """
    suppressed = {
        base for base, best in FAMILY_BEST.items()
        if best in model_data
    }
    return {m: pts for m, pts in model_data.items() if m not in suppressed}


def get_top_performing_models(model_data, bd_results, n=6):
    """
    Selects the top-n predictive models by average BD-rate PSNR across all
    anchors (lower/more-negative = better). Anchor models are always included
    as reference lines. Generative codecs are excluded.
    Returns (filtered_model_data, scores_dict).
    """
    anchors = set(bd_results.keys())
    scores = {}
    for model in model_data:
        if is_generative(model) or model in anchors:
            continue
        vals = [
            res["bd_rate_psnr"]
            for res in (bd_results[anchor].get(model, {}) for anchor in anchors)
            if res and res.get("bd_rate_psnr") is not None and not np.isnan(res["bd_rate_psnr"])
        ]
        if vals:
            scores[model] = float(np.mean(vals))

    top_models = sorted(scores, key=lambda m: scores[m])[:n]
    selected = set(top_models) | anchors
    filtered = {m: pts for m, pts in model_data.items() if m in selected}
    return filtered, scores


def generate_all_plots(output_dir, model_data, variance_stats, bd_results):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # ------------------------------------------------------------------
    # Shared helper: produce one BPP-vs-metric RD curve plot
    # ------------------------------------------------------------------
    def _plot_rd_metric(subset, metric_key, ylabel, title, filename,
                        legend_loc="lower right", xlim=None):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        plotted_any = False
        all_bpps = []
        for model, pts in subset.items():
            if not pts:
                continue
            bpps = [p["bpp"] for p in pts]
            vals = [p[metric_key] for p in pts]
            # Skip models where the metric was never computed (all zeros).
            if all(v == 0.0 for v in vals):
                continue
            all_bpps.extend(bpps)
            c = COLOR_MAP.get(model, "#333333")
            mk = MARKER_MAP.get(model, "o")
            ax.plot(bpps, vals, label=model, color=c, marker=mk,
                    linewidth=2.0, markersize=7)
            
            # Annotate checkpoint quality labels when plotting a single model family (e.g. StableCodec)
            if len(subset) == 1:
                for p in pts:
                    q_tag = p.get("quality", "")
                    if q_tag and q_tag != "default":
                        ax.annotate(q_tag, (p["bpp"], p[metric_key]),
                                    textcoords="offset points", xytext=(0, 6),
                                    ha='center', fontsize=8, fontweight='bold')
            plotted_any = True

        if not plotted_any:
            plt.close()
            return  # Nothing to plot — metric not available

        ax.set_xlabel("Bit-Per-Pixel (BPP)", fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        
        if xlim is not None:
            ax.set_xlim(xlim)
        elif all_bpps and max(all_bpps) > 0:
            ax.set_xlim(left=0, right=max(all_bpps) * 1.1)

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9, loc=legend_loc, frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{filename}.pdf"))
        plt.close()

    # ==================================================================
    # Group 1 — All models, all quality levels
    # ==================================================================
    _plot_rd_metric(model_data, "psnr",  "PSNR (dB)",
                    "RD Curves — All Models (PSNR)",
                    "rd_curve_psnr", xlim=(0, 2.5))
    _plot_rd_metric(model_data, "ssim",  "SSIM",
                    "RD Curves — All Models (SSIM)",
                    "rd_curve_ssim", xlim=(0, 2.5))
    _plot_rd_metric(model_data, "lpips", "LPIPS (Lower is Better)",
                    "RD Curves — All Models (LPIPS)",
                    "rd_curve_lpips", legend_loc="upper right", xlim=(0, 2.5))
    _plot_rd_metric(model_data, "vmaf",  "VMAF (Higher is Better)",
                    "RD Curves — All Models (VMAF)",
                    "rd_curve_vmaf", xlim=(0, 2.5))

    # ==================================================================
    # Family Breakdown 1 — HPCM (Base vs Large, MSE vs SSIM)
    # ==================================================================
    hpcm_family = {m: pts for m, pts in model_data.items() if m.startswith("HPCM")}
    if hpcm_family:
        _plot_rd_metric(hpcm_family, "psnr",  "PSNR (dB)",
                        "HPCM Family — Base vs Large & MSE vs SSIM (PSNR)",
                        "family_hpcm_psnr")
        _plot_rd_metric(hpcm_family, "ssim",  "SSIM",
                        "HPCM Family — Base vs Large & MSE vs SSIM (SSIM)",
                        "family_hpcm_ssim")
        _plot_rd_metric(hpcm_family, "lpips", "LPIPS (Lower is Better)",
                        "HPCM Family — Base vs Large & MSE vs SSIM (LPIPS)",
                        "family_hpcm_lpips", legend_loc="upper right")
        _plot_rd_metric(hpcm_family, "vmaf",  "VMAF (Higher is Better)",
                        "HPCM Family — Base vs Large & MSE vs SSIM (VMAF)",
                        "family_hpcm_vmaf")

    # ==================================================================
    # Family Breakdown 2 — StableCodec Checkpoints
    # ==================================================================
    stable_family = {m: pts for m, pts in model_data.items() if m == "StableCodec"}
    if stable_family:
        _plot_rd_metric(stable_family, "psnr",  "PSNR (dB)",
                        "StableCodec — Checkpoint Finetuning Progression (PSNR)",
                        "family_stablecodec_psnr")
        _plot_rd_metric(stable_family, "ssim",  "SSIM",
                        "StableCodec — Checkpoint Finetuning Progression (SSIM)",
                        "family_stablecodec_ssim")
        _plot_rd_metric(stable_family, "lpips", "LPIPS (Lower is Better)",
                        "StableCodec — Checkpoint Finetuning Progression (LPIPS)",
                        "family_stablecodec_lpips", legend_loc="upper right")
        _plot_rd_metric(stable_family, "vmaf",  "VMAF (Higher is Better)",
                        "StableCodec — Checkpoint Finetuning Progression (VMAF)",
                        "family_stablecodec_vmaf")

    # ==================================================================
    # Family Breakdown 3 — Standard Video Codecs (AV1, HEVC, AVC)
    # ==================================================================
    std_codecs = {"AV1", "HEVC", "AVC"}
    std_family = {m: pts for m, pts in model_data.items() if m in std_codecs}
    if std_family:
        _plot_rd_metric(std_family, "psnr",  "PSNR (dB)",
                        "Standard Codecs Comparison — AV1 vs HEVC vs AVC (PSNR)",
                        "family_standard_codecs_psnr", xlim=(0, 2.5))
        _plot_rd_metric(std_family, "ssim",  "SSIM",
                        "Standard Codecs Comparison — AV1 vs HEVC vs AVC (SSIM)",
                        "family_standard_codecs_ssim", xlim=(0, 2.5))
        _plot_rd_metric(std_family, "lpips", "LPIPS (Lower is Better)",
                        "Standard Codecs Comparison — AV1 vs HEVC vs AVC (LPIPS)",
                        "family_standard_codecs_lpips", legend_loc="upper right", xlim=(0, 2.5))
        _plot_rd_metric(std_family, "vmaf",  "VMAF (Higher is Better)",
                        "Standard Codecs Comparison — AV1 vs HEVC vs AVC (VMAF)",
                        "family_standard_codecs_vmaf", xlim=(0, 2.5))

    # ==================================================================
    # Family Breakdown 4 — Neural Learned Image Codecs
    # ==================================================================
    neural_names = {"DCVC-RT", "ELIC", "RwkvCompress", "LIC-TCM", "HPCM_Large"}
    neural_family = {m: pts for m, pts in model_data.items() if m in neural_names}
    if neural_family:
        _plot_rd_metric(neural_family, "psnr",  "PSNR (dB)",
                        "Learned Neural Codecs Comparison (PSNR)",
                        "family_neural_codecs_psnr")
        _plot_rd_metric(neural_family, "ssim",  "SSIM",
                        "Learned Neural Codecs Comparison (SSIM)",
                        "family_neural_codecs_ssim")
        _plot_rd_metric(neural_family, "lpips", "LPIPS (Lower is Better)",
                        "Learned Neural Codecs Comparison (LPIPS)",
                        "family_neural_codecs_lpips", legend_loc="upper right")
        _plot_rd_metric(neural_family, "vmaf",  "VMAF (Higher is Better)",
                        "Learned Neural Codecs Comparison (VMAF)",
                        "family_neural_codecs_vmaf")

    # ==================================================================
    # Extra — Variance shading (all models, PSNR only)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for model, stats in variance_stats.items():
        if not stats:
            continue
        bpps     = [s["mean_bpp"]  for s in stats]
        psnrs    = [s["mean_psnr"] for s in stats]
        std_psnrs = [s["std_psnr"] for s in stats]
        c  = COLOR_MAP.get(model, "#333333")
        mk = MARKER_MAP.get(model, "o")
        ax.plot(bpps, psnrs, label=model, color=c, marker=mk,
                linewidth=2.0, markersize=7)
        ax.fill_between(bpps,
                        np.array(psnrs) - np.array(std_psnrs),
                        np.array(psnrs) + np.array(std_psnrs),
                        color=c, alpha=0.15)
    ax.set_xlabel("Bit-Per-Pixel (BPP)", fontsize=12, fontweight='bold')
    ax.set_ylabel("PSNR (dB)", fontsize=12, fontweight='bold')
    ax.set_title("RD Performance with Per-Image Variance Shading (±1 Std-Dev)",
                 fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9, loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rd_curve_with_variance.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "rd_curve_with_variance.pdf"))
    plt.close()

    # ==================================================================
    # Extra — BD-Rate bar charts (all predictive models, per anchor)
    # ==================================================================
    for anchor, all_entries in bd_results.items():
        if not all_entries:
            continue
        models_list = [
            m for m, v in all_entries.items()
            if v.get("skipped_reason") != "generative_codec"
        ]
        if not models_list:
            continue

        bd_rates_val = [all_entries[m]["bd_rate_psnr"] for m in models_list]

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        colors = [
            "#2ca02c" if (r is not None and not np.isnan(r) and r < 0) else "#d62728"
            for r in bd_rates_val
        ]
        bars = ax.bar(models_list,
                      [r if (r is not None and not np.isnan(r)) else 0 for r in bd_rates_val],
                      color=colors, width=0.55, edgecolor="black", alpha=0.85)
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_ylabel(f"BD-Rate % (Relative to {anchor})", fontsize=12, fontweight='bold')
        ax.set_title(f"BD-Rate Savings / Penalty Relative to {anchor} Anchor",
                     fontsize=14, fontweight='bold', pad=10)
        ax.tick_params(axis='x', labelrotation=20)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        for bar, yval in zip(bars, bd_rates_val):
            if yval is not None and not np.isnan(yval):
                offset = 1.5 if yval >= 0 else -3.5
                ax.text(bar.get_x() + bar.get_width() / 2.0, yval + offset,
                        f"{yval:+.1f}%",
                        ha='center', va='bottom' if yval >= 0 else 'top',
                        fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"bd_rate_bar_chart_{anchor}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"bd_rate_bar_chart_{anchor}.pdf"))
        if anchor == "AV1" or (anchor == list(bd_results.keys())[0] and "AV1" not in bd_results):
            plt.savefig(os.path.join(output_dir, "bd_rate_bar_chart.png"), dpi=300)
            plt.savefig(os.path.join(output_dir, "bd_rate_bar_chart.pdf"))
        plt.close()

    print(f"SUCCESS: Generated 300 DPI publication plots in {output_dir}")

def export_reports(output_dir, model_data, bd_results, variance_stats):
    os.makedirs(output_dir, exist_ok=True)

    # --- Markdown README Snippet ---
    md_path = os.path.join(output_dir, "readme_results.md")
    with open(md_path, "w") as f:
        f.write("## Rate-Distortion, BD-Rate & Per-Image Variance Benchmark\n\n")
        f.write("> **Note:** Generative codecs (e.g. StableCodec) are excluded from BD-rate comparison.\n"
                "> BD-rate requires both codecs to perform faithful signal reconstruction on the same\n"
                "> quality axis. Diffusion-based codecs synthesise plausible images rather than\n"
                "> reconstructing pixels, making cross-paradigm BD comparisons conceptually invalid.\n\n")

        for anchor, all_entries in bd_results.items():
            if not all_entries:
                continue
            f.write(f"### BD-Rate Comparison (Anchor: `{anchor}`)\n\n")
            f.write("| Model | Type | BD-Rate (PSNR) | BD-Rate (SSIM) | BD-Rate (LPIPS) | BD-PSNR (dB) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for model, res in all_entries.items():
                codec_type_label = CODEC_TYPE.get(model, "predictive").capitalize()
                if res.get("skipped_reason") == "generative_codec":
                    f.write(f"| **{model}** | {codec_type_label} | — | — | — | — |\n")
                    continue
                r_psnr  = f"{res['bd_rate_psnr']:+.2f}%"  if res['bd_rate_psnr']  is not None and not np.isnan(res['bd_rate_psnr'])  else "N/A"
                r_ssim  = f"{res['bd_rate_ssim']:+.2f}%"  if res['bd_rate_ssim']  is not None and not np.isnan(res['bd_rate_ssim'])  else "N/A"
                r_lpips = f"{res['bd_rate_lpips']:+.2f}%" if res['bd_rate_lpips'] is not None and not np.isnan(res['bd_rate_lpips']) else "N/A"
                bd_p    = f"{res['bd_psnr_db']:+.2f} dB"  if res['bd_psnr_db']    is not None and not np.isnan(res['bd_psnr_db'])    else "N/A"
                f.write(f"| **{model}** | {codec_type_label} | {r_psnr} | {r_ssim} | {r_lpips} | {bd_p} |\n")
            f.write("\n")

    print(f"SUCCESS: Exported README Markdown snippet to {md_path}")

def run_analysis(results_dir, output_dir=None):
    results_dir = os.path.abspath(os.path.expanduser(results_dir))
    if output_dir is None:
        output_dir = os.path.join(results_dir, "analysis_report")
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    print(f"Analyzing metrics in: {results_dir}")
    model_data = load_all_metrics(results_dir)
    if not model_data:
        print(f"No metric JSON files found in {results_dir}")
        return

    variance_stats = compute_per_image_variance(model_data)
    bd_results = compute_all_bd_rates(model_data, anchors=["AV1", "HEVC", "AVC", "ELIC"])

    generate_all_plots(output_dir, model_data, variance_stats, bd_results)
    export_reports(output_dir, model_data, bd_results, variance_stats)

    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "codec_types": CODEC_TYPE,
            "model_data": model_data,
            "variance_stats": variance_stats,
            "bd_results": bd_results
        }, f, indent=4)

    print(f"SUCCESS: Analysis report complete! Saved in: {output_dir}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "./results/kodak_benchmark"
    run_analysis(target)
