#!/usr/bin/env python3
"""Plot MPCC manipulator metrics for exponent-scaling comparison.

Baseline:
    final_truncate_addertree_all
Treatment:
    final_truncate_addertree_all_exp_scaling

The script reads existing *_debug_data.mat and *_time_data.mat files and writes
paper-friendly comparison figures plus a CSV/Markdown metric summary.
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings(
    "ignore",
    message=r"A NumPy version .* is required for this version of SciPy.*",
    category=UserWarning,
)
import scipy.io


BASE_VARIANT = "final_truncate_addertree_all"
EXP_VARIANT = "final_truncate_addertree_all_exp_scaling"
REV_VARIANT = "final_truncate_addertree_all_exp_scaling_rev_permute"


@dataclass(frozen=True)
class VariantData:
    name: str
    label: str
    debug: dict[str, np.ndarray]
    time: dict[str, np.ndarray]


def load_variant(result_root: Path, name: str, label: str) -> VariantData:
    return load_variant_dir(result_root / name, name, label)


def load_variant_dir(folder: Path, name: str, label: str) -> VariantData:
    debug_path = folder / f"{name}_debug_data.mat"
    time_path = folder / f"{name}_time_data.mat"
    if not debug_path.exists():
        raise FileNotFoundError(debug_path)
    if not time_path.exists():
        raise FileNotFoundError(time_path)

    debug = {
        key: np.asarray(value)
        for key, value in scipy.io.loadmat(debug_path).items()
        if not key.startswith("__")
    }
    time = {
        key: np.asarray(value)
        for key, value in scipy.io.loadmat(time_path).items()
        if not key.startswith("__")
    }
    return VariantData(name=name, label=label, debug=debug, time=time)


def as_series(arr: np.ndarray, length: int | None = None) -> np.ndarray:
    """Return a 1-D per-solve series for scalar metrics."""
    arr = np.asarray(arr)
    if arr.ndim == 2 and 1 in arr.shape:
        out = arr.reshape(-1)
    elif arr.ndim == 1:
        out = arr
    else:
        raise ValueError(f"Expected scalar series, got shape {arr.shape}")
    return out[:length] if length is not None else out


def env_min_series(debug: dict[str, np.ndarray], length: int | None = None) -> np.ndarray:
    """Return the nearest environment-obstacle distance per solve."""
    env = np.asarray(debug["env_min_dist"])
    if env.ndim < 2:
        out = env.reshape(-1)
    else:
        out = env.reshape(env.shape[0], -1).min(axis=1)
    return out[:length] if length is not None else out


def common_length(variants: list[VariantData]) -> int:
    lengths = []
    for variant in variants:
        lengths.append(as_series(variant.debug["mani"]).size)
        lengths.append(as_series(variant.time["total"]).size)
    return min(lengths)


def pct_change(value: float, baseline: float) -> float:
    if baseline == 0:
        return math.nan
    return 100.0 * (value - baseline) / baseline


def series_stats(series: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
        "min": float(np.min(series)),
        "p95": float(np.percentile(series, 95)),
        "max": float(np.max(series)),
        "rms": float(np.sqrt(np.mean(np.square(series)))),
    }


def collect_metrics(variants: list[VariantData], length: int) -> list[dict[str, str | float]]:
    definitions = [
        ("mani_mean", "Manipulability mean", "debug", "mani", "mean", 1.0),
        ("contour_error_mean", "Contour error mean", "debug", "contour_error", "mean", 1.0),
        ("contour_error_rms", "Contour error RMS", "debug", "contour_error", "rms", 1.0),
        ("self_dist_mean", "Self-collision distance mean", "debug", "sel_min_dist", "mean", 1.0),
        ("self_dist_min", "Self-collision distance min", "debug", "sel_min_dist", "min", 1.0),
        ("env_dist_nearest_mean", "Nearest environment distance mean", "env_min", "", "mean", 1.0),
        ("env_dist_nearest_min", "Nearest environment distance min", "env_min", "", "min", 1.0),
        ("ee_speed_mean", "End-effector speed mean", "debug", "ee_speed", "mean", 1.0),
        ("admm_iter_mean", "ADMM iterations mean", "debug", "iter_count", "mean", 1.0),
        ("admm_iter_p95", "ADMM iterations p95", "debug", "iter_count", "p95", 1.0),
        ("admm_iter_max", "ADMM iterations max", "debug", "iter_count", "max", 1.0),
        ("rho_updates_mean", "Rho updates mean", "time", "rho_updates", "mean", 1.0),
        ("total_time_ms_mean", "Total time mean (ms)", "time", "total", "mean", 1000.0),
        ("total_time_ms_p95", "Total time p95 (ms)", "time", "total", "p95", 1000.0),
        ("solve_qp_ms_mean", "Solve QP mean (ms)", "time", "solve_qp", "mean", 1000.0),
        ("scaling_ms_mean", "Scaling time mean (ms)", "time", "scaling_time", "mean", 1000.0),
        ("init_solver_ms_mean", "Init solver mean (ms)", "time", "init_solver", "mean", 1000.0),
        ("factorization_ms_mean", "Factorization mean (ms)", "time", "factorization_time", "mean", 1000.0),
        ("permutation_ms_mean", "Permutation mean (ms)", "time", "permutation_time", "mean", 1000.0),
    ]

    baseline_values: dict[str, float] = {}
    rows: list[dict[str, str | float]] = []
    for variant in variants:
        for metric_id, label, source, key, stat_name, scale in definitions:
            if source == "debug":
                data = as_series(variant.debug[key], length)
            elif source == "time":
                data = as_series(variant.time[key], length)
            elif source == "env_min":
                data = env_min_series(variant.debug, length)
            else:
                raise ValueError(source)
            value = series_stats(data)[stat_name] * scale
            if variant.name == BASE_VARIANT:
                baseline_values[metric_id] = value
                change = 0.0
            else:
                change = pct_change(value, baseline_values[metric_id])

            rows.append(
                {
                    "variant": variant.name,
                    "label": variant.label,
                    "metric": metric_id,
                    "metric_label": label,
                    "value": value,
                    "baseline_value": baseline_values[metric_id],
                    "relative_change_percent": change,
                    "common_samples": length,
                }
            )
    return rows


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_primary_change(rows: list[dict[str, str | float]], out_dir: Path) -> None:
    metric_order = [
        "mani_mean",
        "contour_error_mean",
        "env_dist_nearest_min",
        "self_dist_min",
        "admm_iter_mean",
        "total_time_ms_mean",
        "solve_qp_ms_mean",
        "scaling_ms_mean",
    ]
    exp_rows = {
        str(row["metric"]): row
        for row in rows
        if row["variant"] == EXP_VARIANT and row["metric"] in metric_order
    }
    labels = [str(exp_rows[m]["metric_label"]) for m in metric_order]
    changes = [float(exp_rows[m]["relative_change_percent"]) for m in metric_order]

    fig, ax = plt.subplots(figsize=(7.3, 3.6))
    y = np.arange(len(labels))
    colors = ["#3f6f8f" if value >= 0 else "#a35d3a" for value in changes]
    ax.barh(y, changes, color=colors, height=0.62)
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Relative change from baseline (%)")
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.6, alpha=0.8)
    left = min(changes + [0.0])
    right = max(changes + [0.0])
    ax.set_xlim(left - 0.16 * (right - left), right + 0.12 * (right - left))
    for idx, value in enumerate(changes):
        label = f"{value:+.3f}%" if abs(value) < 0.1 else f"{value:+.2f}%"
        if value <= -2.0:
            ax.text(value * 0.5, idx, label, va="center", ha="center", color="white")
        elif value < 0.0:
            ax.text(value - 0.35, idx, label, va="center", ha="right")
        else:
            ax.text(value + 0.35, idx, label, va="center", ha="left")
    save_figure(fig, out_dir / "exp_scaling_primary_metric_change")


def plot_timeseries(base: VariantData, exp: VariantData, length: int, out_dir: Path) -> None:
    x = np.arange(length)
    panels = [
        ("Manipulability", as_series(base.debug["mani"], length), as_series(exp.debug["mani"], length), ""),
        ("Contour error", as_series(base.debug["contour_error"], length), as_series(exp.debug["contour_error"], length), ""),
        ("ADMM iterations", as_series(base.debug["iter_count"], length), as_series(exp.debug["iter_count"], length), ""),
        ("Total time", as_series(base.time["total"], length) * 1000.0, as_series(exp.time["total"], length) * 1000.0, "ms"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    for ax, (title, base_series, exp_series, unit) in zip(axes.flat, panels):
        ax.plot(x, base_series, color="#4d4d4d", linewidth=0.9, label="Baseline")
        ax.plot(x, exp_series, color="#1f77b4", linewidth=0.9, label="Exp scaling")
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(color="#d0d0d0", linewidth=0.5, alpha=0.7)
    axes[-1, 0].set_xlabel("Solve index")
    axes[-1, 1].set_xlabel("Solve index")
    axes[0, 0].legend(loc="best")
    save_figure(fig, out_dir / "exp_scaling_timeseries")


def plot_tracking_overlay_with_means(base: VariantData, exp: VariantData, length: int, out_dir: Path) -> None:
    """Plot baseline and exp-scaling tracking-step values with mean markers."""
    x = np.arange(length)
    panels = [
        (
            "Manipulability",
            as_series(base.debug["mani"], length),
            as_series(exp.debug["mani"], length),
            "",
            "higher is better",
        ),
        (
            "Contour error",
            as_series(base.debug["contour_error"], length),
            as_series(exp.debug["contour_error"], length),
            "",
            "lower is better",
        ),
        (
            "Self-collision distance",
            as_series(base.debug["sel_min_dist"], length),
            as_series(exp.debug["sel_min_dist"], length),
            "",
            "higher is better",
        ),
        (
            "Nearest environment distance",
            env_min_series(base.debug, length),
            env_min_series(exp.debug, length),
            "",
            "higher is better",
        ),
        (
            "ADMM iterations",
            as_series(base.debug["iter_count"], length),
            as_series(exp.debug["iter_count"], length),
            "",
            "lower is better",
        ),
        (
            "Total time",
            as_series(base.time["total"], length) * 1000.0,
            as_series(exp.time["total"], length) * 1000.0,
            "ms",
            "lower is better",
        ),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(7.4, 7.2), sharex=True)
    base_color = "#4d4d4d"
    exp_color = "#1f77b4"
    for ax, (title, base_series, exp_series, unit, direction) in zip(axes.flat, panels):
        base_mean = float(np.mean(base_series))
        exp_mean = float(np.mean(exp_series))
        change = pct_change(exp_mean, base_mean)
        unit_label = f" ({unit})" if unit else ""

        ax.plot(x, base_series, color=base_color, linewidth=0.95, alpha=0.88, label="Baseline step")
        ax.plot(
            x,
            exp_series,
            color=exp_color,
            linewidth=0.85,
            linestyle=(0, (4, 1.5)),
            alpha=0.92,
            label="Exp scaling step",
        )
        ax.axhline(base_mean, color=base_color, linewidth=1.1, linestyle="--", alpha=0.85, label="Baseline mean")
        ax.axhline(exp_mean, color=exp_color, linewidth=1.1, linestyle=":", alpha=0.9, label="Exp scaling mean")
        ax.set_title(title)
        ax.set_ylabel(f"Value{unit_label}")
        ax.grid(color="#d0d0d0", linewidth=0.5, alpha=0.7)
        change_label = f"{change:+.3f}%" if abs(change) < 0.1 else f"{change:+.2f}%"
        ax.text(
            0.99,
            0.97,
            f"mean {change_label}\n{direction}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.82, "pad": 2.5},
        )
    axes[-1, 0].set_xlabel("Tracking step")
    axes[-1, 1].set_xlabel("Tracking step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out_dir / "exp_scaling_tracking_overlay_with_means")


def plot_performance_metrics_style_comparison(base: VariantData, exp: VariantData, length: int, out_dir: Path) -> None:
    """Recreate the original *_performance_metrics.png layout with both variants."""
    slecol_buffer = 1.0
    mani_buffer = 0.018
    base_label = base.label
    exp_label = exp.label

    ee_base = as_series(base.debug["ee_speed"], length)
    ee_exp = as_series(exp.debug["ee_speed"], length)
    sel_base = as_series(base.debug["sel_min_dist"], length)
    sel_exp = as_series(exp.debug["sel_min_dist"], length)
    mani_base = as_series(base.debug["mani"], length)
    mani_exp = as_series(exp.debug["mani"], length)
    contour_base = as_series(base.debug["contour_error"], length)
    contour_exp = as_series(exp.debug["contour_error"], length)

    fig = plt.figure(figsize=(14, 8))
    fig.subplots_adjust(hspace=1)

    ax = plt.subplot(411)
    ax.plot(ee_base, label=f"ee_speed ({base_label})", color="#555555", linewidth=1.4)
    ax.plot(ee_exp, label=f"ee_speed ({exp_label})", color="red", linewidth=1.2, linestyle="--")
    ax.set_xlabel("s (m)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("EE Speed per Arc length")
    ax.set_ylim(-0.01, max(ee_base.max(), ee_exp.max(), as_series(base.debug["vs"], length).max(), as_series(exp.debug["vs"], length).max()) * 1.2)
    ax.legend()
    ax.grid(True)

    ax = plt.subplot(412)
    ax.plot(sel_base, label=f"minimum distance ({base_label})", color="#555555", linewidth=1.4)
    ax.plot(sel_exp, label=f"minimum distance ({exp_label})", color="blue", linewidth=1.2, linestyle="--")
    ax.axhline(y=slecol_buffer, color="black", linestyle="--", label="buffer")
    ax.set_xlabel("s (m)")
    ax.set_ylabel("distance (cm)")
    ax.set_title("Minimum distance per Arc length")
    ax.set_ylim(-0.01, max(sel_base.max(), sel_exp.max()) * 1.2)
    ax.legend()
    ax.grid(True)

    ax = plt.subplot(413)
    ax.plot(mani_base, label=f"manip ({base_label})", color="#555555", linewidth=1.4)
    ax.plot(mani_exp, label=f"manip ({exp_label})", color="blue", linewidth=1.2, linestyle="--")
    ax.axhline(y=mani_buffer, color="black", linestyle="--", label="buffer")
    ax.set_xlabel("s (m)")
    ax.set_ylabel("Manipulability")
    ax.set_title("Manipulability per Arc length")
    ax.set_ylim(-0.01, max(mani_base.max(), mani_exp.max()) * 1.2)
    ax.legend()
    ax.grid(True)

    ax = plt.subplot(414)
    ax.plot(contour_base, label=f"Contour Error ({base_label})", color="#555555", linewidth=1.4)
    ax.plot(contour_exp, label=f"Contour Error ({exp_label})", color="blue", linewidth=1.2, linestyle="--")
    ax.set_xlabel("s (m)")
    ax.set_ylabel("Error (m)")
    ax.set_title("Contouring Error per Arc length")
    ax.set_ylim(-max(contour_base.max(), contour_exp.max()) * 0.3, max(contour_base.max(), contour_exp.max()) * 1.2)
    ax.legend()
    ax.grid(True)

    save_figure(fig, out_dir / "exp_scaling_performance_metrics_overlay")


def plot_original_style_performance_overlay(base: VariantData, exp: VariantData, length: int, out_dir: Path) -> None:
    """Overlay both variants using the exact original performance_metrics layout."""
    slecol_buffer = 1.0
    mani_buffer = 0.018

    ee_base = as_series(base.debug["ee_speed"], length)
    ee_exp = as_series(exp.debug["ee_speed"], length)
    vs_base = as_series(base.debug["vs"], length)
    vs_exp = as_series(exp.debug["vs"], length)
    sel_base = as_series(base.debug["sel_min_dist"], length)
    sel_exp = as_series(exp.debug["sel_min_dist"], length)
    mani_base = as_series(base.debug["mani"], length)
    mani_exp = as_series(exp.debug["mani"], length)
    contour_base = as_series(base.debug["contour_error"], length)
    contour_exp = as_series(exp.debug["contour_error"], length)

    with plt.rc_context(plt.rcParamsDefault):
        fig = plt.figure(figsize=(14, 8))
        fig.subplots_adjust(hspace=1)

        plt.subplot(411)
        plt.plot(ee_base, label=f"ee_speed ({base.label})", color="r")
        plt.plot(ee_exp, label=f"ee_speed ({exp.label})", color="r", linestyle="--")
        plt.xlabel("s (m)")
        plt.ylabel("Speed (m/s)")
        plt.title("EE Speed per Arc length")
        plt.ylim(-0.01, max(ee_base.max(), ee_exp.max(), vs_base.max(), vs_exp.max()) * 1.2)
        plt.legend()
        plt.grid(True)

        plt.subplot(412)
        plt.plot(sel_base, label=f"minimum distance ({base.label})", color="b")
        plt.plot(sel_exp, label=f"minimum distance ({exp.label})", color="b", linestyle="--")
        plt.axhline(y=slecol_buffer, color="black", linestyle="--", label="buffer")
        plt.xlabel("s (m)")
        plt.ylabel("distance (cm)")
        plt.title("Minimum distance per Arc length")
        plt.ylim(-0.01, max(sel_base.max(), sel_exp.max()) * 1.2)
        plt.legend()
        plt.grid(True)

        plt.subplot(413)
        plt.plot(mani_base, label=f"manip ({base.label})", color="b")
        plt.plot(mani_exp, label=f"manip ({exp.label})", color="b", linestyle="--")
        plt.axhline(y=mani_buffer, color="black", linestyle="--", label="buffer")
        plt.xlabel("s (m)")
        plt.ylabel("Manipulability")
        plt.title("Manipulability per Arc length")
        plt.ylim(-0.01, max(mani_base.max(), mani_exp.max()) * 1.2)
        plt.legend()
        plt.grid(True)

        plt.subplot(414)
        plt.plot(contour_base, label=f"Contour Error ({base.label})", color="b")
        plt.plot(contour_exp, label=f"Contour Error ({exp.label})", color="b", linestyle="--")
        plt.xlabel("s (m)")
        plt.ylabel("Error (m)")
        plt.title("Contouring Error per Arc length")
        plt.ylim(-max(contour_base.max(), contour_exp.max()) * 0.3, max(contour_base.max(), contour_exp.max()) * 1.2)
        plt.legend()
        plt.grid(True)

        fig.savefig(out_dir / "exp_scaling_performance_metrics_overlay_exact_format.png")
        fig.savefig(out_dir / "exp_scaling_performance_metrics_overlay_exact_format.pdf")
        plt.close(fig)


def plot_runtime_breakdown(base: VariantData, exp: VariantData, length: int, out_dir: Path) -> None:
    components = [
        ("Set env", "set_env"),
        ("Set QP", "set_qp"),
        ("Init solver", "init_solver"),
        ("Solve QP", "solve_qp"),
        ("Scaling", "scaling_time"),
        ("Permutation", "permutation_time"),
        ("Factorization", "factorization_time"),
        ("Get alpha", "get_alpha"),
    ]
    x = np.arange(len(components))
    width = 0.36
    base_vals = [np.mean(as_series(base.time[key], length)) * 1000.0 for _, key in components]
    exp_vals = [np.mean(as_series(exp.time[key], length)) * 1000.0 for _, key in components]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    for ax, title, limit_to_solver in [
        (axes[0], "Runtime breakdown", False),
        (axes[1], "Solver-side components", True),
    ]:
        ax.bar(x - width / 2, base_vals, width, label="Baseline", color="#4d4d4d")
        ax.bar(x + width / 2, exp_vals, width, label="Exp scaling", color="#1f77b4")
        ax.set_ylabel("Mean time (ms)")
        ax.set_title(title)
        ax.grid(axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
        if limit_to_solver:
            solver_max = max(base_vals[1:] + exp_vals[1:])
            ax.set_ylim(0, solver_max * 1.35)
        ax.legend(loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([label for label, _ in components], rotation=25, ha="right")
    save_figure(fig, out_dir / "exp_scaling_runtime_breakdown")


def plot_iteration_distribution(base: VariantData, exp: VariantData, length: int, out_dir: Path) -> None:
    base_iter = as_series(base.debug["iter_count"], length)
    exp_iter = as_series(exp.debug["iter_count"], length)

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    bins = np.arange(0, max(base_iter.max(), exp_iter.max()) + 25, 25)
    ax.hist(base_iter, bins=bins, density=True, histtype="step", linewidth=1.6, color="#4d4d4d", label="Baseline")
    ax.hist(exp_iter, bins=bins, density=True, histtype="step", linewidth=1.6, color="#1f77b4", label="Exp scaling")
    ax.set_xlabel("ADMM iterations")
    ax.set_ylabel("Density")
    ax.grid(color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper right")
    save_figure(fig, out_dir / "exp_scaling_iteration_distribution")


def write_summary(rows: list[dict[str, str | float]], out_dir: Path) -> None:
    csv_path = out_dir / "exp_scaling_comparison_summary.csv"
    fieldnames = [
        "variant",
        "label",
        "metric",
        "metric_label",
        "value",
        "baseline_value",
        "relative_change_percent",
        "common_samples",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "exp_scaling_comparison_summary.md"
    exp_rows = [row for row in rows if row["variant"] == EXP_VARIANT]
    with md_path.open("w") as f:
        f.write("# Exponent Scaling Comparison\n\n")
        f.write(f"Baseline: `{BASE_VARIANT}`\n\n")
        f.write(f"Treatment: `{EXP_VARIANT}`\n\n")
        f.write("| Metric | Baseline | Exp scaling | Change |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in exp_rows:
            f.write(
                "| {metric_label} | {baseline:.6g} | {value:.6g} | {change:+.3f}% |\n".format(
                    metric_label=row["metric_label"],
                    baseline=float(row["baseline_value"]),
                    value=float(row["value"]),
                    change=float(row["relative_change_percent"]),
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("result/main_w_sim_1"),
        help="Directory containing final_truncate_addertree_all* result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/main_w_sim_1/exp_scaling_comparison"),
        help="Directory where comparison figures and tables are written.",
    )
    parser.add_argument(
        "--baseline-variant",
        default=BASE_VARIANT,
        help="Baseline result folder name under --result-root.",
    )
    parser.add_argument(
        "--exp-variant",
        default=EXP_VARIANT,
        help="Exponent-scaling result folder name under --result-root.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        help="Exact baseline result directory. Overrides --result-root for baseline.",
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        help="Exact exponent-scaling result directory. Overrides --result-root for exp.",
    )
    return parser.parse_args()


def main() -> None:
    global BASE_VARIANT, EXP_VARIANT
    args = parse_args()
    BASE_VARIANT = args.baseline_variant
    EXP_VARIANT = args.exp_variant
    set_paper_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline_dir:
        base = load_variant_dir(args.baseline_dir, BASE_VARIANT, "Baseline")
    else:
        base = load_variant(args.result_root, BASE_VARIANT, "Baseline")
    if args.exp_dir:
        exp = load_variant_dir(args.exp_dir, EXP_VARIANT, "Exp scaling")
    else:
        exp = load_variant(args.result_root, EXP_VARIANT, "Exp scaling")
    variants = [base, exp]
    length = common_length(variants)

    rows = collect_metrics(variants, length)
    write_summary(rows, args.output_dir)
    plot_primary_change(rows, args.output_dir)
    plot_timeseries(base, exp, length, args.output_dir)
    plot_tracking_overlay_with_means(base, exp, length, args.output_dir)
    plot_performance_metrics_style_comparison(base, exp, length, args.output_dir)
    plot_original_style_performance_overlay(base, exp, length, args.output_dir)
    plot_runtime_breakdown(base, exp, length, args.output_dir)
    plot_iteration_distribution(base, exp, length, args.output_dir)

    print(f"Wrote comparison artifacts to {args.output_dir}")
    print(f"Common sample count: {length}")


if __name__ == "__main__":
    main()
