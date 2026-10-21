"""Create the NeurIPS-width 1x4 station-sparsity figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "paper_results" / "sparsity_results.csv"
OUTPUT_DIR = ROOT / "analysis" / "generated_figures"

MODEL_ORDER = ["ordinary_kriging", "convcnp", "latent_transformer"]
MODEL_LABELS = {
    "ordinary_kriging": "Kriging",
    "convcnp": "ConvCNP",
    "latent_transformer": "Latent-grid Transformer",
}
MODEL_COLORS = {
    "ordinary_kriging": "#4C72B0",
    "convcnp": "#DD8452",
    "latent_transformer": "#55A868",
}
NEURAL_MODELS = {"convcnp", "latent_transformer"}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman No9 L",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 6.7,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "lines.linewidth": 1.25,
            "lines.markersize": 2.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def load_results() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS_PATH)
    selected = frame.loc[
        (frame["training_experiment"] == "global")
        & (frame["calibration"] == "global")
        & (frame["calibration_region"] == "global")
        & (frame["test_region"] == "global")
        & (frame["subset"] == "all")
        & (frame["calibration_mode"] == "deployed_full_density")
    ].copy()
    selected["retention_pct"] = 100.0 * selected["retention"]

    expected_rows = 2 * len(MODEL_ORDER) * 5
    if len(selected) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} global sparsity rows, found {len(selected)}."
        )
    if selected.duplicated(["variable", "model", "retention"]).any():
        raise RuntimeError("Duplicate variable/model/retention rows in sparsity results.")
    return selected


def draw_series(
    ax: mpl.axes.Axes,
    frame: pd.DataFrame,
    metric: str,
) -> None:
    for model in MODEL_ORDER:
        series = frame.loc[frame["model"] == model].sort_values("retention_pct")
        x = series["retention_pct"].to_numpy(dtype=float)
        y = series[f"{metric}_mean"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            marker="o",
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            zorder=3,
        )

        if model in NEURAL_MODELS:
            low = series[f"{metric}_ci95_low"].to_numpy(dtype=float)
            high = series[f"{metric}_ci95_high"].to_numpy(dtype=float)
            if not (np.isfinite(low).all() and np.isfinite(high).all()):
                raise RuntimeError(f"Missing {metric} confidence band for {model}.")
            ax.fill_between(
                x,
                low,
                high,
                color=MODEL_COLORS[model],
                alpha=0.17,
                linewidth=0,
                zorder=2,
            )


def style_axis(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#B0B0B0", linewidth=0.4, alpha=0.30)
    ax.set_axisbelow(True)
    ax.set_xticks([10, 25, 50, 75, 100])
    ax.set_xlim(8, 102)


def main() -> None:
    configure_style()
    sparse = load_results()

    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.85))
    panel_specs = [
        ("temperature", "rmse", "(a) Temperature: RMSE", r"RMSE ($^\circ$C)"),
        (
            "temperature",
            "conformal_coverage",
            "(b) Temperature: Coverage",
            "Empirical coverage",
        ),
        ("precipitation", "rmse", "(c) Precipitation: RMSE", "RMSE (mm/day)"),
        (
            "precipitation",
            "conformal_coverage",
            "(d) Precipitation: Coverage",
            "Empirical coverage",
        ),
    ]

    for ax, (variable, metric, title, ylabel) in zip(axes, panel_specs):
        variable_rows = sparse.loc[sparse["variable"] == variable]
        draw_series(ax, variable_rows, metric)

        if metric == "conformal_coverage":
            ax.axhline(
                0.90,
                color="black",
                linestyle=(0, (4, 2.4)),
                linewidth=0.85,
                zorder=1,
            )
            ax.set_ylim(0.68, 0.94)
            ax.set_yticks([0.70, 0.80, 0.90])

        ax.set_title(title, pad=2.4)
        ax.set_xlabel("Stations retained (%)", labelpad=1.5)
        ax.set_ylabel(ylabel, labelpad=1.8)
        style_axis(ax)

    legend_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            marker="o",
            label=MODEL_LABELS[model],
        )
        for model in MODEL_ORDER
    ]
    legend_handles.append(
        mpl.lines.Line2D(
            [0],
            [0],
            color="black",
            linestyle=(0, (4, 2.4)),
            linewidth=0.85,
            label="Nominal 90% coverage",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        handlelength=1.75,
        columnspacing=0.78,
        handletextpad=0.32,
    )

    fig.subplots_adjust(
        left=0.070,
        right=0.995,
        bottom=0.225,
        top=0.755,
        wspace=0.68,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "station_sparsity_1x4_neurips.png"
    pdf_path = OUTPUT_DIR / "station_sparsity_1x4_neurips.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
