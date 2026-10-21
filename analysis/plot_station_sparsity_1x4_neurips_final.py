"""Render the final, compact 1x4 station-sparsity paper figure."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from plot_station_sparsity_1x4_neurips import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    OUTPUT_DIR,
    configure_style,
    draw_series,
    load_results,
    style_axis,
)


def main() -> None:
    configure_style()
    sparse = load_results()
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.85))

    panel_specs = [
        ("temperature", "rmse", "(a) Temperature RMSE", r"RMSE ($^\circ$C)"),
        (
            "temperature",
            "conformal_coverage",
            "(b) Temperature coverage",
            "Empirical coverage",
        ),
        ("precipitation", "rmse", "(c) Precipitation RMSE", "RMSE (mm/day)"),
        (
            "precipitation",
            "conformal_coverage",
            "(d) Precipitation coverage",
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
        right=0.985,
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
