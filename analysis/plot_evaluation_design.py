"""Create the compact geographic-transfer map for the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
STYLE = Path(__file__).with_name("neurips.mplstyle")

DISPLAY_LON = (-12.0, 45.0)
DISPLAY_LAT = (34.0, 72.0)
TRANSFER_LON = 15.0

COLORS = {
    "source": "#dbe8f4",
    "target": "#f7e2cc",
}


def draw_domain(ax: plt.Axes) -> None:
    data_crs = ccrs.PlateCarree()
    ax.set_extent([*DISPLAY_LON, *DISPLAY_LAT], crs=data_crs)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f7f7f7", zorder=0)
    ax.add_patch(
        Rectangle(
            (DISPLAY_LON[0], DISPLAY_LAT[0]),
            TRANSFER_LON - DISPLAY_LON[0],
            DISPLAY_LAT[1] - DISPLAY_LAT[0],
            facecolor=COLORS["source"],
            edgecolor="none",
            alpha=0.78,
            transform=data_crs,
            zorder=1,
        )
    )
    ax.add_patch(
        Rectangle(
            (TRANSFER_LON, DISPLAY_LAT[0]),
            DISPLAY_LON[1] - TRANSFER_LON,
            DISPLAY_LAT[1] - DISPLAY_LAT[0],
            facecolor=COLORS["target"],
            edgecolor="none",
            alpha=0.78,
            transform=data_crs,
            zorder=1,
        )
    )
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#777777", linewidth=0.35, zorder=2)
    ax.coastlines(resolution="50m", color="#333333", linewidth=0.55, zorder=2)

    ax.plot(
        [TRANSFER_LON, TRANSFER_LON],
        [DISPLAY_LAT[0], DISPLAY_LAT[1]],
        color="#222222",
        linestyle="--",
        linewidth=1.15,
        transform=data_crs,
        zorder=4,
    )
    ax.set_xticks([-10, 0, 10, 20, 30, 40], crs=data_crs)
    ax.set_yticks([35, 45, 55, 65], crs=data_crs)
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.gridlines(crs=data_crs, color="white", linewidth=0.5, alpha=0.8, zorder=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/generated_figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if STYLE.exists():
        plt.style.use(STYLE)

    fig = plt.figure(figsize=(3.35, 2.15))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    draw_domain(ax)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / "evaluation_design.png"
    pdf = args.output_dir / "evaluation_design.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
