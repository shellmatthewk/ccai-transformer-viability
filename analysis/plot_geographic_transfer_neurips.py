"""Create the NeurIPS-style geographic-transfer error map for seed 2026.

The available spatial exports contain one neural seed. The figure is therefore
an illustrative spatial diagnostic; aggregate transfer claims must use the
eight-seed summary tables.
"""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "figure_data"
OUTPUT_DIR = ROOT / "analysis" / "generated_figures"

MODEL_ORDER = ["ordinary_kriging", "convcnp", "latent_transformer"]
MODEL_LABELS = {
    "ordinary_kriging": "Kriging",
    "convcnp": "ConvCNP",
    "latent_transformer": "Latent-grid Transformer",
}

VARIABLES = {
    "temperature": {
        "path": DATA_DIR / "temperature_seed_2026_transfer_retention_1.parquet",
        "truth": "truth_c",
        "prediction": {
            "ordinary_kriging": "ordinary_kriging_mean_c",
            "convcnp": "convcnp_mean_c",
            "latent_transformer": "latent_transformer_mean_c",
        },
        "row_label": "Temperature",
        "colorbar_label": r"Mean absolute error ($^\circ$C)",
    },
    "precipitation": {
        "path": DATA_DIR / "precipitation_seed_2026_transfer_retention_1.parquet",
        "truth": "truth_mm_day",
        "prediction": {
            "ordinary_kriging": "ordinary_kriging_median_mm_day",
            "convcnp": "convcnp_median_mm_day",
            "latent_transformer": "latent_transformer_median_mm_day",
        },
        "row_label": "Precipitation",
        "colorbar_label": "Mean absolute error (mm/day)",
        "fixed_vmax": 5.0,
        "colorbar_ticks": np.arange(0.0, 5.1, 1.0),
    },
}


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
            "axes.titlesize": 7.5,
            "axes.labelsize": 6.7,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "axes.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def land_mask(longitudes: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    land_path = shpreader.natural_earth(
        resolution="50m",
        category="physical",
        name="land",
    )
    land = unary_union(list(shpreader.Reader(land_path).geometries()))
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    return shapely.contains_xy(land, lon_grid, lat_grid)


def aggregate_spatial_error(config: dict) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    required = ["longitude", "latitude", config["truth"], *config["prediction"].values()]
    frame = pd.read_parquet(config["path"], columns=required)

    longitude = np.sort(frame["longitude"].unique())
    latitude = np.sort(frame["latitude"].unique())
    index = pd.MultiIndex.from_product(
        [latitude, longitude],
        names=["latitude", "longitude"],
    )

    spatial_errors: dict[str, np.ndarray] = {}
    for model in MODEL_ORDER:
        error = (frame[config["truth"]] - frame[config["prediction"][model]]).abs()
        aggregated = (
            frame.assign(abs_error=error)
            .groupby(["latitude", "longitude"], sort=True)["abs_error"]
            .mean()
            .reindex(index)
        )
        spatial_errors[model] = aggregated.to_numpy(dtype=float).reshape(
            len(latitude), len(longitude)
        )
    return longitude, latitude, spatial_errors


def main() -> None:
    configure_style()

    prepared = {}
    for variable, config in VARIABLES.items():
        longitude, latitude, errors = aggregate_spatial_error(config)
        mask = land_mask(longitude, latitude)
        masked_errors = {
            model: np.where(mask, values, np.nan) for model, values in errors.items()
        }
        finite_values = np.concatenate(
            [values[np.isfinite(values)] for values in masked_errors.values()]
        )
        prepared[variable] = {
            "longitude": longitude,
            "latitude": latitude,
            "errors": masked_errors,
            "vmax": float(
                config.get("fixed_vmax", np.quantile(finite_values, 0.95))
            ),
        }

    fig = plt.figure(figsize=(5.5, 3.05))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1, 1, 1, 0.045],
        left=0.075,
        right=0.955,
        bottom=0.035,
        top=0.925,
        wspace=0.025,
        hspace=0.045,
    )

    axes = np.empty((2, 3), dtype=object)
    row_mappables = []
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad(alpha=0.0)

    for row, variable in enumerate(["temperature", "precipitation"]):
        config = VARIABLES[variable]
        item = prepared[variable]
        mappable = None

        for col, model in enumerate(MODEL_ORDER):
            ax = fig.add_subplot(grid[row, col], projection=ccrs.PlateCarree())
            axes[row, col] = ax
            ax.set_extent([-12, 45, 34, 72], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
            ax.add_feature(cfeature.LAND, facecolor="#EEEEEE", edgecolor="none", zorder=0)

            mappable = ax.pcolormesh(
                item["longitude"],
                item["latitude"],
                item["errors"][model],
                transform=ccrs.PlateCarree(),
                shading="nearest",
                cmap=cmap,
                vmin=0.0,
                vmax=item["vmax"],
                rasterized=True,
                zorder=1,
            )
            ax.coastlines(resolution="50m", color="#222222", linewidth=0.38, zorder=3)
            ax.add_feature(
                cfeature.BORDERS,
                edgecolor="#555555",
                linewidth=0.22,
                zorder=3,
            )
            ax.plot(
                [15, 15],
                [34, 72],
                transform=ccrs.PlateCarree(),
                color="#444444",
                linewidth=0.55,
                linestyle=(0, (3, 2)),
                zorder=4,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(MODEL_LABELS[model], pad=2.0)

        if mappable is None:
            raise RuntimeError(f"No map produced for {variable}.")
        row_mappables.append(mappable)

        cax = fig.add_subplot(grid[row, 3])
        colorbar = fig.colorbar(
            mappable,
            cax=cax,
            extend="max",
            ticks=config.get("colorbar_ticks"),
        )
        colorbar.set_label(config["colorbar_label"], labelpad=2.2)
        colorbar.ax.tick_params(labelsize=5.8, width=0.45, length=2.0, pad=1.5)
        colorbar.outline.set_linewidth(0.45)

    fig.text(
        0.035,
        0.705,
        VARIABLES["temperature"]["row_label"],
        rotation=90,
        va="center",
        ha="center",
        fontsize=7.2,
    )
    fig.text(
        0.035,
        0.270,
        VARIABLES["precipitation"]["row_label"],
        rotation=90,
        va="center",
        ha="center",
        fontsize=7.2,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "geographic_transfer_seed2026_neurips.png"
    pdf_path = OUTPUT_DIR / "geographic_transfer_seed2026_neurips.pdf"
    save_options = {"bbox_inches": "tight", "pad_inches": 0.02, "facecolor": "white"}
    fig.savefig(png_path, dpi=300, **save_options)
    fig.savefig(pdf_path, **save_options)
    plt.close(fig)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
