"""All-station ECA&D precipitation contexts aligned to daily ERA5 precipitation targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr

from experiment_data import Normalization, SPLITS, region_mask
from full_context_data import padded_collate


def read_precipitation_stations(path: Path, splits: tuple[str, ...], era5_path: Path) -> pd.DataFrame:
    start = min(SPLITS[name][0] for name in splits)
    end = max(SPLITS[name][1] for name in splits)
    frame = pd.read_parquet(path, columns=["station_id", "date", "latitude", "longitude", "rr"]).dropna()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    with xr.open_dataset(era5_path) as ds:
        lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
        lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
    selected = (
        frame.date.between(start, end)
        & frame.latitude.between(lat_min, lat_max)
        & frame.longitude.between(lon_min, lon_max)
        & (frame.rr >= 0.0)
    )
    return frame.loc[selected].reset_index(drop=True)


def fit_precipitation_normalization(
    frame: pd.DataFrame,
    era5_path: Path,
    train_region: str,
    cutoff: float,
) -> Normalization:
    start, end = SPLITS["train"]
    selected = frame.loc[frame.date.between(start, end)]
    selected = selected.loc[region_mask(selected.longitude.to_numpy(), train_region, cutoff)]
    transformed = np.log1p(selected.rr.to_numpy(dtype="float64"))
    with xr.open_dataset(era5_path) as ds:
        coord_min = np.array([float(ds.longitude.min()), float(ds.latitude.min())], dtype="float32")
        coord_max = np.array([float(ds.longitude.max()), float(ds.latitude.max())], dtype="float32")
    return Normalization(float(transformed.mean()), float(transformed.std()), coord_min, coord_max)


class PrecipitationDataset(Dataset):
    """All valid daily stations and sampled ERA5 precipitation cells in log1p(mm/day) space."""

    def __init__(
        self,
        frame: pd.DataFrame,
        era5_path: Path,
        split: str,
        context_region: str,
        target_region: str,
        normalization: Normalization,
        targets_per_day: int = 1024,
        retention: float = 1.0,
        seed: int = 2026,
        training: bool = False,
        cutoff: float = 15.0,
        era5_variable: str = "tp",
    ) -> None:
        start, end = SPLITS[split]
        selected = frame.loc[frame.date.between(start, end)].copy()
        selected = selected.loc[region_mask(selected.longitude.to_numpy(), context_region, cutoff)]
        columns = ["station_id", "longitude", "latitude", "rr"]
        self.groups = {date: group[columns].to_numpy() for date, group in selected.groupby("date", sort=True)}
        self.ds = xr.open_dataset(era5_path)
        self.field = self.ds[era5_variable]
        units = str(self.field.attrs.get("units", "")).lower()
        if "mm" not in units:
            raise ValueError(f"Expected ERA5 precipitation in mm/day, found units={units!r}")
        era_dates = pd.DatetimeIndex(self.field.time.values).normalize()
        self.time_lookup = {date: index for index, date in enumerate(era_dates)}
        self.dates = sorted(date for date in self.groups if date in self.time_lookup)
        lon, lat = np.meshgrid(self.field.longitude.values, self.field.latitude.values)
        self.grid_coords = np.stack((lon.ravel(), lat.ravel()), axis=-1).astype("float32")
        self.target_indices = np.flatnonzero(region_mask(self.grid_coords[:, 0], target_region, cutoff))
        self.norm = normalization
        self.targets_per_day = targets_per_day
        self.retention = retention
        self.seed = seed
        self.training = training
        self.epoch = 0
        if not self.dates or not len(self.target_indices):
            raise ValueError(f"Empty precipitation dataset: {split}, {context_region}, {target_region}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _coords(self, values: np.ndarray) -> np.ndarray:
        return (2 * (values - self.norm.coord_min) / (self.norm.coord_max - self.norm.coord_min) - 1).astype("float32")

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + index + (1_000_003 * self.epoch if self.training else 0))

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        date = self.dates[index]
        rows = self.groups[date]
        station_ids = rows[:, 0].astype("uint64")
        ranks = (station_ids * np.uint64(2654435761) + np.uint64(self.seed)) & np.uint64(0xFFFFFFFF)
        count = min(len(rows), max(1, round(len(rows) * self.retention)))
        context = rows[np.argsort(ranks, kind="stable")[:count]]
        target_count = min(self.targets_per_day, len(self.target_indices))
        target_choice = self._rng(index).choice(self.target_indices, target_count, replace=False)
        reference_mm = self.field.isel(time=self.time_lookup[date]).values.reshape(-1).astype("float32")
        context_log = np.log1p(np.maximum(context[:, 3].astype("float32"), 0.0))
        target_log = np.log1p(np.maximum(reference_mm[target_choice], 0.0))
        return {
            "context_x": torch.from_numpy(self._coords(context[:, 1:3].astype("float32"))),
            "context_y": torch.from_numpy((context_log - self.norm.value_mean) / self.norm.value_std),
            "target_x": torch.from_numpy(self._coords(self.grid_coords[target_choice])),
            "target_y": torch.from_numpy((target_log - self.norm.value_mean) / self.norm.value_std),
            "date_ns": torch.tensor(date.value, dtype=torch.int64),
            "context_count": torch.tensor(count, dtype=torch.int64),
        }

