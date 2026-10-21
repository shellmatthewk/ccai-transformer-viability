"""Deterministic temporal, geographic, and sparse-network experiment datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr


SPLITS = {
    "train": ("2010-01-01", "2017-12-31"),
    "validation": ("2018-01-01", "2018-12-31"),
    "calibration": ("2019-01-01", "2019-12-31"),
    "test": ("2020-01-01", "2021-12-31"),
}


def region_mask(longitude: np.ndarray, region: str, cutoff: float = 15.0) -> np.ndarray:
    if region == "global":
        return np.ones(longitude.shape, dtype=bool)
    if region == "source":
        return longitude <= cutoff
    if region == "target":
        return longitude > cutoff
    raise ValueError(f"Unknown region {region!r}")


@dataclass(frozen=True)
class Normalization:
    value_mean: float
    value_std: float
    coord_min: np.ndarray
    coord_max: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "value_mean": self.value_mean,
            "value_std": self.value_std,
            "coord_min": self.coord_min.tolist(),
            "coord_max": self.coord_max.tolist(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "Normalization":
        return cls(float(values["value_mean"]), float(values["value_std"]), np.asarray(values["coord_min"], dtype="float32"), np.asarray(values["coord_max"], dtype="float32"))


def read_stations(path: Path, variable: str, splits: tuple[str, ...]) -> pd.DataFrame:
    start = min(SPLITS[name][0] for name in splits)
    end = max(SPLITS[name][1] for name in splits)
    frame = pd.read_parquet(path, columns=["station_id", "date", "latitude", "longitude", variable]).dropna()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.loc[frame["date"].between(start, end)].reset_index(drop=True)


def fit_normalization(frame: pd.DataFrame, era5_path: Path, variable: str, train_region: str, cutoff: float) -> Normalization:
    start, end = SPLITS["train"]
    selected = frame.loc[frame.date.between(start, end)]
    selected = selected.loc[region_mask(selected.longitude.to_numpy(), train_region, cutoff)]
    if selected.empty:
        raise ValueError("Training normalization selection is empty.")
    with xr.open_dataset(era5_path) as ds:
        coord_min = np.array([float(ds.longitude.min()), float(ds.latitude.min())], dtype="float32")
        coord_max = np.array([float(ds.longitude.max()), float(ds.latitude.max())], dtype="float32")
    return Normalization(float(selected[variable].mean()), float(selected[variable].std()), coord_min, coord_max)


class ReconstructionDataset(Dataset):
    """Same-day ECA&D contexts and ERA5 targets for one split and region pair."""

    def __init__(self, frame: pd.DataFrame, era5_path: Path, station_variable: str, era5_variable: str, era5_offset: float, split: str, context_region: str, target_region: str, normalization: Normalization, max_context: int = 256, targets_per_day: int = 1024, retention: float = 1.0, seed: int = 0, training: bool = False, cutoff: float = 15.0) -> None:
        start, end = SPLITS[split]
        selected = frame.loc[frame.date.between(start, end)].copy()
        selected = selected.loc[region_mask(selected.longitude.to_numpy(), context_region, cutoff)]
        columns = ["station_id", "longitude", "latitude", station_variable]
        self.groups = {date: group[columns].to_numpy() for date, group in selected.groupby("date", sort=True)}
        self.ds = xr.open_dataset(era5_path)
        self.field = self.ds[era5_variable]
        era_dates = pd.DatetimeIndex(self.field.time.values).normalize()
        self.time_lookup = {date: index for index, date in enumerate(era_dates)}
        self.dates = sorted(date for date in self.groups if date in self.time_lookup)
        lon, lat = np.meshgrid(self.field.longitude.values, self.field.latitude.values)
        grid = np.stack((lon.ravel(), lat.ravel()), axis=-1).astype("float32")
        self.target_indices = np.flatnonzero(region_mask(grid[:, 0], target_region, cutoff))
        self.grid_coords = grid
        self.station_variable = station_variable
        self.offset = era5_offset
        self.norm = normalization
        self.max_context = max_context
        self.targets_per_day = targets_per_day
        self.retention = retention
        self.seed = seed
        self.training = training
        self.epoch = 0
        if not self.dates or not len(self.target_indices):
            raise ValueError(f"Empty dataset for split={split}, context={context_region}, target={target_region}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _coords(self, values: np.ndarray) -> np.ndarray:
        return (2 * (values - self.norm.coord_min) / (self.norm.coord_max - self.norm.coord_min) - 1).astype("float32")

    def _rng(self, index: int) -> np.random.Generator:
        epoch = self.epoch if self.training else 0
        return np.random.default_rng(self.seed + 1_000_003 * epoch + index)

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        date = self.dates[index]
        rows = self.groups[date]
        context_count = max(1, round(self.max_context * self.retention))
        context_count = min(context_count, len(rows))
        if self.training:
            context_choice = self._rng(index).choice(len(rows), context_count, replace=False)
        else:
            station_ids = rows[:, 0].astype("uint64")
            ranks = (station_ids * np.uint64(2654435761) + np.uint64(self.seed)) & np.uint64(0xFFFFFFFF)
            context_choice = np.argsort(ranks, kind="stable")[:context_count]
        target_count = min(self.targets_per_day, len(self.target_indices))
        target_choice = self._rng(index).choice(self.target_indices, target_count, replace=False)
        context = rows[context_choice]
        reference = self.field.isel(time=self.time_lookup[date]).values.reshape(-1).astype("float32") + self.offset
        return {
            "context_x": torch.from_numpy(self._coords(context[:, 1:3].astype("float32"))),
            "context_y": torch.from_numpy(((context[:, 3].astype("float32") - self.norm.value_mean) / self.norm.value_std)),
            "target_x": torch.from_numpy(self._coords(self.grid_coords[target_choice])),
            "target_y": torch.from_numpy((reference[target_choice] - self.norm.value_mean) / self.norm.value_std),
            "date_ns": torch.tensor(date.value, dtype=torch.int64),
        }

    def daily_target_means(self, points_per_day: int = 256) -> np.ndarray:
        means = []
        for index, date in enumerate(self.dates):
            rng = np.random.default_rng(self.seed + index)
            choice = rng.choice(self.target_indices, min(points_per_day, len(self.target_indices)), replace=False)
            values = self.field.isel(time=self.time_lookup[date]).values.reshape(-1)[choice].astype("float32") + self.offset
            means.append(float(np.nanmean(values)))
        return np.asarray(means, dtype="float32")
