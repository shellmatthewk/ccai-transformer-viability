"""Train a ConvCNP or station transformer against same-day ERA5 reference fields.

Example (ERA5 t2m stored in kelvin):
  python train_models.py --model convcnp --era5 data/era5_t2m.nc --era5-variable t2m --era5-offset -273.15

The script deliberately refuses to substitute station readings as targets: the
paper's reconstruction evaluation requires the separate ERA5 reference field.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from models import ConvCNP, StationTransformer
from models.common import normal_nll


class DailyERA5Pairs(Dataset):
    def __init__(self, station_path: Path, era5_path: Path, station_variable: str, era5_variable: str, era5_offset: float, max_context: int, targets_per_day: int, seed: int) -> None:
        frame = pd.read_parquet(station_path, columns=["date", "latitude", "longitude", station_variable]).dropna()
        frame["date"] = pd.to_datetime(frame.date).dt.normalize()
        self.groups = {d: g[["longitude", "latitude", station_variable]].to_numpy("float32") for d, g in frame.groupby("date")}
        self.ds = xr.open_dataset(era5_path)
        if era5_variable not in self.ds:
            raise KeyError(f"ERA5 variable {era5_variable!r} not found; available: {list(self.ds.data_vars)}")
        self.field = self.ds[era5_variable]
        for coordinate in ("time", "latitude", "longitude"):
            if coordinate not in self.field.coords:
                raise KeyError(f"Expected ERA5 coordinate {coordinate!r}; rename it before using this script.")
        self.dates = [d for d in self.groups if np.datetime64(d) in self.field.time.values]
        if not self.dates:
            raise ValueError("Station and ERA5 files have no exact common dates.")
        lon, lat = np.meshgrid(self.field.longitude.values, self.field.latitude.values)
        self.grid_coords = np.stack((lon.ravel(), lat.ravel()), axis=-1).astype("float32")
        all_coords = np.concatenate((frame[["longitude", "latitude"]].to_numpy("float32"), self.grid_coords))
        self.coord_min, self.coord_max = all_coords.min(0), all_coords.max(0)
        self.mean, self.std = float(frame[station_variable].mean()), float(frame[station_variable].std())
        self.max_context, self.targets_per_day, self.offset = max_context, targets_per_day, era5_offset
        self.rng = np.random.default_rng(seed)

    def _coords(self, array: np.ndarray) -> np.ndarray:
        return 2 * (array - self.coord_min) / (self.coord_max - self.coord_min) - 1

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rows = self.groups[self.dates[index]]
        context_idx = self.rng.choice(len(rows), min(len(rows), self.max_context), replace=False)
        target_idx = self.rng.choice(len(self.grid_coords), min(len(self.grid_coords), self.targets_per_day), replace=False)
        reference = self.field.sel(time=np.datetime64(self.dates[index])).values.reshape(-1).astype("float32") + self.offset
        context = rows[context_idx]
        return {
            "context_x": torch.from_numpy(self._coords(context[:, :2])),
            "context_y": torch.from_numpy((context[:, 2] - self.mean) / self.std),
            "target_x": torch.from_numpy(self._coords(self.grid_coords[target_idx])),
            "target_y": torch.from_numpy((reference[target_idx] - self.mean) / self.std),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("convcnp", "transformer"), required=True)
    parser.add_argument("--stations", type=Path, default=Path("processed/daily_temperature_2010_2021.parquet"))
    parser.add_argument("--era5", type=Path, required=True)
    parser.add_argument("--station-variable", default="tg")
    parser.add_argument("--era5-variable", default="t2m")
    parser.add_argument("--era5-offset", type=float, default=0.0, help="Use -273.15 for ERA5 kelvin temperature.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--targets-per-day", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    dataset = DailyERA5Pairs(args.stations, args.era5, args.station_variable, args.era5_variable, args.era5_offset, args.max_context, args.targets_per_day, seed=0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = ConvCNP() if args.model == "convcnp" else StationTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            mean, log_std = model(batch["context_x"], batch["context_y"], batch["target_x"])
            loss = normal_nll(mean, log_std, batch["target_y"])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += loss.item()
        print(f"epoch={epoch:03d} nll={total / len(loader):.5f}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_type": args.model, "value_mean": dataset.mean, "value_std": dataset.std, "coord_min": dataset.coord_min, "coord_max": dataset.coord_max, "args": vars(args)}, args.output / f"{args.model}.pt")


if __name__ == "__main__":
    main()
