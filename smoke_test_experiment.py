"""Small real-data integration check for split, regional, and sparse datasets."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiment_data import ReconstructionDataset, fit_normalization, read_stations
from models import ConvCNP, StationTransformer
from models.common import normal_nll


STATIONS = Path("data/tg/daily_temperature_2010_2021.parquet")
ERA5 = Path("data/era5_daily_t2m_2010_2021.nc")


def main() -> None:
    frame = read_stations(STATIONS, "tg", ("train", "validation", "calibration", "test"))
    for experiment, region in (("global", "global"), ("source", "source")):
        norm = fit_normalization(frame, ERA5, "tg", region, 15.0)
        data = ReconstructionDataset(frame, ERA5, "tg", "t2m", -273.15, "train", region, region, norm, max_context=16, targets_per_day=32, seed=7, training=True)
        batch = next(iter(DataLoader(data, batch_size=2)))
        for model in (ConvCNP(grid_size=16, channels=8), StationTransformer(d_model=32, nhead=4, layers=1)):
            mean, log_std = model(batch["context_x"], batch["context_y"], batch["target_x"])
            loss = normal_nll(mean, log_std, batch["target_y"]); loss.backward()
            print(experiment, model.__class__.__name__, tuple(mean.shape), f"nll={loss.item():.4f}")
    sparse = ReconstructionDataset(frame, ERA5, "tg", "t2m", -273.15, "test", "target", "target", norm, max_context=256, targets_per_day=32, retention=0.10, seed=7, training=False)
    assert sparse[0]["context_x"].shape[0] == 26
    print("target sparse context count", sparse[0]["context_x"].shape[0])


if __name__ == "__main__":
    torch.manual_seed(7)
    main()
