"""Real-data forward/backward and memory check with every daily station."""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from full_context_data import FullContextDataset, fit_full_normalization, padded_collate, read_domain_stations
from models.common import normal_nll
from models.convcnp import ConvCNP
from models.latent_grid_transformer import LatentGridTransformer


def run(model: torch.nn.Module, batch: dict[str, torch.Tensor], device_index: int) -> None:
    device = torch.device(f"cuda:{device_index}"); torch.cuda.reset_peak_memory_stats(device); model.to(device)
    started = time.perf_counter()
    mean, log_std = model(batch["context_x"].to(device), batch["context_y"].to(device), batch["target_x"].to(device), batch["context_mask"].to(device))
    loss = normal_nll(mean, log_std, batch["target_y"].to(device)); loss.backward(); torch.cuda.synchronize(device)
    print(model.__class__.__name__, torch.cuda.get_device_name(device_index), f"stations={batch['context_count'].item()}", f"targets={mean.shape[1]}", f"seconds={time.perf_counter()-started:.3f}", f"peak_gb={torch.cuda.max_memory_allocated(device)/1e9:.3f}", f"nll={loss.item():.4f}")


def main() -> None:
    stations = Path("data/tg/daily_temperature_2010_2021.parquet"); era5 = Path("data/era5_daily_t2m_2010_2021.nc")
    frame = read_domain_stations(stations, "tg", ("train",), era5); norm = fit_full_normalization(frame, era5, "tg", "global", 15.0)
    data = FullContextDataset(frame, era5, "tg", "t2m", -273.15, "train", "global", "global", norm, targets_per_day=1024, seed=2026, training=True)
    batch = next(iter(DataLoader(data, batch_size=1, collate_fn=padded_collate)))
    run(LatentGridTransformer(), batch, 0); run(ConvCNP(), batch, 1)


if __name__ == "__main__": main()
