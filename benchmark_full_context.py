"""Warm-step benchmark for full-station training runtime estimates."""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from full_context_data import FullContextDataset, fit_full_normalization, padded_collate, read_domain_stations
from models.common import normal_nll
from models.convcnp import ConvCNP
from models.latent_grid_transformer import LatentGridTransformer


def benchmark(model, batch, gpu):
    device = torch.device(f"cuda:{gpu}"); model.to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    tensors = {key: value.to(device) for key, value in batch.items() if key in ("context_x", "context_y", "context_mask", "target_x", "target_y")}
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True); mean, log_std = model(tensors["context_x"], tensors["context_y"], tensors["target_x"], tensors["context_mask"]); normal_nll(mean, log_std, tensors["target_y"]).backward(); optimizer.step()
    torch.cuda.synchronize(device); started = time.perf_counter()
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True); mean, log_std = model(tensors["context_x"], tensors["context_y"], tensors["target_x"], tensors["context_mask"]); normal_nll(mean, log_std, tensors["target_y"]).backward(); optimizer.step()
    torch.cuda.synchronize(device); seconds = (time.perf_counter() - started) / 10
    print(model.__class__.__name__, torch.cuda.get_device_name(gpu), f"seconds_per_day={seconds:.4f}", f"estimated_train_epoch_minutes={seconds * 2922 / 60:.2f}")


def main():
    station_path=Path("data/tg/daily_temperature_2010_2021.parquet"); era5_path=Path("data/era5_daily_t2m_2010_2021.nc")
    frame=read_domain_stations(station_path,"tg",("train",),era5_path); norm=fit_full_normalization(frame,era5_path,"tg","global",15)
    data=FullContextDataset(frame,era5_path,"tg","t2m",-273.15,"train","global","global",norm,targets_per_day=1024,training=True)
    batch=next(iter(DataLoader(data,batch_size=1,collate_fn=padded_collate)))
    benchmark(LatentGridTransformer(),batch,0); benchmark(ConvCNP(),batch,1)


if __name__ == "__main__": main()
