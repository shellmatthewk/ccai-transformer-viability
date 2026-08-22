# CUDA selection

Use `--device auto`, `--device cuda`, or `--device cpu` with
`train_models_device.py`.

`auto` is the default. It uses CUDA only when the active PyTorch installation
reports `torch.cuda.is_available() == True`; otherwise it runs on CPU. `cuda`
fails early if CUDA is unavailable, preventing an accidental multi-hour CPU run.
The chosen device and PyTorch version are printed at startup and saved in the
checkpoint metadata.

The currently active PyTorch build is CPU-only (`2.13.0+cpu`). Although this
machine has NVIDIA GPUs, a CUDA-enabled PyTorch build must be installed before
`--device cuda` can work.
