import logging
import time
from pathlib import Path

import numpy as np
import safetensors
import torch

from modules.processing import StableDiffusionProcessingTxt2Img

from . import checkpoint_pickle


@torch.no_grad()
@torch.inference_mode()
def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y


@torch.no_grad()
@torch.inference_mode()
def pytorch_to_numpy(x):
    return [np.clip(255.0 * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]


def align_dim_latent(x: int) -> int:
    """Align the pixel dimension (w/h) to latent dimension.
    Stable diffusion 1:8 ratio for latent/pixel, i.e.,
    1 latent unit == 8 pixel unit."""
    return (x // 8) * 8


def calculate_image_dimensions(p):
    """Returns (h, w, hr_h, hr_w, has_high_res_fix)."""
    h = align_dim_latent(p.height)
    w = align_dim_latent(p.width)
    has_high_res_fix = isinstance(p, StableDiffusionProcessingTxt2Img) and getattr(p, "enable_hr", False)
    if has_high_res_fix:
        if p.hr_resize_x == 0 and p.hr_resize_y == 0:
            hr_y = int(p.height * p.hr_scale)
            hr_x = int(p.width * p.hr_scale)
        else:
            hr_y, hr_x = p.hr_resize_y, p.hr_resize_x
        hr_y = align_dim_latent(hr_y)
        hr_x = align_dim_latent(hr_x)
    else:
        hr_y = h
        hr_x = w
    return h, w, hr_y, hr_x, has_high_res_fix


def load_torch_file(ckpt: Path | str, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if str(ckpt).lower().endswith(".safetensors"):
        return safetensors.torch.load_file(ckpt, device=device.type)
    if safe_load and "weights_only" not in torch.load.__code__.co_varnames:
        print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
        safe_load = False
    pl_sd = (
        torch.load(ckpt, map_location=device, weights_only=True)
        if safe_load
        else torch.load(
            ckpt,
            map_location=device,
            pickle_module=checkpoint_pickle,
        )
    )
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    return pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd


class Timer:
    def __init__(self, name="Time taken"):
        self.start = time.time()
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type or exc_value or traceback:
            logging.info(f"{self.name}: Exception occurred.")

        self.end = time.time()
        self.interval = self.end - self.start
        logging.info(f"{self.name}: {self.interval:.2f} seconds.")
