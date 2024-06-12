import logger

logger.initialize()

import logging
import warnings
from pathlib import Path

import numpy as np
import PIL.Image

import modules.sd_models
from extensions.sd_forge_controlnet.lib_controlnet.external_code import ResizeMode
from extensions.sd_forge_controlnet.lib_controlnet.utils import crop_and_resize_image
from modules.processing import (
    StableDiffusionProcessingImg2Img,
    StableDiffusionProcessingTxt2Img,
    process_images_inner,
)
from modules_forge.forge_util import numpy_to_pytorch
from patchers.reference import PreprocessorReference
from simplediffusion.utils import Timer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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


def main():
    with Timer("Total"):
        img = PIL.Image.open("outputs/result.png")
        refenence_patcher = PreprocessorReference()
        patchers = [refenence_patcher]

        with Timer("Initialize"):
            checkpoint_path = Path("C:/Code/webui_forge_cu121_torch21/webui/models/Stable-diffusion/AOM3B2_orangemixs.safetensors")
            checkpoint_info = modules.sd_models.CheckpointInfo(checkpoint_path)
            sd_model = modules.sd_models.load_model(checkpoint_info)
        with Timer("Process"):
            p = StableDiffusionProcessingImg2Img(
                sd_model=sd_model,
                prompt="A picture of a cat",
                seed=47,
                outpath_samples="./outputs",
                init_images=[img],
                patchers=patchers,
            )

            h, w, hr_y, hr_x, has_high_res_fix = calculate_image_dimensions(p)
            logging.info(f"Image dimensions: {h}x{w}, HR: {hr_y}x{hr_x}, high_res_fix: {has_high_res_fix}")
            ref_image = img
            resize_mode = ResizeMode.RESIZE
            # TODO: preprocess
            ref_image_np = np.array(ref_image)
            control_cond = crop_and_resize_image(ref_image_np, resize_mode, h, w)
            control_cond = numpy_to_pytorch(control_cond).movedim(-1, 1)
            refenence_patcher.cond = control_cond

            res = process_images_inner(p)
            res.images[0].save("outputs/result3.png")

        p = StableDiffusionProcessingTxt2Img()
        p.sd_model = sd_model
        p.steps = 20
        p.seed = 47
        p.prompt = "A picture of a ((cat)), (((one girl)))"
        p.outpath_samples = "./outputs"
        with Timer("Process"):
            res = process_images_inner(p)
        res.images[0].save("outputs/result.png")
        p.prompt = "a cat in dark room"
        with Timer("Process"):
            res = process_images_inner(p)
        res.images[0].save("outputs/result2.png")

    logging.info("Done.")


if __name__ == "__main__":
    main()
