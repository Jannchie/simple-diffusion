import logger

logger.initialize()

import logging
import warnings
from pathlib import Path

import modules.sd_models
from modules.processing import StableDiffusionProcessingTxt2Img, process_images_inner
from modules.shared_init import initialize
from simplediffusion.utils import Timer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    with Timer("Total"):
        with Timer("Initialize"):
            initialize()
            checkpoint_path = Path("C:/Code/webui_forge_cu121_torch21/webui/models/Stable-diffusion/AOM3B2_orangemixs.safetensors")
            checkpoint_info = modules.sd_models.CheckpointInfo(checkpoint_path)
            sd_model = modules.sd_models.load_model(checkpoint_info)
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
