import warnings
from pathlib import Path

import rich

import modules.sd_models
from ldm_patched.modules.model_management import load_models_gpu
from modules.processing import StableDiffusionProcessingTxt2Img, process_images_inner
from modules.shared_init import initialize
from simplediffusion.utils import Timer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

console = rich.get_console()


def main():
    with Timer("Total"):
        with Timer("Initialize"):
            initialize()
            checkpoint_path = Path("C:/Code/webui_forge_cu121_torch21/webui/models/Stable-diffusion/AOM3B2_orangemixs.safetensors")
            checkpoint_info = modules.sd_models.CheckpointInfo(checkpoint_path)
            modules.sd_models.load_model(checkpoint_info)
        p = StableDiffusionProcessingTxt2Img()
        p.steps = 20
        p.seed = 47
        p.prompt = "A picture of a ((cat)), (((one girl)))"
        p.outpath_samples = "./outputs"
        with Timer("Process"):
            process_images_inner(p)
        pass


if __name__ == "__main__":
    main()
    pass
