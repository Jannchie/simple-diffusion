import logger

logger.initialize()
import logging
import warnings
from pathlib import Path

import numpy as np
import PIL.Image

import modules.sd_models
from simplediffusion.patchers.controlnet import ControlNetHook
from simplediffusion.patchers.controlnet.lib_controlnet.external_code import (
    ControlNetUnit,
    ResizeMode,
)
from simplediffusion.patchers.controlnet.lib_controlnet.utils import (
    crop_and_resize_image,
)
from simplediffusion.patchers.controlnet.supported_controlnet import ControlNetPatcher
from simplediffusion.patchers.reference_only import ReferenceOnlyHook
from simplediffusion.preprocessors.legacy_preprocessors.legacy_preprocessors import (
    LegacyPreprocessor,
)
from simplediffusion.preprocessors.legacy_preprocessors.preprocessor_compiled import (
    legacy_preprocessors,
)
from simplediffusion.processing import (
    StableDiffusionProcessingImg2Img,
    StableDiffusionProcessingTxt2Img,
    process_images_inner,
)

openpose_preprocessor = LegacyPreprocessor(legacy_preprocessors["openpose"])

from simplediffusion.utils import (
    Timer,
    calculate_image_dimensions,
    load_torch_file,
    numpy_to_pytorch,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    with Timer("Total"):
        img = PIL.Image.open("outputs/result.png")
        img2 = PIL.Image.open("outputs/result2.png")
        pose_img = PIL.Image.open(R"C:\Users\Jannchie\OneDrive\图片\1.jpg")
        refenence_only_hook = ReferenceOnlyHook(use_attn=True)
        openpose_controlnet_path = Path(R"E:\stable-diffusion-webui-forge\models\ControlNet\OpenPoseXL2.safetensors")
        resize_mode = ResizeMode.RESIZE

        image_np = np.array(pose_img)

        state_dict = load_torch_file(openpose_controlnet_path, safe_load=True)
        state_dict_copy = dict(state_dict.items())
        openpose_model = ControlNetPatcher.try_build_from_state_dict(state_dict_copy, openpose_controlnet_path)
        controlnet_hook = ControlNetHook(
            [
                ControlNetUnit(
                    module="openpose",
                    preprocessor=openpose_preprocessor,
                    model=openpose_model,
                    image={
                        "image": image_np,
                        "mask": np.zeros_like(image_np),
                    },
                    weight=1.6,
                    
                )
            ]
        )
        hooks = [controlnet_hook]
        with Timer("Initialize"):
            checkpoint_path = Path(R"E:\webui_forge_cu121_torch21\webui\models\Stable-diffusion\animagine-xl-3.1.safetensors")
            checkpoint_info = modules.sd_models.CheckpointInfo(checkpoint_path)
            sd_model = modules.sd_models.load_model(checkpoint_info)
        with Timer("Process"):
            p = StableDiffusionProcessingTxt2Img(
                sd_model=sd_model,
                prompt="1girl, masterpiece, best quality, light brown background, from side, portrait, green eyes, tsurime, long hair, medium breasts, knees, headdress, string bikini, closed eyes",
                negative_prompt="nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
                seed=47,
                outpath_samples="./outputs",
                # sampler_name="DPM++ 2M SDE",
                # init_images=[img],
                hooks=hooks,
                width=832,
                height=1216,
                cfg_scale=5,
                steps=20,
            )
            h, w, hr_y, hr_x, has_high_res_fix = calculate_image_dimensions(p)
            logging.info(f"Image dimensions: {h}x{w}, HR: {hr_y}x{hr_x}, high_res_fix: {has_high_res_fix}")

            ref_image = img2

            ref_image_np = np.array(ref_image)
            ref_cond = crop_and_resize_image(ref_image_np, resize_mode, h, w)
            ref_cond = numpy_to_pytorch(ref_cond).movedim(-1, 1)

            refenence_only_hook.cond = ref_cond

            res = process_images_inner(p)
            res.images[0].save("outputs/result3.png")

        # p = StableDiffusionProcessingTxt2Img()
        # p.sd_model = sd_model
        # p.steps = 20
        # p.seed = 47
        # p.prompt = "1girl, masterpiece, best quality, light brown background, from side, portrait, green eyes, tsurime, long hair, medium breasts, knees, headdress, string bikini, closed eyes"
        # p.negative_prompt = "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
        # p.outpath_samples = "./outputs"
        # p.width = 832
        # p.cfg_scale = 5.0
        # p.height = 1216
        # p.scripts = script_runner
        # with Timer("Process"):
        #     res = process_images_inner(p)
        # res.images[0].save("outputs/result.png")
        # p.prompt = "a cat in dark room"
        # with Timer("Process"):
        #     res = process_images_inner(p)
        # res.images[0].save("outputs/result2.png")

    logging.info("Done.")


if __name__ == "__main__":
    main()
