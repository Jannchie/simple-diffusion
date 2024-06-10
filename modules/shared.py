import os
import sys

from modules import (
    devices,
    interrogate,
    memmon,
    options,
    sd_models_types,
    shared_cmd_options,
    shared_gradio_themes,
    shared_items,
    shared_state,
    shared_total_tqdm,
    styles,
    util,
)
from modules.paths_internal import (  # noqa: F401; default_sd_model_file,; extensions_builtin_dir,; extensions_dir,; models_path,; script_path,; sd_configs_path,; sd_model_file,
    data_path,
    sd_default_config,
)

# import gradio as gr


cmd_opts = shared_cmd_options.cmd_opts
eight_load_location = None if cmd_opts.lowram else "cpu"


state = shared_state.State()

styles_filename = cmd_opts.styles_file = cmd_opts.styles_file if len(cmd_opts.styles_file) > 0 else [os.path.join(data_path, "styles.csv")]
prompt_styles = styles.StyleDatabase(styles_filename)


interrogator = interrogate.InterrogateModels("interrogate")


total_tqdm = shared_total_tqdm.TotalTQDM()


parser = shared_cmd_options.parser

batch_cond_uncond = True  # old field, unused now in favor of shared.opts.batch_cond_uncond
parallel_processing_allowed = True
config_filename = cmd_opts.ui_settings_file
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}

demo = None

device = None

weight_load_location = None

xformers_available = False

hypernetworks = {}

loaded_hypernetworks = []

interrogator = None

face_restorers = []

options_templates = None

from dataclasses import dataclass


@dataclass()
class Opts:
    hide_samplers = []
    disable_mmap_load_safetensors = False
    sd_checkpoint_cache = 0
    hide_ldm_prints = True
    data = {}
    sd_vae_overrides_per_model_preferences = True
    sd_vae = "Automatic"
    sd_vae_checkpoint_cache = 0
    textual_inversion_print_at_load = False
    s_min_uncond = 0.0
    s_churn = 0.0
    s_tmin = 0.0
    s_tmax = 0.0
    s_noise = 1.0
    use_old_emphasis_implementation = False
    emphasis = "Original"
    CLIP_stop_at_last_layers = 1
    textual_inversion_add_hashes_to_infotext = True
    face_restoration = False
    tiling = False
    disable_all_extensions: str = "none"
    restore_config_state_file = ""
    live_previews_enable = False
    show_progress_type = "Approx NN"
    sd_unet = "Automatic"
    randn_source = "GPU"
    token_merging_ratio = 0.0
    token_merging_ratio_hr = 0.0
    eta_noise_seed_delta = 0
    add_model_hash_to_info = True
    add_model_name_to_info = True
    fp8_storage: str = "Disable"
    cache_fp16_weight: bool = False
    auto_backcompat: bool = True
    add_vae_hash_to_info = True
    add_vae_name_to_info = True
    face_restoration_model = None
    add_version_to_infotext: bool = True
    add_user_name_to_info = True
    use_old_scheduling = False
    sdxl_crop_left = 0
    sdxl_crop_top = 0
    enable_quantization = False
    always_discard_next_to_last_sigma: bool = False
    k_sched_type = "Automatic"
    use_old_karras_scheduler_sigmas = False
    img2img_extra_noise = 0.0
    sgm_noise_multiplier = False
    show_progress_every_n_steps = 1
    multiple_tqdm = True
    disable_console_progressbars = False
    sd_vae_decode_method = "Full"
    samples_save = True
    save_incomplete_images = False
    overlay_inpaint = True
    samples_format = "png"
    grid_save_to_dirs = True
    save_to_dirs = True
    directories_filename_pattern = "[date]"
    samples_filename_pattern = None
    save_images_add_number = True
    enable_pnginfo = True
    jpeg_quality = 80
    save_images_replace_action = "Replace"
    target_side_length = 4000
    export_for_4chan = True
    img_downscale_threshold = 4.0
    save_txt = True
    grid_only_if_multiple = True
    return_grid = True
    grid_save = True
    comma_padding_backtrack = 20


opts = Opts()

mem_mon = memmon.MemUsageMonitor("MemMon", devices.device, opts)
mem_mon.start()

restricted_opts = None

sd_model: sd_models_types.WebuiSdModel = None

settings_components = None
"""assinged from ui.py, a mapping on setting names to gradio components repsponsible for those settings"""

tab_names = []

latent_upscale_default_mode = "Latent"
latent_upscale_modes = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}

sd_upscalers = []

clip_model = None

progress_print_out = sys.stdout

mem_mon = None

options_section = options.options_section
OptionInfo = options.OptionInfo
OptionHTML = options.OptionHTML

natural_sort_key = util.natural_sort_key
listfiles = util.listfiles
html_path = util.html_path
html = util.html
walk_files = util.walk_files
ldm_print = util.ldm_print

reload_gradio_theme = shared_gradio_themes.reload_gradio_theme

list_checkpoint_tiles = shared_items.list_checkpoint_tiles
refresh_checkpoints = shared_items.refresh_checkpoints
list_samplers = shared_items.list_samplers
reload_hypernetworks = shared_items.reload_hypernetworks
