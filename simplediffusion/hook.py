import torch

from modules.processing import StableDiffusionProcessing


class Hook:

    def apply_on_before_component_callback(self):
        pass

    def before_process(self, p: StableDiffusionProcessing):
        pass

    def process(self, p: StableDiffusionProcessing):
        pass

    def before_process_batch(self, *args, **kwargs):
        pass

    def after_extra_networks_activate(self, *args, **kwargs):
        pass

    def process_batch(self, *args, **kwargs):
        pass

    def process_before_every_sampling(self, process: StableDiffusionProcessing, x: torch.Tensor, noise: torch.Tensor, conditioning: torch.Tensor, unconditional_conditioning: torch.Tensor):
        pass

    def postprocess(self, *args, **kwargs):
        pass

    def postprocess_batch(self, *args, **kwargs):
        pass

    def postprocess_batch_list(self, *args, **kwargs):
        pass

    def post_sample(self, *args, **kwargs):
        pass

    def on_mask_blend(self, *args, **kwargs):
        pass

    def postprocess_image(self, *args, **kwargs):
        pass

    def postprocess_maskoverlay(self, *args, **kwargs):
        pass

    def postprocess_image_after_composite(self, *args, **kwargs):
        pass

    def before_component(self, *args, **kwargs):
        pass

    def after_component(self, *args, **kwargs):
        pass


# class HookWrapper(Hook):
#     def __init__(self, hooks: list[Hook] | None = None):
#         if hooks is None:
#             hooks = []
#         self.hooks = hooks

#     def apply_on_before_component_callback(self):
#         for hook in self.hooks:
#             hook.apply_on_before_component_callback()

#     def before_process(self):
#         for hook in self.hooks:
#             hook.before_process()

#     def process(self):
#         for hook in self.hooks:
#             hook.process()

#     def before_process_batch(self, *args, **kwargs):
#         for hook in self.hooks:
#             hook.before_process_batch(*args, **kwargs)

#     def after_extra_networks_activate(self, *args, **kwargs):
#         for hook in self.hooks:
#             hook.after_extra_networks_activate(*args, **kwargs)

#     def process_batch(self, *args, **kwargs):
#         for hook in self.hooks:
#             hook.process_batch(*args, **kwargs)

#     def process_before_every_sampling(self, process, x: torch.Tensor, noise: torch.Tensor, conditioning: torch.Tensor, unconditional_conditioning: torch.Tensor):
#         for hook in self.hooks:
#             hook.process_before_every_sampling(process, x, noise, conditioning, unconditional_conditioning)

#     def postprocess(self, *args, **kwargs):
#         for hook in self.hooks:
#             hook.postprocess(*args, **kwargs)

#     def postprocess_batch(self, *args, **kwargs):
#         for hook in self.hooks:
#             hook.postprocess_batch(*args, **kwargs)

#     def postprocess_batch_list(self, *args, **kwargs):
#         for hook in self.hooks:
#             hook.postprocess_batch_list(*args, **kwargs)

#     def post_sample(self, *args, **kwargs):
#         for hook in self.hooks:
#             hook.post_sample(*args, **kwargs)

#     def on_mask_blend(self, *args, **kwargs):
#         for hook in self.hooks:
#             hook.on_mask_blend(*args, **kwargs)

#     def postprocess_image(self, *args, **kwargs):
#         for hook in self.hooks:
#             hook.postprocess_image(*args, **kwargs)

#     def postprocess_maskoverlay(self, *args, **kwargs):
#         for hook in self.hooks:
#             hook.postprocess_maskoverlay(*args, **kwargs)
