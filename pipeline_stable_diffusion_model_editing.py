from typing import List

import torch
from diffusers import StableDiffusionModelEditingPipeline as SDTIME
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.deprecated.stable_diffusion_variants.pipeline_stable_diffusion_model_editing import (
    AUGS_CONST,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer


class StableDiffusionModelEditingPipeline(SDTIME):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        with_to_k: bool = True,
        with_augs: List[str] = AUGS_CONST,
    ) -> None:
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
            with_to_k,
            with_augs,
        )

        # get cross-attention layers
        ca_layers = []

        def append_ca(net_):
            # In diffusers v1.15.0 and later, `CrossAttention` has been changed to `Attention`
            # Refer to the pipeline in the fork:
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deprecated/stable_diffusion_variants/pipeline_stable_diffusion_model_editing.py#L135
            if net_.__class__.__name__ == "Attention":
                ca_layers.append(net_)
            elif hasattr(net_, "children"):
                for net__ in net_.children():
                    append_ca(net__)

        # recursively find all cross-attention layers in unet
        for net in self.unet.named_children():
            if "down" in net[0]:
                append_ca(net[1])
            elif "up" in net[0]:
                append_ca(net[1])
            elif "mid" in net[0]:
                append_ca(net[1])

        # get projection matrices
        self.ca_clip_layers = [l for l in ca_layers if l.to_v.in_features == 768]
        assert len(self.ca_clip_layers) > 0
        self.projection_matrices = [l.to_v for l in self.ca_clip_layers]
        assert len(self.projection_matrices) > 0

        if self.with_to_k:
            projection_matrices = [l.to_k for l in self.ca_clip_layers]
            self.projection_matrices = self.projection_matrices + projection_matrices
            assert len(self.projection_matrices) > 0

    @torch.no_grad()
    def edit_model(
        self,
        source_prompt: str,
        destination_prompt: str,
        lamb: float = 0.1,
        **kwargs,
    ) -> None:
        # `restart_params` creates a copy of the object when restoring the original weights, which can lead to problems such as the device not being set correctly when exiting the pipeline. For these reasons, `restart_params` is set to `False`. If you want to restore the original weights, it is recommended to reload the pipeline.
        super().edit_model(
            source_prompt, destination_prompt, lamb, restart_params=False
        )
