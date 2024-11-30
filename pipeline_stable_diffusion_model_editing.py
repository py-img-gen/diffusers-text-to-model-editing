from typing import List

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
    ):
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
