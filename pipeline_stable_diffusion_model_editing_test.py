import os

import pytest
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import HfApi

from pipeline_stable_diffusion_model_editing import StableDiffusionModelEditingPipeline


@pytest.fixture
def model_id() -> str:
    return "stable-diffusion-v1-5/stable-diffusion-v1-5"


@pytest.fixture
def torch_dtype() -> torch.dtype:
    return torch.float32


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def root_dir() -> str:
    return os.path.dirname(__file__)


@pytest.fixture
def pipeline_script_path(root_dir: str) -> str:
    return os.path.join(root_dir, "pipeline_stable_diffusion_model_editing.py")


class TestPipelineStableDiffusionModelEditing:
    @pytest.fixture
    def source_prompt(self) -> str:
        return "A pack of roses"

    @pytest.fixture
    def destination_prompt(self) -> str:
        return "A pack of blue roses"

    @pytest.fixture
    def prompt(self) -> str:
        return "A field of roses"

    @pytest.fixture
    def seed(self) -> int:
        return 19950815

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="No GPUs available for testing.",
    )
    def test_pipeline(
        self,
        model_id: str,
        torch_dtype: torch.dtype,
        device: torch.device,
        source_prompt: str,
        destination_prompt: str,
        prompt: str,
        seed: int,
    ):
        pipe = StableDiffusionModelEditingPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
        pipe = pipe.to(device)

        pipe.edit_model(
            source_prompt=source_prompt,
            destination_prompt=destination_prompt,
        )

        output = pipe(prompt=prompt, generator=torch.manual_seed(seed))
        image = output.images[0]
        image.save("output.png")

    def test_custom_pipeline(
        self,
        model_id: str,
        torch_dtype: torch.dtype,
        pipeline_script_path: str,
    ):
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            custom_pipeline=pipeline_script_path,
            torch_dtype=torch_dtype,
        )
        assert pipe.__class__.__name__ == "StableDiffusionModelEditingPipeline"


class TestPushToHub:
    @pytest.fixture
    def hf_org_name(self) -> str:
        return "py-img-gen"

    @pytest.fixture
    def hf_pipeline_name(self) -> str:
        return "stable-diffusion-text-to-model-editing"

    @pytest.fixture
    def hf_repo_id(self, hf_org_name: str, hf_pipeline_name: str) -> str:
        return f"{hf_org_name}/{hf_pipeline_name}"

    @pytest.fixture
    def readme_path(self, root_dir: str) -> str:
        return os.path.join(root_dir, "README.md")

    def test_push_to_hub(
        self,
        hf_repo_id: str,
        pipeline_script_path: str,
        readme_path: str,
        repo_type: str = "model",
    ) -> None:
        api = HfApi()

        api.upload_file(
            path_or_fileobj=pipeline_script_path,
            path_in_repo="pipeline.py",
            repo_id=hf_repo_id,
            repo_type=repo_type,
        )
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=hf_repo_id,
            repo_type=repo_type,
        )
