import pytest
import torch

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
def source_prompt() -> str:
    return "A pack of roses"


@pytest.fixture
def destination_prompt() -> str:
    return "A pack of blue roses"


@pytest.fixture
def prompt() -> str:
    return "A field of roses"


@pytest.fixture
def seed() -> int:
    return 19950815


def test_pipeline_stable_diffusion_model_editing(
    model_id: str,
    torch_dtype: torch.dtype,
    device: torch.device,
    source_prompt: str,
    destination_prompt: str,
    prompt: str,
    seed: int,
) -> None:
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