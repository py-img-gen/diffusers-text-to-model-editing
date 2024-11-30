# Fork of [Text-to-image model editing (TIME)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/model_editing) 🤗

This is [a forked version of the diffusers 🤗 implementation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/model_editing) of [Text-to-image model editing (TIME, Editing Implicit Assumptions in Text-to-Image Diffusion Models)](https://arxiv.org/abs/2303.08084).

The code in this repository is managed at [py-img-gen/diffusers-text-to-model-editing](https://github.com/py-img-gen/diffusers-text-to-model-editing).

Here are the minor changes:
- Changes due to the renaming of the `CrossAttention` class to the `Attention` class
- Removed the `restart_params` with large side effects from the `edit_model` function

## How to Run

```python
import torch
from diffusers import DiffusionPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5"
    custom_pipeline="py-img-gen/stable-diffusion-text-to-model-editing",
)
pipe = pipe.to(device)

prompt = "A field of roses"
source_prompt = "A pack of roses"
destination_prompt = "A pack of blue roses"

pipe.edit_model(
    source_prompt=source_prompt,
    destination_prompt=destination_prompt,
)

output = pipe(prompt=prompt)
image_edited = output.images[0]
image_edited
```

You can find the notebook example in [notebooks/run_pipeline.ipynb](examples/).

## Comparison of the generated results

- Prompt: `A field of roses`
- Source Prompt: `A pack of roses`
- Destination Prompt: `A pack of blue roses`

| Original | TIME |
| --- | --- |
| ![image](https://github.com/user-attachments/assets/3707841a-de34-4ba8-8ca1-3278a76b161b) | ![image](https://github.com/user-attachments/assets/5e672a3a-8c72-4f76-bba2-5f6147dcf546) |


## Acknowledgements

The code in this repository is based on the [Text-to-image model editing (TIME)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/model_editing) implementation by [🤗 diffusers](https://github.com/huggingface/diffusers).
