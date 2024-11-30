# Fork of [Text-to-image model editing (TIME)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/model_editing) 🤗

This is [a forked version of the diffusers 🤗 implementation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/model_editing) of [Text-to-image model editing (TIME, Editing Implicit Assumptions in Text-to-Image Diffusion Models)](https://arxiv.org/abs/2303.08084).

The code in this repository is managed at [py-img-gen/diffusers-text-to-model-editing](https://github.com/py-img-gen/diffusers-text-to-model-editing).

Here are the minor changes:
- Changes due to the renaming of the `CrossAttention` class to the `Attention` class
- Removed the `restart_params` with large side effects from the `edit_model` function
