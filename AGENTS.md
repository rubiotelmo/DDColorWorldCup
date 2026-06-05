# AGENTS.md

## Scope
- These instructions apply to the whole repository.
- If a nested `AGENTS.md` is added later, the nested file takes precedence for files under that directory.
- User instructions in the active conversation take precedence over this file.

## Project Overview
- DDColor is a PyTorch image colorization project for the ICCV 2023 paper "DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders".
- The repo contains two closely related code paths:
  - `ddcolor/`: lightweight inference-facing package with `DDColor`, `ColorizationPipeline`, and checkpoint-loading helpers.
  - `basicsr/`: forked/adapted BasicSR training framework, registries, losses, metrics, datasets, and the training-time DDColor architecture.
- Utility entry points live in `scripts/`; demo/deployment entry points live in `demo/` and `cog.yaml`.
- Pretrained weights, generated images, datasets, experiment logs, and exported models are local artifacts and should not be committed.

## Repo Map
- `ddcolor/model.py`: inference model definition. It supports `convnext-t` and `convnext-l` encoders and the `MultiScaleColorDecoder` path.
- `ddcolor/pipeline.py`: shared BGR OpenCV image pipeline. `ColorizationPipeline.process` accepts a BGR `uint8` image and returns a BGR `uint8` image.
- `basicsr/archs/ddcolor_arch.py`: training architecture registered in `ARCH_REGISTRY`; includes both `MultiScaleColorDecoder` and `SingleColorDecoder`.
- `basicsr/models/color_model.py`: GAN/perceptual/colorfulness training loop, validation, FID/colorfulness metrics, and checkpoint saving.
- `basicsr/data/lab_dataset.py`: Lab colorization dataset. It reads meta-info file lists and returns `lq` L-channel tensors plus `gt` AB-channel tensors.
- `options/train/train_ddcolor.yml`: default distributed training config.
- `scripts/infer.py`: local-weight or Hugging Face inference CLI.
- `scripts/export_onnx.py`: ONNX export and verification CLI.
- `scripts/get_meta_file.py`: dataset list generator.
- `demo/gradio_app.py`: local Gradio app; expects a ModelScope checkpoint path by default.
- `demo/cog_predict.py` and `cog.yaml`: Replicate Cog predictor and build settings.
- `.devcontainer/`: VS Code devcontainer setup when present. It creates a Python 3.9 conda env named `ddcolor`.

## Environment Setup
- Baseline from `README.md`: Python >= 3.7 and PyTorch >= 1.7.
- Recommended local/devcontainer setup uses Python 3.9 with:
  - `pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118`
  - `pip install -r requirements.txt`
  - `python setup.py develop`
- `requirements.txt` currently includes both inference and training dependencies.
- `requirements.train.txt` is referenced by docs/devcontainer logic but is not present in this checkout. Do not assume it exists; skip it unless a user adds it.
- Cog uses its own environment in `cog.yaml` with Python 3.11 and PyTorch 2.0.1.
- Do not set `BASICSR_EXT=True` unless the BasicSR CUDA extension source tree is present; the default setup path builds no custom extensions.

## Common Commands
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  python setup.py develop
  ```
- Run inference with local weights:
  ```bash
  python scripts/infer.py --model_path ./modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt --input ./assets/test_images --output ./results
  ```
- Run inference with Hugging Face weights:
  ```bash
  python scripts/infer.py --model_name ddcolor_modelscope --input ./assets/test_images --output ./results
  ```
- Run the shell inference wrapper:
  ```bash
  sh scripts/inference.sh
  ```
- Generate a dataset meta file:
  ```bash
  python scripts/get_meta_file.py --data-path /path/to/images --output-name data_list/dataset.txt
  ```
- Start distributed training:
  ```bash
  sh scripts/train.sh
  ```
- Export ONNX after installing ONNX extras:
  ```bash
  pip install onnx==1.16.1 onnxruntime==1.19.2 onnxsim==0.4.36
  python scripts/export_onnx.py --model_path pretrain/ddcolor_paper_tiny.pth --export_path weights/ddcolor-tiny.onnx
  ```
- Start the Gradio demo after installing UI extras:
  ```bash
  pip install gradio gradio_imageslider
  python demo/gradio_app.py
  ```

## Validation
- There is no formal test suite or CI configuration in this repository.
- For syntax-only validation after Python edits, run:
  ```bash
  python -m compileall -q ddcolor basicsr scripts demo
  ```
- For import-level validation when dependencies are installed, run:
  ```bash
  python - <<'PY'
  from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model
  print("ddcolor import ok")
  PY
  ```
- For inference changes, prefer a small smoke run on `assets/test_images` with `ddcolor_paper_tiny` or an available local checkpoint. Note that Hugging Face/ModelScope modes may download model weights.
- For training changes, at minimum validate option parsing and imports if full multi-GPU training is impractical. Full training depends on ImageNet-style data lists, ConvNeXt/VGG/Inception weights under `pretrain/`, and GPU availability.
- For ONNX changes, run the export command only when ONNX dependencies and a suitable checkpoint are available.
- If a validation command cannot be run because weights, GPUs, network access, or optional dependencies are unavailable, state that clearly in the final response.

## Code Style
- Follow `setup.cfg`:
  - flake8 max line length is 120.
  - yapf style is based on pep8 with column limit 120.
  - isort line length is 120, with `basicsr` treated as first-party.
- Keep edits consistent with the surrounding code. This codebase mixes upstream BasicSR style and newer helper modules; do not reformat unrelated files.
- Prefer single-purpose helper functions over broad refactors.
- Keep public APIs in `ddcolor/__init__.py`, `scripts/infer.py`, `demo/gradio_app.py`, and `demo/cog_predict.py` stable unless the task explicitly changes them.
- Preserve OpenCV BGR conventions at file boundaries. Convert to RGB/Lab only where the existing pipeline does so.
- Preserve tensor contracts:
  - inference model input is RGB-like 3-channel float tensor in `[0, 1]`;
  - colorization model output is AB channels when `num_output_channels=2`;
  - `ColorizationPipeline.process` handles resizing and recombines predicted AB with the original L channel.

## Architecture Guidelines
- Use the registry pattern for new BasicSR components:
  - datasets: `DATASET_REGISTRY`
  - architectures: `ARCH_REGISTRY`
  - models: `MODEL_REGISTRY`
  - losses: `LOSS_REGISTRY`
  - metrics: `METRIC_REGISTRY`
- Add training-time architectures under `basicsr/archs/*_arch.py` so lazy registry import in `basicsr/archs/__init__.py` can discover them.
- Add inference-facing, dependency-light code under `ddcolor/` when the change is meant for scripts, Gradio, Cog, or Hugging Face usage.
- Keep `ddcolor/model.py` and `basicsr/archs/ddcolor_arch.py` aligned when changing shared model behavior, but do not blindly copy code between them. The training version has registry and pretrained-encoder behavior that the inference package intentionally avoids.
- Be careful with model-size behavior:
  - `tiny` maps to `convnext-t`;
  - `large` maps to `convnext-l`;
  - local-weight inference defaults to `large` unless `--model_size tiny` is passed.

## Data, Weights, and Generated Files
- Do not commit datasets, downloaded checkpoints, generated colorizations, experiment logs, TensorBoard logs, WandB runs, ONNX exports, or Cog/Gradio temporary outputs.
- Expected local artifact locations include `pretrain/`, `modelscope/`, `checkpoints/`, `weights/`, `results/`, `experiments/`, `tb_logger/`, and `wandb/`.
- `setup.py develop` generates `basicsr/version.py`; this is ignored and should not be hand-edited.
- Training FID uses `pretrain/inception_v3_google-1a9a5a14.pth`.
- Perceptual loss may require VGG weights under `pretrain/` depending on the selected `vgg_type`.
- Training with `encoder_from_pretrain: True` expects ConvNeXt weights in `pretrain/`.

## Security and Operations
- Do not add secrets, tokens, private dataset paths, or machine-specific absolute paths to tracked files.
- Treat external model weights as untrusted binary inputs. Load only from expected sources or user-provided paths.
- Avoid network downloads in validation unless the task requires them or the user has already chosen a download-based workflow.
- Keep GPU-dependent commands optional and document when a task could not be fully validated without CUDA.

## Change Discipline
- This repository may contain user edits. Check `git status --short` before changing files and avoid overwriting unrelated work.
- Keep changes focused on the requested behavior.
- Do not modernize deprecated training commands, dependency pins, or upstream BasicSR internals unless that is the task.
- Update `README.md` or `MODEL_ZOO.md` when user-facing commands, model names, or artifact locations change.
- Add or update targeted validation notes in the final response because automated test coverage is currently absent.
