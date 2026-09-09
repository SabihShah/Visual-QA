# Visual QA & Image Captioning

A computer vision project combining image captioning and visual question answering (VQA) into a single pipeline, with an interactive Gradio demo.

## Overview

Two components:

1. **Pretrained inference (active)** — Qwen2.5-VL-7B-Instruct handles both captioning and VQA out of the box via prompting, loaded in 4-bit for local GPU use.
2. **Custom training pipeline (available, not currently trained)** — a ClipCap-style architecture (frozen CLIP ViT-B/32 encoder → learned projection → GPT-2 decoder) for training a lightweight captioning/VQA model from scratch on Flickr8k / VQAv2.

## Architecture

**Pretrained path (in use):**
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Quantization: 4-bit (bitsandbytes) for ~8GB VRAM budgets
- Single model, single interface (`caption()` / `answer()`) for both tasks via prompt templates

**Custom path (built, optional):**
- Vision encoder: CLIP ViT-B/32 (frozen)
- Bridge: MLP projector mapping CLIP embedding → 10 GPT-2 prefix tokens
- Decoder: GPT-2, fine-tuned on prefix + caption/answer tokens

## Project Structure

```
visual-qa-captioning/
├── configs/
│   └── base.yaml              # model + training config
├── src/
│   ├── models/
│   │   ├── model.py           # custom ClipCap-style model (training)
│   │   └── pretrained.py      # Qwen2.5-VL wrapper (inference only)
│   ├── data/
│   │   ├── dataset.py         # CaptioningDataset / VQADataset
│   │   └── prepare_flickr8k.py
│   ├── train/
│   │   └── train.py           # training loop for custom model
│   └── eval/
│       └── evaluate.py        # BLEU / CIDEr scoring
├── demo/
│   └── app.py                 # Gradio demo (uses pretrained.py)
├── checkpoints/                # saved training checkpoints (if trained)
├── data/                       # datasets / manifests
└── requirements.txt
```

## Setup

```bash
conda create -n vqa-caption python=3.10 -y
conda activate vqa-caption
pip install -r requirements.txt
```

**requirements.txt** includes: `torch`, `transformers`, `accelerate`, `bitsandbytes`, `qwen-vl-utils`, `gradio`, `pillow`, plus `pycocoevalcap` and `open_clip_torch` for the optional training path.

## Usage

### Run the demo (pretrained, no training required)
```bash
python -m demo.app
```
Opens a local Gradio UI to upload an image, generate a caption, and ask free-form questions about it.

### Optional: train the custom model
```bash
# 1. Prepare data
python -m src.data.prepare_flickr8k

# 2. Train
python -m src.train.train --config configs/base.yaml

# 3. Evaluate
python -m src.eval.evaluate
```
Set `train.task` in `configs/base.yaml` to `captioning` or `vqa`.

## Training / Fine-tuning on Your Own Data

The custom pipeline (`src/models/model.py`) can be trained on any image-caption or image-question-answer dataset, not just Flickr8k/VQAv2.

### 1. Format your data

Create a JSON file — a list of dicts. No fixed schema requirement beyond these keys:

**Captioning:**
```json
[
  {"image_path": "data/my_images/img001.jpg", "caption": "A red bicycle parked outside a shop."},
  {"image_path": "data/my_images/img002.jpg", "caption": "Two people hiking on a mountain trail."}
]
```

**VQA:**
```json
[
  {"image_path": "data/my_images/img001.jpg", "question": "What color is the bicycle?", "answer": "red"},
  {"image_path": "data/my_images/img002.jpg", "question": "How many people are in the image?", "answer": "two"}
]
```

Save as:
- `data/captioning_train.json` / `data/captioning_val.json` (for captioning)
- `data/vqa_train.json` / `data/vqa_val.json` (for VQA)

If your data comes from a spreadsheet or another format (COCO-style, CSV, etc.), write a small converter script following the pattern in `src/data/prepare_flickr8k.py` — read your raw format, output the JSON structure above.

### 2. Update the config

Edit `configs/base.yaml`:
```yaml
train:
  task: captioning        # or vqa
  batch_size: 16           # lower if you hit VRAM limits
  lr: 5e-5
  epochs: 5
  freeze_vision: true      # keep true unless you have a large dataset
```

For small custom datasets (a few hundred to a few thousand samples), keep `freeze_vision: true` — only the projector and GPT-2 get updated, which avoids overfitting and fits comfortably in 8GB VRAM.

### 3. Train

```bash
python -m src.train.train --config configs/base.yaml
```

Checkpoints save each epoch to `checkpoints/{task}_epoch{N}.pt`.

### 4. Evaluate

Point `evaluate.py` at your own checkpoint and val file:
```bash
python -m src.eval.evaluate
```
(edit the `checkpoint` and `val_json` paths at the bottom of `src/eval/evaluate.py` if not using the defaults)

### 5. Use your fine-tuned model in the demo

Swap `demo/app.py` to load your custom checkpoint via `VisualCaptioner` (from `src/models/model.py`) instead of `PretrainedVQACaptioner`, following the same pattern used in `src/eval/evaluate.py` for loading weights.

### Tips for fine-tuning on custom data

- **Small datasets (<1k samples):** freeze CLIP, keep LR low (5e-5), fewer epochs (3–5) to avoid overfitting.
- **Larger datasets (10k+):** consider unfreezing the last few CLIP layers for a bigger accuracy gain.
- **Mixed captioning + VQA:** train separate checkpoints per task first; combining into one multitask run is a valid next step but needs balanced sampling between the two datasets.
- **VRAM limits:** reduce `batch_size` before reducing image resolution; the CLIP encoder is frozen so gradient memory is dominated by GPT-2 + projector.

## Tech Stack

- Qwen2.5-VL-7B-Instruct (pretrained VLM)
- PyTorch, Transformers, bitsandbytes (4-bit quantization)
- CLIP, GPT-2 (custom training path)
- Gradio (demo UI)
- pycocoevalcap (BLEU, CIDEr)

## Tags

`computer-vision` `vision-language-model` `image-captioning` `visual-question-answering` `qwen2.5-vl` `gradio` `pytorch` `transformers`
