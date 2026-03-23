<p align="center">
  <h1 align="center">DIFFUSION-LM WATERMARKING</h1>
</p>

<p align="center">
  <em>Watermarking diffusion-based language models with discrete masking diffusion (D3PM)</em>
</p>

<p align="center">
  <img src="https://img.shields.io/github/last-commit/idhantsingh027/diffusion-lm-watermarking?style=flat&color=007ec6" alt="Last Commit" />
  <img src="https://img.shields.io/github/languages/top/idhantsingh027/diffusion-lm-watermarking?style=flat&color=007ec6&cache=none" alt="Top Language" />
  <img src="https://img.shields.io/github/languages/count/idhantsingh027/diffusion-lm-watermarking?style=flat&color=5c5c5c" alt="Languages Count" />
</p>

<p align="center">
  <em>Built with the tools and technologies:</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Transformers-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white" alt="Transformers" />
</p>

<br>

## 🚀 Features

- **D3PM-style forward process** using token masking
- **Timestep-conditioned denoiser** for $p_\theta(x_0 \mid x_t, t)$
- **D3PM reverse process** with monotonic unmasking (no re-masking)
- **Stochastic sampling** for watermark-friendly generation
- **Mask inspection** to visualize forward corruption on real data
- **Comprehensive evaluation**: reconstruction accuracy + GPT-2 perplexity metrics

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/idhantsingh027/diffusion-lm-watermarking.git
cd diffusion-lm-watermarking
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Google Colab (GPU)

If you're running in Colab and want GPU training, install the CUDA-enabled PyTorch build:
```bash
pip install -r requirements-colab.txt
```

Sanity check (should print `True` and a CUDA version):
```bash
python -c "import torch; print('cuda_available=', torch.cuda.is_available(), 'torch_cuda=', torch.version.cuda)"
```

## 💻 Usage

### 1) Quick smoke-test training
This is a **local sanity check** (CPU, a couple of batches). Full training is intended to run in **Google Colab**.

```bash
python models/diffusion_lm.py train \
  --epochs 1 \
  --batch_size 2 \
  --max_length 32 \
  --steps 5 \
  --max_train_batches 2 \
  --device cpu
```

### 1b) Training with hard masking (loss around ~7)

**Direct hard-masking training** (`--min_mask_prob 0.15 --max_mask_prob 0.90`, `--epochs 3`, `--max_train_batches 200`) often stayed around **~7 loss**.

```bash
python models/diffusion_lm.py train \
  --device cpu \
  --epochs 3 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --warmup_steps 200 \
  --max_length 64 \
  --steps 25 \
  --min_mask_prob 0.15 \
  --max_mask_prob 0.90 \
  --max_train_batches 200 \
  --log_every 10
```

### 1c) Curriculum training (recommended to reach lower loss)

A practical way to drive the loss down is to **start with BERT-like masking** (`min=max=0.15`), then resume and ramp the masking up.

Observed (in-notebook run, CPU):
- **Stage 1 easy masking** (`--min_mask_prob 0.15 --max_mask_prob 0.15`, `--epochs 3`) reached about **~2.73 loss** (e.g. `loss≈2.7368` at step ~1410).

Stage 1 (easy / BERT-like masking - 15%):
```bash
python models/diffusion_lm.py train \
  --device cpu \
  --output_dir checkpoints/bert-mlm-curriculum-easy \
  --epochs 3 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --warmup_steps 200 \
  --max_length 64 \
  --steps 25 \
  --min_mask_prob 0.15 \
  --max_mask_prob 0.15 \
  --max_train_batches 0 \
  --log_every 10
```

Your exact curve depends on hardware and batch settings.

Stage 2 (resume + harder diffusion masking - 90%):
```bash
python models/diffusion_lm.py train \
  --device cpu \
  --model_name checkpoints/bert-mlm-curriculum-easy/epoch-3 \
  --output_dir checkpoints/bert-mlm-diffusion-baseline \
  --epochs 5 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --warmup_steps 200 \
  --max_length 64 \
  --steps 25 \
  --min_mask_prob 0.15 \
  --max_mask_prob 0.90 \
  --max_train_batches 0 \
  --log_every 10
```

### 2) Sample text (D3PM reverse)
**One-line description:** runs the D3PM reverse process to generate a 48‑token sample from a *diffusion‑trained checkpoint* using temperature sampling and top‑k filtering.
```bash
python models/diffusion_lm.py sample \
  --ckpt checkpoints/bert-mlm-diffusion-baseline/epoch-1 \
  --length 48 \
  --steps 25 \
  --temperature 1.0 \
  --top_k 50
```
> Note: `bert-base-uncased` is a *baseline* and not trained with diffusion noise. It will run, but generation is not a correct diffusion sample.

### 3) Inspect forward masking
```bash
python models/diffusion_lm.py inspect-mask \
  --split train \
  --idx 0 \
  --max_length 24 \
  --steps 5 \
  --min_mask_prob 0.2 \
  --max_mask_prob 0.8 \
  --t 4
```

### 4) training_v3.ipynb - 2-Stage Curriculum
Implements initial curriculum learning with MDLM improvements:
- **Stage 1**: 15% masking, 5 epochs — learns basic token reconstruction
- **Stage 2**: 50% masking, 5 epochs — learns to denoise higher noise levels

**Results achieved:**
- Reconstruction accuracy: **69.7%** (target: >30%)
- GPT-2 perplexity (generated samples fluency): **151.9** (target: <500)
- Final best: **65.8% accuracy, 187.7 perplexity**

Features:
- Low-discrepancy timestep sampling (MDLM)
- Zero masking probabilities (prevents predicting [MASK] tokens)
- Gradient accumulation (effective batch size 128)
- Per-epoch checkpoints for watermarking experiments
- GPT-2 perplexity evaluation for generation quality

### 5) training_v4.ipynb - 3-Stage Continuation (Future Work 🔥)
Continues from v3 best checkpoint with advanced techniques:
- **Stage 3**: 30% masking, 4 epochs — fills the gap progressively
- **Stage 4**: 70% masking, 5 epochs — pushes to high noise levels
- **Stage 5**: 70% masking + embeddings unfrozen, 3 epochs — fine-tunes everything

## 📁 Project Structure

```
diffusion-lm-watermarking/
├── data/
│   └── dataset.py
├── docs/
│   └── problem-statement.md
├── models/
│   └── diffusion_lm.py
├── requirements.txt
└── README.md
```

## 🧪 Notes

- This repo currently implements a **D3PM-style masking diffusion baseline**.
- Model weights under `checkpoints/` are large and stored via **Git LFS**. If you clone and see pointer files, install Git LFS (`git lfs install`) and run `git lfs pull`.
- Watermarking integration (logit bias + detection) is intended as the next step.

## 🤝 Contributing

Contributions are welcome. Please open an issue or pull request with clear details.

## 👨‍💻 Author

**Idhant Singh**
- GitHub: [@idhantsingh027](https://github.com/idhantsingh027)

## 🌟 Acknowledgments

- Diffusion LM literature (D3PM, Masked diffusion)
- **MDLM**: Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (NeurIPS 2024)
- **dLLM**: Zhou, "Discrete Diffusion Language Models" (2025)
- Hugging Face Transformers & Datasets
