import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizerFast, get_linear_schedule_with_warmup

# Allow running as: `python models/diffusion_lm.py ...`
# so imports like `data.dataset` work without installing as a package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from data.dataset import WikiTextDataset


@dataclass
class TrainConfig:
	model_name: str = "bert-base-uncased"
	split: str = "train"
	max_length: int = 128
	batch_size: int = 32
	lr: float = 2e-5
	weight_decay: float = 0.01
	epochs: int = 5
	max_train_batches: int = 0
	log_every: int = 50
	warmup_steps: int = 500
	steps: int = 100
	min_mask_prob: float = 0.15
	max_mask_prob: float = 0.70
	grad_accum_steps: int = 4
	dataset_version: str = "wikitext-2-raw-v1"
	lr_schedule: str = "cosine"
	unfreeze_embeddings: bool = False
	output_dir: str = "checkpoints/bert-mlm-diffusion-v5"
	seed: int = 1234
	device: str = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
	use_timestep_conditioning: bool = True


def resolve_device(requested: str) -> str:
	"""Return a usable torch device string.

	If the user asks for CUDA but this environment doesn't have a CUDA-enabled
	PyTorch build (or no GPU runtime), we fall back to CPU with a clear warning
	instead of crashing during `model.to('cuda')`.
	"""
	device = (requested or "cpu").strip()
	if device.startswith("cuda"):
		try:
			if not torch.cuda.is_available():
				print(
					"WARNING: --device cuda requested but CUDA is not available in this runtime. "
					"Falling back to CPU. If you're on Colab, enable GPU (Runtime → Change runtime type) "
					"and install CUDA PyTorch (pip install -r requirements-colab.txt)."
				)
				return "cpu"
			# Sanity-check that the requested CUDA device is actually usable.
			torch.empty(1).to(device)
		except Exception as e:
			print(
				f"WARNING: Requested device '{device}' is not usable ({e!s}). "
				"Falling back to CPU."
			)
			return "cpu"
	if device == "mps":
		mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
		if not mps_ok:
			print("WARNING: --device mps requested but MPS is not available. Falling back to CPU.")
			return "cpu"
	return device


def set_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def get_scheduler(
	optimizer,
	schedule: str,
	warmup_steps: int,
	total_steps: int,
):
	"""
	Build LR scheduler.
	'linear': linear warmup then linear decay (original)
	'cosine': linear warmup then cosine decay (MDLM)
	Cosine prevents LR dropping too fast mid-training.
	"""
	if schedule == "cosine":
		from torch.optim.lr_scheduler import (
			LambdaLR, CosineAnnealingLR, SequentialLR
		)
		warmup = LambdaLR(
			optimizer,
			lr_lambda=lambda step: min(
				1.0, step / max(warmup_steps, 1)
			)
		)
		cosine = CosineAnnealingLR(
			optimizer,
			T_max=max(total_steps - warmup_steps, 1),
			eta_min=1e-7,
		)
		return SequentialLR(
			optimizer,
			schedulers=[warmup, cosine],
			milestones=[warmup_steps],
		)
	else:
		return get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=warmup_steps,
			num_training_steps=total_steps,
		)


def mask_prob_for_t(t: torch.Tensor, *, steps: int, min_p: float, max_p: float) -> torch.Tensor:
	"""Linear schedule from min_p..max_p over timesteps 1..steps."""
	if steps <= 1:
		return torch.full_like(t, fill_value=max_p, dtype=torch.float32)
	frac = (t.float() - 1.0) / float(steps - 1)
	return min_p + frac * (max_p - min_p)


def sample_timesteps_low_discrepancy(
	bsz: int, steps: int, device: str
) -> torch.Tensor:
	"""
	Low-discrepancy sampler for diffusion timesteps.
	Ensures all noise levels are evenly covered each batch.
	From MDLM (Sahoo et al. 2024) — more stable than uniform.
	"""
	offset = torch.rand(1).item()
	t_float = [(offset + i / bsz) % 1.0 for i in range(bsz)]
	t_int = [max(1, min(steps, int(v * steps) + 1)) 
	         for v in t_float]
	return torch.tensor(t_int, device=device, dtype=torch.long)


def make_noised_batch(
	input_ids: torch.Tensor,
	*,
	tokenizer: BertTokenizerFast,
	steps: int,
	min_mask_prob: float,
	max_mask_prob: float,
	device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Create x_t by masking tokens with p(t); return (x_t, labels, t).

	We interpret p(t) as the *cumulative* mask rate at diffusion timestep t, i.e.
	q(x_t | x_0) masks each non-special token independently with probability p_t.

	Training objective: predict x0 tokens at masked positions, i.e. a denoiser for
	pθ(x0 | x_t, t) (or pθ(x_{t-1} | x_t, t) in a stricter formulation).
	"""
	bsz, seqlen = input_ids.shape
	t = sample_timesteps_low_discrepancy(bsz, steps, device)
	p = mask_prob_for_t(t, steps=steps, min_p=min_mask_prob, max_p=max_mask_prob)  # [B]

	special = torch.zeros_like(input_ids, dtype=torch.bool)
	for tok_id in (
		tokenizer.cls_token_id,
		tokenizer.sep_token_id,
		tokenizer.pad_token_id,
	):
		if tok_id is not None:
			special |= input_ids.eq(tok_id)

	# Sample mask decisions per token.
	# Each position i in batch b: mask with probability p[b].
	probs = p[:, None].expand(bsz, seqlen)
	mask = (torch.rand((bsz, seqlen), device=device) < probs) & (~special)

	x_t = input_ids.clone()
	if tokenizer.mask_token_id is None:
		raise ValueError("Tokenizer has no [MASK] token; BERT tokenizer is required.")
	x_t[mask] = tokenizer.mask_token_id

	labels = input_ids.clone()
	labels[~mask] = -100  # ignore non-masked positions
	labels[labels == tokenizer.mask_token_id] = -100
	return x_t, labels, t


def make_noised_example(
	input_ids: torch.Tensor,
	*,
	tokenizer: BertTokenizerFast,
	t_int: int,
	steps: int,
	min_mask_prob: float,
	max_mask_prob: float,
	device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Mask a single example at a fixed timestep t_int.

	Returns (x_t, mask_bool) where mask_bool marks which positions were masked.
	"""
	if input_ids.dim() != 1:
		raise ValueError("input_ids must be 1D (single example)")
	if tokenizer.mask_token_id is None:
		raise ValueError("Tokenizer has no [MASK] token; BERT tokenizer is required.")
	if t_int < 1 or t_int > steps:
		raise ValueError(f"t_int must be in [1, {steps}]")

	input_ids = input_ids.to(device)
	p = float(
		mask_prob_for_t(
			torch.tensor([t_int], device=device),
			steps=steps,
			min_p=min_mask_prob,
			max_p=max_mask_prob,
		)[0]
	)

	special = torch.zeros_like(input_ids, dtype=torch.bool)
	for tok_id in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id):
		if tok_id is not None:
			special |= input_ids.eq(tok_id)

	mask = (torch.rand_like(input_ids.float()) < p) & (~special)
	x_t = input_ids.clone()
	x_t[mask] = tokenizer.mask_token_id
	return x_t, mask


class TimestepConditionedBertForMaskedLM(nn.Module):
	"""A thin wrapper that conditions BERT-MLM on timestep t.

	We add a learned embedding e(t) to every token embedding before passing through
	BERT. This keeps the architecture close to BERT while giving the denoiser access
	to the corruption level, approximating a diffusion-style pθ(x0 | xt, t).
	"""

	TIME_EMBED_FILENAME = "time_embed.pt"

	def __init__(self, base: BertForMaskedLM, *, num_steps: int):
		super().__init__()
		self.base = base
		self.num_steps = int(num_steps)
		hidden = self.base.config.hidden_size
		self.time_embed = nn.Embedding(self.num_steps + 1, hidden)
		nn.init.normal_(self.time_embed.weight, mean=0.0, std=0.02)
		self.time_ln = nn.LayerNorm(hidden)
		self.base.cls.apply(self.base._init_weights)

	def forward(
		self,
		*,
		input_ids: torch.Tensor,
		t: torch.Tensor,
		attention_mask: torch.Tensor | None = None,
		labels: torch.Tensor | None = None,
	):
		if attention_mask is None:
			pad_id = self.base.config.pad_token_id
			if pad_id is None:
				attention_mask = torch.ones_like(input_ids, dtype=torch.long)
			else:
				attention_mask = input_ids.ne(pad_id).long()

		t = t.clamp_(min=0, max=self.num_steps)
		# Build token embeddings as BERT would, then add timestep embedding.
		emb = self.base.bert.embeddings(input_ids=input_ids)
		emb = emb + self.time_embed(t).unsqueeze(1)
		emb = self.time_ln(emb)

		outputs = self.base.bert(
			inputs_embeds=emb,
			attention_mask=attention_mask,
			return_dict=True,
		)
		sequence_output = outputs.last_hidden_state
		logits = self.base.cls(sequence_output)

		loss = None
		if labels is not None:
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

		return {"loss": loss, "logits": logits}

	def save_pretrained(self, save_directory: str) -> None:
		os.makedirs(save_directory, exist_ok=True)
		self.base.save_pretrained(save_directory)
		payload = {
			"num_steps": self.num_steps,
			"time_embed_state_dict": self.time_embed.state_dict(),
			"time_ln_state_dict": self.time_ln.state_dict(),
		}
		torch.save(payload, os.path.join(save_directory, self.TIME_EMBED_FILENAME))

	@classmethod
	def from_pretrained(cls, ckpt_or_name: str, *, num_steps: int, device: str) -> "TimestepConditionedBertForMaskedLM":
		device = resolve_device(device)
		base = BertForMaskedLM.from_pretrained(ckpt_or_name)
		obj = cls(base, num_steps=num_steps)
		path = os.path.join(ckpt_or_name, cls.TIME_EMBED_FILENAME)
		if os.path.exists(path):
			payload = torch.load(path, map_location="cpu")
			obj.num_steps = int(payload.get("num_steps", num_steps))
			obj.time_embed = nn.Embedding(obj.num_steps + 1, obj.base.config.hidden_size)
			obj.time_embed.load_state_dict(payload["time_embed_state_dict"])
			if "time_ln_state_dict" in payload:
				obj.time_ln.load_state_dict(payload["time_ln_state_dict"])
		obj.to(device)
		return obj


@torch.no_grad()
def generate_d3pm(
	model: BertForMaskedLM | TimestepConditionedBertForMaskedLM,
	tokenizer: BertTokenizerFast,
	*,
	length: int,
	steps: int,
	min_mask_prob: float,
	max_mask_prob: float,
	device: str,
	temperature: float = 1.0,
	top_k: int = 50,
) -> str:
	"""Generate text using a D3PM-style reverse process (monotonic unmasking).

	Forward: q(x_t | x_0) masks tokens independently with cumulative rate p_t.
	Reverse: starting from x_T (masked), at each step t -> t-1 we *only unmask*
	a subset of currently-masked tokens so that the expected mask rate decreases
	from p_t to p_{t-1}.
	"""
	if tokenizer.mask_token_id is None:
		raise ValueError("Tokenizer has no [MASK] token; BERT tokenizer is required.")
	if length < 4:
		raise ValueError("length must be >= 4 to fit [CLS] ... [SEP]")
	if steps < 1:
		raise ValueError("steps must be >= 1")

	# x_T: fully masked (except special tokens)
	input_ids = torch.full((1, length), fill_value=tokenizer.mask_token_id, device=device, dtype=torch.long)
	input_ids[0, 0] = tokenizer.cls_token_id
	input_ids[0, -1] = tokenizer.sep_token_id

	cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
	for t_int in range(steps, 0, -1):
		t_tensor = torch.tensor([t_int], device=device)
		p_t = float(mask_prob_for_t(t_tensor, steps=steps, min_p=min_mask_prob, max_p=max_mask_prob)[0])
		if t_int == 1:
			p_prev = 0.0
		else:
			p_prev = float(mask_prob_for_t(torch.tensor([t_int - 1], device=device), steps=steps, min_p=min_mask_prob, max_p=max_mask_prob)[0])

		p_t = max(p_t, 1e-6)
		p_prev = max(p_prev, 0.0)
		keep_mask_prob = min(max(p_prev / p_t, 0.0), 1.0)

		special = input_ids.eq(cls_id) | input_ids.eq(sep_id)
		if pad_id is not None:
			special |= input_ids.eq(pad_id)

		is_mask = input_ids.eq(tokenizer.mask_token_id) & (~special)
		if not bool(is_mask.any()):
			continue

		# Decide which masked positions stay masked in x_{t-1}
		stay_masked = (torch.rand_like(input_ids.float()) < keep_mask_prob) & is_mask
		to_unmask = is_mask & (~stay_masked)
		if not bool(to_unmask.any()):
			continue

		# Predict tokens for the positions that will be unmasked.
		if isinstance(model, TimestepConditionedBertForMaskedLM):
			out = model(input_ids=input_ids, t=t_tensor)
			logits = out["logits"]
		else:
			logits = model(input_ids=input_ids).logits

		logits = logits / max(temperature, 1e-6)
		if top_k > 0:
			topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
			filtered = torch.full_like(logits, fill_value=-float("inf"))
			filtered.scatter_(-1, topk_idx, topk_vals)
			logits = filtered

		probs = torch.softmax(logits, dim=-1)
		sampled = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(1, -1)
		input_ids[to_unmask] = sampled[to_unmask]
		# stay_masked remains [MASK]

	# Any remaining masks (should be rare if p_1 isn't too high) -> argmax fill
	remaining = input_ids.eq(tokenizer.mask_token_id) & (~(input_ids.eq(cls_id) | input_ids.eq(sep_id)))
	if pad_id is not None:
		remaining &= ~input_ids.eq(pad_id)
	if bool(remaining.any()):
		if isinstance(model, TimestepConditionedBertForMaskedLM):
			out = model(input_ids=input_ids, t=torch.tensor([1], device=device))
			logits = out["logits"]
		else:
			logits = model(input_ids=input_ids).logits
		logits = logits / max(temperature, 1e-6)
		if top_k > 0:
			topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
			filtered = torch.full_like(logits, fill_value=-float("inf"))
			filtered.scatter_(-1, topk_idx, topk_vals)
			logits = filtered

		probs = torch.softmax(logits, dim=-1)
		sampled = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(1, -1)
		input_ids[remaining] = sampled[remaining]

	text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
	return " ".join(text.split())


@torch.no_grad()
def reconstruction_accuracy(
	model: "TimestepConditionedBertForMaskedLM",
	tokenizer: BertTokenizerFast,
	sentences: list[str],
	*,
	mask_prob: float = 0.15,
	steps: int = 100,
	device: str,
) -> dict:
	"""
	Evaluate masked token reconstruction accuracy.
	
	For each sentence:
	  1. Tokenize it
	  2. Mask mask_prob fraction of tokens (forward process at t=1)
	  3. Run ONE denoising step at t=1 (lowest noise level)
	  4. Compare predictions at masked positions to ground truth
	
	Returns top-1 accuracy on masked positions.
	This directly measures if the model can fill in masked words.
	From dLLM evaluation methodology.
	"""
	model.eval()
	correct = 0
	total = 0

	for sentence in sentences:
		enc = tokenizer(
			sentence,
			return_tensors="pt",
			truncation=True,
			max_length=128,
			padding="max_length",
		)
		input_ids = enc["input_ids"].to(device)  # [1, L]

		# Forward process: mask at t=1 (lowest noise = 15%)
		x_t, mask = make_noised_example(
			input_ids[0],
			tokenizer=tokenizer,
			t_int=1,
			steps=steps,
			min_mask_prob=mask_prob,
			max_mask_prob=mask_prob,
			device=device,
		)
		if not mask.any():
			continue

		x_t_batch = x_t.unsqueeze(0)
		t_tensor = torch.tensor([1], device=device)

		out = model(input_ids=x_t_batch, t=t_tensor)
		logits = out["logits"]  # [1, L, vocab]

		# Get predictions at masked positions
		pred_ids = logits[0].argmax(dim=-1)  # [L]
		
		masked_positions = mask.nonzero(as_tuple=True)[0]
		for pos in masked_positions:
			pred = int(pred_ids[pos].item())
			gold = int(input_ids[0, pos].item())
			if pred == gold:
				correct += 1
			total += 1

	model.train()

	if total == 0:
		return {"accuracy": 0.0, "correct": 0, "total": 0}

	return {
		"accuracy": round(correct / total, 4),
		"correct": correct,
		"total": total,
	}


def train(cfg: TrainConfig) -> None:
	set_seed(cfg.seed)
	os.makedirs(cfg.output_dir, exist_ok=True)
	log_every = max(int(cfg.log_every), 0)

	tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
	dataset = WikiTextDataset(split=cfg.split, max_length=cfg.max_length, dataset_version=cfg.dataset_version)
	loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

	if cfg.use_timestep_conditioning:
		model = TimestepConditionedBertForMaskedLM.from_pretrained(cfg.model_name, num_steps=cfg.steps, device=cfg.device)
	else:
		model = BertForMaskedLM.from_pretrained(cfg.model_name).to(cfg.device)
	model.train()

	# Freeze word/position/token-type embeddings
	if not cfg.unfreeze_embeddings:
		for param in model.base.bert.embeddings.parameters():
			param.requires_grad = False
	else:
		# Unfreeze — used in later training stages
		for param in model.base.bert.embeddings.parameters():
			param.requires_grad = True
		print("Embeddings unfrozen for fine-tuning.")

	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
	total_steps = cfg.epochs * len(loader)
	scheduler = get_scheduler(
		optimizer,
		schedule=cfg.lr_schedule,
		warmup_steps=cfg.warmup_steps,
		total_steps=total_steps,
	)

	for epoch in range(cfg.epochs):
		running = 0.0
		seen = 0
		for i, input_ids in enumerate(loader, start=1):
			input_ids = input_ids.to(cfg.device)
			if tokenizer.pad_token_id is None:
				attention_mask = torch.ones_like(input_ids, dtype=torch.long)
			else:
				attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
			x_t, labels, _t = make_noised_batch(
				input_ids,
				tokenizer=tokenizer,
				steps=cfg.steps,
				min_mask_prob=cfg.min_mask_prob,
				max_mask_prob=cfg.max_mask_prob,
				device=cfg.device,
			)

			if cfg.use_timestep_conditioning:
				out = model(input_ids=x_t, t=_t, attention_mask=attention_mask, labels=labels)
				loss = out["loss"]
			else:
				out = model(input_ids=x_t, attention_mask=attention_mask, labels=labels)
				loss = out.loss
			
			loss = loss / cfg.grad_accum_steps
			loss.backward()

			if i % cfg.grad_accum_steps == 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad(set_to_none=True)

			running += float(loss.item()) * cfg.grad_accum_steps
			seen += 1
			# If this is a smoke-test run, stop after N batches *before* resetting any log window.
			if cfg.max_train_batches and i >= cfg.max_train_batches:
				avg = running / float(max(seen, 1))
				print(
					f"Stopping early: max_train_batches={cfg.max_train_batches} (avg_loss_over_last_{seen}_batches={avg:.4f})",
					flush=True,
				)
				break

			if log_every and i % log_every == 0:
				avg = running / float(seen)
				print(f"epoch={epoch+1} step={i}/{len(loader)} loss={avg:.4f}", flush=True)
				running = 0.0
				seen = 0

		# Save each epoch
		epoch_dir = os.path.join(cfg.output_dir, f"epoch-{epoch+1}")
		os.makedirs(epoch_dir, exist_ok=True)
		if cfg.use_timestep_conditioning:
			model.save_pretrained(epoch_dir)
		else:
			model.save_pretrained(epoch_dir)
		tokenizer.save_pretrained(epoch_dir)

	print(f"Saved checkpoints to: {cfg.output_dir}", flush=True)


def main() -> None:
	parser = argparse.ArgumentParser(description="Discrete masking-diffusion LM baseline (D3PM-style).")
	sub = parser.add_subparsers(dest="cmd", required=True)

	p_train = sub.add_parser("train", help="Train the denoiser (MLM) on masked diffusion corruption.")
	p_train.add_argument("--model_name", default=TrainConfig.model_name)
	p_train.add_argument("--split", default=TrainConfig.split)
	p_train.add_argument("--max_length", type=int, default=TrainConfig.max_length)
	p_train.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
	p_train.add_argument("--lr", type=float, default=TrainConfig.lr)
	p_train.add_argument("--epochs", type=int, default=TrainConfig.epochs)
	p_train.add_argument(
		"--max_train_batches",
		type=int,
		default=TrainConfig.max_train_batches,
		help="Smoke-test helper: stop after this many batches per epoch (0 = no limit).",
	)
	p_train.add_argument(
		"--log_every",
		type=int,
		default=TrainConfig.log_every,
		help="Print a running average loss every N batches (0 = disable).",
	)
	p_train.add_argument("--warmup_steps", type=int, default=TrainConfig.warmup_steps)
	p_train.add_argument("--steps", type=int, default=TrainConfig.steps)
	p_train.add_argument("--min_mask_prob", type=float, default=TrainConfig.min_mask_prob)
	p_train.add_argument("--max_mask_prob", type=float, default=TrainConfig.max_mask_prob)
	p_train.add_argument("--dataset_version", default=TrainConfig.dataset_version)
	p_train.add_argument("--output_dir", default=TrainConfig.output_dir)
	p_train.add_argument("--seed", type=int, default=TrainConfig.seed)
	p_train.add_argument("--device", default=TrainConfig.device)
	p_train.add_argument("--grad_accum_steps", type=int, default=TrainConfig.grad_accum_steps)
	p_train.add_argument(
		"--lr_schedule",
		default=TrainConfig.lr_schedule,
		choices=["linear", "cosine"],
		help="LR schedule: linear or cosine decay "
		     "after warmup. Cosine recommended (MDLM).",
	)
	p_train.add_argument(
		"--unfreeze_embeddings",
		action=argparse.BooleanOptionalAction,
		default=TrainConfig.unfreeze_embeddings,
		help="If set, unfreezes BERT embedding layer. "
		     "Use in later training stages after model "
		     "has converged.",
	)
	p_train.add_argument(
		"--use_timestep_conditioning",
		action=argparse.BooleanOptionalAction,
		default=TrainConfig.use_timestep_conditioning,
		help="If enabled, conditions the denoiser on diffusion timestep t.",
	)

	p_sample = sub.add_parser("sample", help="Generate text via iterative denoising.")
	p_sample.add_argument("--ckpt", default=TrainConfig.model_name, help="Checkpoint dir or HF model name")
	p_sample.add_argument("--length", type=int, default=48)
	p_sample.add_argument("--steps", type=int, default=25)
	p_sample.add_argument("--min_mask_prob", type=float, default=0.15)
	p_sample.add_argument("--max_mask_prob", type=float, default=0.95)
	p_sample.add_argument("--temperature", type=float, default=1.0)
	p_sample.add_argument("--top_k", type=int, default=50)
	p_sample.add_argument("--seed", type=int, default=1234)
	p_sample.add_argument("--device", default=TrainConfig.device)
	p_sample.add_argument(
		"--use_timestep_conditioning",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="If enabled, uses timestep-conditioned model if available.",
	)

	p_inspect = sub.add_parser("inspect-mask", help="Print an example before/after forward masking.")
	p_inspect.add_argument("--split", default="train", choices=["train", "validation", "test"])
	p_inspect.add_argument("--idx", type=int, default=0, help="Dataset index to inspect")
	p_inspect.add_argument("--max_length", type=int, default=32)
	p_inspect.add_argument("--steps", type=int, default=25)
	p_inspect.add_argument("--min_mask_prob", type=float, default=0.15)
	p_inspect.add_argument("--max_mask_prob", type=float, default=0.95)
	p_inspect.add_argument("--t", type=int, default=0, help="Fixed timestep (0 = random)")
	p_inspect.add_argument("--seed", type=int, default=1234)
	p_inspect.add_argument("--device", default=TrainConfig.device)

	p_eval = sub.add_parser("evaluate")
	p_eval.add_argument("--ckpt", required=True)
	p_eval.add_argument("--steps", type=int, default=100)
	p_eval.add_argument("--device", default=TrainConfig.device)
	p_eval.add_argument("--n_eval", type=int, default=50)
	p_eval.add_argument("--n_samples", type=int, default=5)

	args = parser.parse_args()
	args.device = resolve_device(getattr(args, "device", "cpu"))

	if args.cmd == "train":
		cfg = TrainConfig(
			model_name=args.model_name,
			split=args.split,
			max_length=args.max_length,
			batch_size=args.batch_size,
			lr=args.lr,
			epochs=args.epochs,
			max_train_batches=args.max_train_batches,
			log_every=args.log_every,
			warmup_steps=args.warmup_steps,
			steps=args.steps,
			min_mask_prob=args.min_mask_prob,
			max_mask_prob=args.max_mask_prob,
			grad_accum_steps=args.grad_accum_steps,
			dataset_version=args.dataset_version,
			lr_schedule=args.lr_schedule,
			unfreeze_embeddings=args.unfreeze_embeddings,
			output_dir=args.output_dir,
			seed=args.seed,
			device=args.device,
			use_timestep_conditioning=args.use_timestep_conditioning,
		)
		train(cfg)
		return

	if args.cmd == "sample":
		set_seed(args.seed)
		tokenizer = BertTokenizerFast.from_pretrained(args.ckpt)
		if args.use_timestep_conditioning:
			model = TimestepConditionedBertForMaskedLM.from_pretrained(args.ckpt, num_steps=args.steps, device=args.device)
			model.eval()
		else:
			model = BertForMaskedLM.from_pretrained(args.ckpt).to(args.device)
			model.eval()

		text = generate_d3pm(
			model,
			tokenizer,
			length=args.length,
			steps=args.steps,
			min_mask_prob=args.min_mask_prob,
			max_mask_prob=args.max_mask_prob,
			device=args.device,
			temperature=args.temperature,
			top_k=args.top_k,
		)
		print(text)
		return

	if args.cmd == "evaluate":
		tokenizer = BertTokenizerFast.from_pretrained(args.ckpt)
		model = TimestepConditionedBertForMaskedLM.from_pretrained(
			args.ckpt, num_steps=args.steps, device=args.device)
		model.eval()

		# Load validation sentences
		from data.dataset import WikiTextDataset
		ds = WikiTextDataset(
			split="validation", 
			max_length=128,
			dataset_version="wikitext-2-raw-v1"
		)
		sentences = [ds.texts[i] for i in 
		             range(min(args.n_eval, len(ds)))]

		# Reconstruction accuracy
		result = reconstruction_accuracy(
			model, tokenizer, sentences,
			mask_prob=0.15, steps=args.steps, 
			device=args.device
		)
		print(f"\nReconstruction accuracy: "
		      f"{result['accuracy']*100:.1f}%  "
		      f"({result['correct']}/{result['total']} tokens)")

		# Sample generation
		print(f"\nGenerated samples:")
		for i in range(args.n_samples):
			text = generate_d3pm(
				model, tokenizer,
				length=48, steps=args.steps,
				min_mask_prob=0.15, max_mask_prob=0.50,
				device=args.device,
				temperature=1.0, top_k=50,
			)
			print(f"  [{i+1}] {text}")
		return

	if args.cmd == "inspect-mask":
		set_seed(args.seed)
		tokenizer = BertTokenizerFast.from_pretrained(TrainConfig.model_name)
		ds = WikiTextDataset(split=args.split, max_length=args.max_length)
		if args.idx < 0 or args.idx >= len(ds):
			raise ValueError(f"idx out of range: 0..{len(ds)-1}")
		x0 = ds[args.idx].to(args.device)

		if args.t == 0:
			t_int = int(torch.randint(low=1, high=args.steps + 1, size=(1,)).item())
		else:
			t_int = int(args.t)

		p_t = float(
			mask_prob_for_t(
				torch.tensor([t_int], device=args.device),
				steps=args.steps,
				min_p=args.min_mask_prob,
				max_p=args.max_mask_prob,
			)[0]
		)

		x_t, mask = make_noised_example(
			x0,
			tokenizer=tokenizer,
			t_int=t_int,
			steps=args.steps,
			min_mask_prob=args.min_mask_prob,
			max_mask_prob=args.max_mask_prob,
			device=args.device,
		)

		orig_text = tokenizer.decode(x0, skip_special_tokens=True)
		masked_text = tokenizer.decode(x_t, skip_special_tokens=True)
		orig_tokens = tokenizer.convert_ids_to_tokens(x0.tolist())
		masked_tokens = tokenizer.convert_ids_to_tokens(x_t.tolist())

		num_masked = int(mask.sum().item())
		print(f"split={args.split} idx={args.idx} max_length={args.max_length}")
		print(f"t={t_int}/{args.steps}  p_t={p_t:.3f}  masked={num_masked}/{int((~(x0.eq(tokenizer.cls_token_id)|x0.eq(tokenizer.sep_token_id)|x0.eq(tokenizer.pad_token_id))).sum().item())}")
		print("\nORIGINAL (decoded):")
		print(" ".join(orig_text.split()))
		print("\nMASKED (decoded):")
		print(" ".join(masked_text.split()))
		print("\nTOKENS (original):")
		print(orig_tokens)
		print("\nTOKENS (masked):")
		print(masked_tokens)
		return

	raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
	main()
