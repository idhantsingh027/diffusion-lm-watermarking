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
	max_length: int = 64
	batch_size: int = 16
	lr: float = 5e-5
	weight_decay: float = 0.0
	epochs: int = 1
	max_train_batches: int = 0
	log_every: int = 50
	warmup_steps: int = 0
	steps: int = 25
	min_mask_prob: float = 0.15
	max_mask_prob: float = 0.95
	output_dir: str = "checkpoints/bert-mlm-diffusion-baseline"
	seed: int = 1234
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
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


def mask_prob_for_t(t: torch.Tensor, *, steps: int, min_p: float, max_p: float) -> torch.Tensor:
	"""Linear schedule from min_p..max_p over timesteps 1..steps."""
	if steps <= 1:
		return torch.full_like(t, fill_value=max_p, dtype=torch.float32)
	frac = (t.float() - 1.0) / float(steps - 1)
	return min_p + frac * (max_p - min_p)


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
	t = torch.randint(low=1, high=steps + 1, size=(bsz,), device=device)
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


def train(cfg: TrainConfig) -> None:
	set_seed(cfg.seed)
	os.makedirs(cfg.output_dir, exist_ok=True)
	log_every = max(int(cfg.log_every), 0)

	tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
	dataset = WikiTextDataset(split=cfg.split, max_length=cfg.max_length)
	loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

	if cfg.use_timestep_conditioning:
		model = TimestepConditionedBertForMaskedLM.from_pretrained(cfg.model_name, num_steps=cfg.steps, device=cfg.device)
	else:
		model = BertForMaskedLM.from_pretrained(cfg.model_name).to(cfg.device)
	model.train()

	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
	total_steps = cfg.epochs * len(loader)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)

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
			loss.backward()

			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad(set_to_none=True)

			running += float(loss.item())
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
	p_train.add_argument("--steps", type=int, default=TrainConfig.steps)
	p_train.add_argument("--min_mask_prob", type=float, default=TrainConfig.min_mask_prob)
	p_train.add_argument("--max_mask_prob", type=float, default=TrainConfig.max_mask_prob)
	p_train.add_argument("--output_dir", default=TrainConfig.output_dir)
	p_train.add_argument("--seed", type=int, default=TrainConfig.seed)
	p_train.add_argument("--device", default=TrainConfig.device)
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
			steps=args.steps,
			min_mask_prob=args.min_mask_prob,
			max_mask_prob=args.max_mask_prob,
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