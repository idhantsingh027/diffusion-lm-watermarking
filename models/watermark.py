"""
Watermarking for Discrete Diffusion Language Models

This module implements logit-bias watermarking for D3PM-style text generation.
The approach is adapted from Kirchenbauer et al. (2023) "A Watermark for LLMs"
but modified for the iterative denoising process of diffusion models.

Key idea:
- During generation, we bias the model toward "green list" tokens
- The green list is determined by hashing a secret key + position
- Detection checks if output has statistically more green tokens than random

For diffusion LMs specifically:
- We inject bias at each denoising step when unmasking tokens
- The watermark signal accumulates across the reverse process
"""

import hashlib
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast

from models.diffusion_lm import (
    TimestepConditionedBertForMaskedLM,
    mask_prob_for_t,
)


@dataclass
class WatermarkConfig:
    """Configuration for watermark generation and detection."""
    
    # Secret key for watermark (change this to your own secret)
    secret_key: str = "diffusion-lm-watermark-2024"
    
    # Fraction of vocab that goes into "green list" (biased tokens)
    green_fraction: float = 0.5
    
    # Bias added to green list logits (higher = stronger watermark, lower quality)
    # Note: 5.0 works well with top_k=20; use higher for smaller top_k
    bias_strength: float = 5.0
    
    # For detection: minimum tokens needed for statistical significance
    min_tokens_for_detection: int = 25
    
    # Z-score threshold for watermark detection (higher = fewer false positives)
    z_threshold: float = 4.0


def _hash_to_seed(key: str, position: int) -> int:
    """Hash secret key + position to get a deterministic seed."""
    data = f"{key}:{position}".encode("utf-8")
    h = hashlib.sha256(data).hexdigest()
    return int(h[:8], 16)


def get_green_list(
    vocab_size: int,
    position: int,
    config: WatermarkConfig,
) -> torch.Tensor:
    """
    Get the "green list" token IDs for a given position.
    
    The green list is a random subset of the vocabulary, determined
    by hashing the secret key + position. This makes it:
    - Deterministic (same position always gets same green list)
    - Unpredictable without the secret key
    
    Returns: Boolean tensor of shape [vocab_size] where True = green token
    """
    seed = _hash_to_seed(config.secret_key, position)
    generator = torch.Generator().manual_seed(seed)
    
    # Random permutation of vocab indices
    perm = torch.randperm(vocab_size, generator=generator)
    
    # First green_fraction of permutation are "green"
    num_green = int(vocab_size * config.green_fraction)
    green_indices = perm[:num_green]
    
    green_mask = torch.zeros(vocab_size, dtype=torch.bool)
    green_mask[green_indices] = True
    
    return green_mask


def apply_watermark_bias(
    logits: torch.Tensor,
    positions: torch.Tensor,
    config: WatermarkConfig,
    special_token_ids: Optional[set] = None,
) -> torch.Tensor:
    """
    Apply watermark bias to logits during generation.
    
    Args:
        logits: [batch, seq_len, vocab_size] or [batch, vocab_size]
        positions: [seq_len] or scalar - position indices for green list
        config: Watermark configuration
        special_token_ids: Set of token IDs to never bias (CLS, SEP, PAD, MASK)
    
    Returns:
        Modified logits with bias added to green list tokens
    """
    vocab_size = logits.shape[-1]
    device = logits.device
    
    # Handle both [B, V] and [B, L, V] shapes
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)
        positions = torch.tensor([positions]) if isinstance(positions, int) else positions
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, seq_len, _ = logits.shape
    biased_logits = logits.clone()
    
    for pos_idx in range(seq_len):
        pos = int(positions[pos_idx]) if positions.dim() > 0 else int(positions)
        green_mask = get_green_list(vocab_size, pos, config).to(device)
        
        # Don't bias special tokens
        if special_token_ids:
            for tok_id in special_token_ids:
                if tok_id < vocab_size:
                    green_mask[tok_id] = False
        
        # Add bias to green tokens
        biased_logits[:, pos_idx, green_mask] += config.bias_strength
    
    if squeeze_output:
        biased_logits = biased_logits.squeeze(1)
    
    return biased_logits


@torch.no_grad()
def generate_d3pm_watermarked(
    model: TimestepConditionedBertForMaskedLM,
    tokenizer: BertTokenizerFast,
    *,
    length: int,
    steps: int,
    min_mask_prob: float,
    max_mask_prob: float,
    device: str,
    temperature: float = 0.7,
    top_k: int = 20,
    config: Optional[WatermarkConfig] = None,
) -> str:
    """
    Generate text with watermark using D3PM reverse process.
    
    This is identical to generate_d3pm() but injects watermark bias
    at each denoising step when sampling tokens.
    
    Args:
        model: Trained diffusion LM
        tokenizer: BERT tokenizer
        length: Output sequence length
        steps: Number of diffusion steps
        min_mask_prob, max_mask_prob: Masking schedule bounds
        device: torch device
        temperature: Sampling temperature
        top_k: Top-k filtering
        config: Watermark configuration (None = no watermark)
    
    Returns:
        Generated text string
    """
    if config is None:
        config = WatermarkConfig()
    
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer has no [MASK] token")
    if length < 4:
        raise ValueError("length must be >= 4")
    
    # Special tokens to never watermark
    special_ids = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        tokenizer.mask_token_id,
    }
    special_ids = {x for x in special_ids if x is not None}
    
    # Start with fully masked sequence
    input_ids = torch.full(
        (1, length), 
        fill_value=tokenizer.mask_token_id, 
        device=device, 
        dtype=torch.long
    )
    input_ids[0, 0] = tokenizer.cls_token_id
    input_ids[0, -1] = tokenizer.sep_token_id
    
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    
    # Reverse diffusion process
    for t_int in range(steps, 0, -1):
        t_tensor = torch.tensor([t_int], device=device)
        p_t = float(mask_prob_for_t(t_tensor, steps=steps, min_p=min_mask_prob, max_p=max_mask_prob)[0])
        
        if t_int == 1:
            p_prev = 0.0
        else:
            p_prev = float(mask_prob_for_t(
                torch.tensor([t_int - 1], device=device), 
                steps=steps, min_p=min_mask_prob, max_p=max_mask_prob
            )[0])
        
        p_t = max(p_t, 1e-6)
        keep_mask_prob = min(max(p_prev / p_t, 0.0), 1.0)
        
        # Find masked positions
        special = input_ids.eq(cls_id) | input_ids.eq(sep_id)
        if pad_id is not None:
            special |= input_ids.eq(pad_id)
        
        is_mask = input_ids.eq(tokenizer.mask_token_id) & (~special)
        if not bool(is_mask.any()):
            continue
        
        # Decide which stay masked
        stay_masked = (torch.rand_like(input_ids.float()) < keep_mask_prob) & is_mask
        to_unmask = is_mask & (~stay_masked)
        if not bool(to_unmask.any()):
            continue
        
        # Get model predictions
        out = model(input_ids=input_ids, t=t_tensor)
        logits = out["logits"]
        
        # Temperature first
        logits = logits / max(temperature, 1e-6)
        
        # Top-k filtering BEFORE watermark (get candidate pool)
        if top_k > 0:
            # Expand top_k to give watermark room to work
            effective_k = min(top_k * 3, logits.shape[-1])  # 3x candidates
            topk_vals, topk_idx = torch.topk(logits, k=effective_k, dim=-1)
            
            # === WATERMARK INJECTION ===
            # Apply bias only to green tokens within top-k candidates
            for b in range(logits.shape[0]):
                for pos in range(logits.shape[1]):
                    content_pos = max(0, pos - 1)  # Subtract 1 for [CLS]
                    green_mask = get_green_list(logits.shape[-1], content_pos, config)
                    for k_idx in range(effective_k):
                        tok_id = topk_idx[b, pos, k_idx].item()
                        if tok_id not in special_ids and green_mask[tok_id]:
                            topk_vals[b, pos, k_idx] += config.bias_strength
            
            # Now filter to actual top_k after biasing
            _, rerank_idx = torch.topk(topk_vals, k=min(top_k, effective_k), dim=-1)
            final_idx = torch.gather(topk_idx, -1, rerank_idx)
            final_vals = torch.gather(topk_vals, -1, rerank_idx)
            
            filtered = torch.full_like(logits, fill_value=-float("inf"))
            filtered.scatter_(-1, final_idx, final_vals)
            logits = filtered
        
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(1, -1)
        input_ids[to_unmask] = sampled[to_unmask]
    
    # Final fill for any remaining masks
    remaining = input_ids.eq(tokenizer.mask_token_id) & (~(input_ids.eq(cls_id) | input_ids.eq(sep_id)))
    if pad_id is not None:
        remaining &= ~input_ids.eq(pad_id)
    
    if bool(remaining.any()):
        out = model(input_ids=input_ids, t=torch.tensor([1], device=device))
        logits = out["logits"]
        logits = logits / max(temperature, 1e-6)
        
        if top_k > 0:
            effective_k = min(top_k * 3, logits.shape[-1])
            topk_vals, topk_idx = torch.topk(logits, k=effective_k, dim=-1)
            
            # Apply watermark bias within candidates
            for b in range(logits.shape[0]):
                for pos in range(logits.shape[1]):
                    content_pos = max(0, pos - 1)
                    green_mask = get_green_list(logits.shape[-1], content_pos, config)
                    for k_idx in range(effective_k):
                        tok_id = topk_idx[b, pos, k_idx].item()
                        if tok_id not in special_ids and green_mask[tok_id]:
                            topk_vals[b, pos, k_idx] += config.bias_strength
            
            _, rerank_idx = torch.topk(topk_vals, k=min(top_k, effective_k), dim=-1)
            final_idx = torch.gather(topk_idx, -1, rerank_idx)
            final_vals = torch.gather(topk_vals, -1, rerank_idx)
            
            filtered = torch.full_like(logits, fill_value=-float("inf"))
            filtered.scatter_(-1, final_idx, final_vals)
            logits = filtered
        
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(1, -1)
        input_ids[remaining] = sampled[remaining]
    
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return " ".join(text.split())


def detect_watermark(
    text: str,
    tokenizer: BertTokenizerFast,
    config: Optional[WatermarkConfig] = None,
) -> dict:
    """
    Detect if text contains a watermark.
    
    Uses a statistical test: if the text has significantly more green-list
    tokens than expected by chance (50%), it's likely watermarked.
    
    Args:
        text: Text to check
        tokenizer: BERT tokenizer
        config: Watermark configuration (must match generation config)
    
    Returns:
        dict with:
            - is_watermarked: bool
            - z_score: statistical significance
            - green_fraction: observed fraction of green tokens
            - num_tokens: number of tokens analyzed
            - p_value: probability of seeing this many green tokens by chance
    """
    import math
    
    if config is None:
        config = WatermarkConfig()
    
    # Tokenize
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = enc["input_ids"][0].tolist()
    
    # Filter out special tokens
    special_ids = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        tokenizer.mask_token_id,
        tokenizer.unk_token_id,
    }
    special_ids = {x for x in special_ids if x is not None}
    
    vocab_size = tokenizer.vocab_size
    
    # Count green tokens at each position
    green_count = 0
    total_count = 0
    
    for pos, tok_id in enumerate(token_ids):
        if tok_id in special_ids:
            continue
        
        green_mask = get_green_list(vocab_size, pos, config)
        if green_mask[tok_id]:
            green_count += 1
        total_count += 1
    
    if total_count < config.min_tokens_for_detection:
        return {
            "is_watermarked": False,
            "z_score": 0.0,
            "green_fraction": 0.0,
            "num_tokens": total_count,
            "p_value": 1.0,
            "message": f"Too few tokens ({total_count} < {config.min_tokens_for_detection})",
        }
    
    # Statistical test
    # Under null hypothesis (no watermark): each token has green_fraction chance of being green
    expected_green = total_count * config.green_fraction
    std_dev = math.sqrt(total_count * config.green_fraction * (1 - config.green_fraction))
    
    z_score = (green_count - expected_green) / std_dev if std_dev > 0 else 0.0
    
    # One-sided p-value (probability of seeing this many or more green tokens by chance)
    # Using normal approximation
    from math import erfc
    p_value = 0.5 * erfc(z_score / math.sqrt(2))
    
    observed_fraction = green_count / total_count
    is_watermarked = z_score > config.z_threshold
    
    return {
        "is_watermarked": is_watermarked,
        "z_score": round(z_score, 3),
        "green_fraction": round(observed_fraction, 4),
        "expected_fraction": config.green_fraction,
        "green_count": green_count,
        "num_tokens": total_count,
        "p_value": round(p_value, 6),
        "threshold": config.z_threshold,
    }


def evaluate_watermark(
    texts_watermarked: list[str],
    texts_baseline: list[str],
    tokenizer: BertTokenizerFast,
    config: Optional[WatermarkConfig] = None,
) -> dict:
    """
    Evaluate watermark effectiveness across multiple samples.
    
    Args:
        texts_watermarked: List of watermarked texts
        texts_baseline: List of non-watermarked texts (for false positive rate)
        tokenizer: BERT tokenizer
        config: Watermark configuration
    
    Returns:
        dict with detection rates and statistics
    """
    if config is None:
        config = WatermarkConfig()
    
    # Detect on watermarked texts
    wm_results = [detect_watermark(t, tokenizer, config) for t in texts_watermarked]
    wm_detected = sum(1 for r in wm_results if r["is_watermarked"])
    wm_z_scores = [r["z_score"] for r in wm_results]
    
    # Detect on baseline texts (should be false positives)
    bl_results = [detect_watermark(t, tokenizer, config) for t in texts_baseline]
    bl_detected = sum(1 for r in bl_results if r["is_watermarked"])
    bl_z_scores = [r["z_score"] for r in bl_results]
    
    import statistics
    
    return {
        "watermarked": {
            "count": len(texts_watermarked),
            "detected": wm_detected,
            "detection_rate": round(wm_detected / len(texts_watermarked), 4) if texts_watermarked else 0,
            "mean_z_score": round(statistics.mean(wm_z_scores), 3) if wm_z_scores else 0,
            "std_z_score": round(statistics.stdev(wm_z_scores), 3) if len(wm_z_scores) > 1 else 0,
        },
        "baseline": {
            "count": len(texts_baseline),
            "false_positives": bl_detected,
            "false_positive_rate": round(bl_detected / len(texts_baseline), 4) if texts_baseline else 0,
            "mean_z_score": round(statistics.mean(bl_z_scores), 3) if bl_z_scores else 0,
            "std_z_score": round(statistics.stdev(bl_z_scores), 3) if len(bl_z_scores) > 1 else 0,
        },
        "config": {
            "bias_strength": config.bias_strength,
            "green_fraction": config.green_fraction,
            "z_threshold": config.z_threshold,
        },
    }


if __name__ == "__main__":
    # Quick test
    print("Watermark module loaded successfully")
    
    config = WatermarkConfig()
    print(f"\nDefault config:")
    print(f"  Secret key: {config.secret_key[:20]}...")
    print(f"  Green fraction: {config.green_fraction}")
    print(f"  Bias strength: {config.bias_strength}")
    print(f"  Z-threshold: {config.z_threshold}")
    
    # Test green list generation
    vocab_size = 30522  # BERT vocab
    green = get_green_list(vocab_size, position=0, config=config)
    print(f"\nGreen list test (pos=0): {green.sum().item()} / {vocab_size} tokens")
    
    green2 = get_green_list(vocab_size, position=1, config=config)
    overlap = (green & green2).sum().item()
    print(f"Green list overlap (pos=0 vs pos=1): {overlap} tokens")
