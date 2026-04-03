# A Principled Approach to Watermarking Diffusion Language Models

---

## Abstract

As AI-generated text becomes increasingly indistinguishable from human writing, the ability to verify whether a piece of text was produced by a specific AI system has become a critical problem. This paper presents a complete implementation and evaluation of a watermarking system for discrete diffusion language models. We train a D3PM-style diffusion LM using a 3-stage curriculum, embed hidden statistical signals into generated text using a green/red token list scheme, and evaluate detection performance, text quality tradeoffs, and robustness against adversarial attacks. Our system achieves 95% detection rate with 0% false positives at optimal settings, and we introduce a position-independent robust mode that maintains 100% detection even after 20% word deletion — an attack that completely defeats the standard approach.

---

## 1. Introduction

### 1.1 The Problem

Imagine you are a researcher who has spent months training a language model. Someone else downloads it, generates thousands of articles using your model, and publishes them as human-written content. How do you prove that text came from your model? How do you detect misuse?

This is not a hypothetical. As AI text generation becomes more capable and accessible, distinguishing human from AI content is increasingly difficult — and increasingly important. The stakes include:

- **Academic integrity** — detecting AI-written essays submitted as student work
- **Misinformation** — identifying AI-generated fake news at scale
- **Intellectual property** — proving a model was used without authorization
- **AI safety** — tracking the provenance of AI outputs in the wild

Watermarking offers a solution: embed a hidden, statistically detectable signal into every piece of text the model generates. The signal is invisible to humans but measurable by anyone who holds the secret key.

### 1.2 Why Diffusion Models Are Different

Most watermarking research focuses on autoregressive models — models like GPT that generate text one token at a time, left to right. For these models, watermarking is relatively straightforward: bias each token selection toward a "green list" of preferred tokens as you go.

Diffusion language models work completely differently. Instead of generating token by token, they:

1. Start with a completely masked (noisy) sequence
2. Iteratively unmask tokens over many steps
3. All positions are updated in parallel at each step

**Example of the diffusion process:**

```
Step 100 (fully masked):
[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]

Step 75 (partially denoised):
[MASK]  he   [MASK] studied [MASK]  at  [MASK]

Step 50 (more revealed):
 naresh  he   [MASK] studied philosophy  at  [MASK]

Step 1 (almost complete):
 naresh  gained  his  degree  in  philosophy  from  oxford

Step 0 (final output):
"naresh gained his undergraduate degree in philosophy from oxford"
```

This parallel, iterative process means classical watermarking techniques do not directly apply. We need a method that works with the diffusion process rather than against it.

### 1.3 Our Approach

We address this by:

1. Training a discrete diffusion LM from scratch using a carefully designed curriculum
2. Injecting watermarks by biasing the model's token selection during the denoising process
3. Detecting watermarks using a statistical z-test on the green/red token ratio
4. Evaluating robustness against realistic adversarial attacks

---

## 2. Background

### 2.1 What Is a Language Model?

A language model is a system trained to understand and generate human language. Given a partial sentence, it assigns probabilities to every possible next word. For example:

```
Input:  "The capital of France is ___"
Output probabilities:
  "Paris"    → 94.2%
  "London"   → 0.3%
  "Berlin"   → 0.2%
  ...
```

The model learns these probabilities by training on large amounts of text.

### 2.2 What Is a Diffusion Model?

A diffusion model learns to reverse a corruption process. In our case, the corruption is masking: we randomly hide tokens with a `[MASK]` symbol, and the model learns to reconstruct the original text.

**Forward process (corruption):**
```
Original:  "he studied philosophy at oxford university"
Masked:    "he [MASK] philosophy at [MASK] university"
```

**Reverse process (denoising):**
```
Start:     "[MASK] [MASK] [MASK] [MASK] [MASK] [MASK]"
Step 1:    "[MASK] studied [MASK] at [MASK] [MASK]"
Step 2:    "he studied philosophy at [MASK] university"
Final:     "he studied philosophy at oxford university"
```

The key insight is that by training the model to undo masking, it learns a deep understanding of language structure. Then at generation time, we start from all masks and run the reverse process to generate new text from scratch.

### 2.3 What Is Watermarking?

A watermark is a hidden signal embedded in content to prove ownership or track provenance. Physical banknotes have visible watermarks. Digital images can have invisible watermarks embedded in their pixel values.

For text, we embed the watermark statistically: the generated text should use certain words more often than chance. A detector that knows the secret pattern can measure this bias and determine whether the text was watermarked.

**Key properties we want:**
- **Invisible** — humans cannot tell watermarked from non-watermarked text
- **Detectable** — a detector with the secret key can reliably find it
- **Robust** — the signal survives common edits (truncation, reordering, etc.)

---

## 3. Model Architecture and Training

### 3.1 Base Architecture

Our model is built on BERT (Bidirectional Encoder Representations from Transformers), a pre-trained language model with 110 million parameters. We extend it with:

- **Timestep conditioning** — the model receives the current noise level `t` as an additional input, allowing it to adapt its predictions to the difficulty of the denoising step
- **Masking diffusion** — we use token masking as our corruption process (D3PM-style)

**Why timestep conditioning matters:**

Without it, the model sees all noise levels as the same task. At step 75 (many masks), it should be uncertain and predict likely words. At step 5 (few masks), it should be precise and fill in exact words. Knowing `t` lets the model calibrate accordingly.

```
Without conditioning:  Same prediction strategy for all noise levels
With conditioning:     "I'm at t=75, I'll predict high-probability words"
                       "I'm at t=5, I should be very precise here"
```

### 3.2 Curriculum Training

We train the model in three stages, progressively increasing the difficulty:

| Stage | Masking Rate | Epochs | Learning Rate | Purpose |
|-------|-------------|--------|---------------|---------|
| Stage 1 | 15% | 5 | 2×10⁻⁵ | Learn basic language patterns with light corruption |
| Stage 2 | 50% | 8 | 1×10⁻⁵ | Handle moderate noise, unfreeze embeddings |
| Stage 3 | 75% | 6 | 5×10⁻⁶ | Reconstruct severely corrupted text |

**Why curriculum training?**

Starting directly at 75% masking is like asking a student to sit an exam before learning the basics. The model would struggle with too many unknowns simultaneously. By starting easy and progressively increasing difficulty, the model builds foundational understanding before tackling harder problems.

**Analogy:** A student learning to fill in blanks:

```
Stage 1 (easy):   "The cat sat on the ____."         (1 blank)
Stage 2 (medium): "The ____ sat on the ____."        (2 blanks)
Stage 3 (hard):   "The ____ ____ on ____ ____."      (4 blanks)
```

### 3.3 Training Results

| Checkpoint | Reconstruction Accuracy | GPT-2 Perplexity |
|------------|------------------------|------------------|
| Stage 1 best | 64.13% ± 1.18% | 146.04 ± 13.11 |
| Stage 2 best | 64.41% ± 1.14% | 134.03 ± 13.94 |
| **Stage 3 best** | **64.38% ± 0.88%** | **122.78 ± 15.32** |

Results are reported as mean ± standard deviation across 5 random seeds, ensuring fair comparison.

**What these numbers mean:**

- **Reconstruction accuracy** — given masked text, what fraction of hidden tokens does the model correctly predict? 64% means the model correctly fills in roughly 2 out of every 3 masked words.
- **Perplexity (PPL)** — how surprised is GPT-2 when it reads the generated text? Lower is better. A perplexity of 122 means the text is reasonably fluent; 1000+ would indicate gibberish.

**Why Stage 3 wins despite harder training:** The curriculum worked. Each stage improved the model's internal representations, and Stage 3 — despite handling 75% masking — produces the lowest perplexity under optimized decoding (59.46 ± 9.61 at temperature=0.7, top_k=20).

---

## 4. Watermarking Method

### 4.1 The Green/Red List Scheme

Our watermarking method is based on a token list partition. The entire vocabulary (roughly 30,000 words) is divided into two groups using a secret key:

- **Green list** — tokens that are slightly preferred during generation
- **Red list** — tokens that are slightly penalized during generation

**Example:**

Suppose our vocabulary contains these words, and the secret key assigns them as follows:

```
GREEN: "the", "university", "studied", "oxford", "philosophy", "degree"
RED:   "a", "college", "learned", "cambridge", "history", "certificate"
```

Without watermarking, the model might generate:
```
"he learned history at cambridge college"
```

With watermarking (green list bias), the model is steered toward:
```
"he studied philosophy at oxford university"
```

Both sentences are equally grammatical. The difference is invisible to a human reader. But a detector that knows the secret key can check: how many of these words are green? If the ratio is much higher than 50%, the text was likely watermarked.

### 4.2 How the Bias Is Injected

During the denoising process, every time the model predicts a token, we modify its probability distribution:

```python
# Model predicts raw probabilities
logits = model(masked_text, t=current_step)

# We add a bonus to green tokens
for token in vocabulary:
    if is_green(token, secret_key):
        logits[token] += bias_strength   # e.g., +3.0
    # red tokens are unchanged (relative penalty)

# Sample from the modified distribution
next_token = sample(softmax(logits))
```

This is done at every denoising step, so the green bias accumulates across the entire generation.

### 4.3 Detection: The Z-Test

Detection works by measuring whether the text contains significantly more green tokens than would be expected by chance.

**Setup:**
- Vocabulary is 50% green, 50% red (by design)
- A random text should have ~50% green tokens
- A watermarked text should have noticeably more than 50%

#### 4.3.1 The Binomial Model

Under **no watermark**, each token has:
```
P(green) = 0.5
```

#### 4.3.2 Why μ = n × 0.5?

For a binomial distribution with n trials and probability p:
```
Expected value = n × p
```

In our case:
- n = total tokens
- p = 0.5 (green fraction with no watermark)

Therefore:
```
μ = n × 0.5
```

**Interpretation:** μ is the **expected number of green tokens if there is NO watermark**. For 64 tokens, we expect 32 green tokens by random chance.

#### 4.3.3 Why σ = √(n × 0.5 × 0.5)?

The standard deviation of a binomial distribution is:
```
σ = √(n × p × (1 - p))
```

Since p = 0.5:
```
σ = √(n × 0.5 × 0.5)
```

**Interpretation:** σ tells us **how much natural fluctuation to expect due to randomness**. For 64 tokens, σ = 4, meaning we typically see random deviations of ±4 tokens from the expected 32.

#### 4.3.4 Why the Z-Score?

The z-score formula is:
```
z = (k - μ) / σ
```

This measures **how far the observed value is from expected, in units of randomness**.

Breaking it down:
- k = observed green tokens
- μ = expected (no watermark)
- σ = typical random fluctuation

So:
```
z = (actual - expected) / noise
```

#### 4.3.5 Final test:

For a text with `n` total tokens, `k` of which are green:

```
Expected green tokens:  μ = n × 0.5
Standard deviation:     σ = √(n × 0.5 × 0.5)
Z-score:               z = (k - μ) / σ
```

**One-line summary:**
> The z-score tells us how many "random fluctuations" away the observed green count is — if it's too far (z > 4), it's not random → it's watermarked.

#### 4.3.6 Worked example with actual results:

For a 64-token watermarked text:
```
n = 64 tokens
μ = 64 × 0.5 = 32 expected green tokens
σ = √(64 × 0.25) = 4

If we observe k = 52 green tokens:
z = (52 - 32) / 4 = 5.0  →  Detected as watermarked
```

For a 64-token non-watermarked text:
```
If we observe k = 33 green tokens (roughly 50%):
z = (33 - 32) / 4 = 0.25  →  Not watermarked
```

**Decision rule:** If z > 4.0, flag as watermarked.

### 4.4 Why Z-Threshold = 4.0?

The threshold controls the tradeoff between catching watermarked text and falsely accusing innocent text. The z-test provides provable statistical guarantees:

| Z-threshold | False positive probability | Meaning |
|-------------|--------------------------|---------|
| 3.0 | 1 in 770 | Too risky for production use |
| **4.0** | **1 in 31,250** | **Our chosen threshold** |
| 5.0 | 1 in 3,500,000 | Extremely conservative |

At z=4.0, a non-watermarked text would only be falsely accused once in every 31,250 checks. In practice, the results showed 0 false positives across 20 baseline samples — consistent with this theoretical guarantee.

---

## 5. Evaluation

### 5.1 Experimental Setup

- **Primary model:** Stage 3 best checkpoint
- **Decoding:** temperature=0.7, top_k=20
- **Evaluation protocol:** 5 seeds (seeds=[11,22,33,42,77]), n=20 samples per condition
- **Canonical baseline:** PPL = 59.46 ± 9.61, reconstruction accuracy = 64.38% ± 0.88%

### 5.2 Detection Performance

| Condition | Samples | Detection Rate | Mean Z-score |
|-----------|---------|---------------|--------------|
| Watermarked | 20 | **95%** (19/20) | 6.84 ± 1.37 |
| Baseline (no watermark) | 20 | **0% false positives** | 0.04 ± 1.04 |

**Sample watermarked outputs (from actual results):**

```
[z=7.62] "finale (the second half of a opera, repeated). in an example, 
          the first line of the aria is sung by the tenor..."

[z=7.11] "one year later he also participated in the competition, 
          this was the final victory of the season..."

[z=6.60] "naresh gained his undergraduate degree from oxford university, 
          a graduate - and..."
```

**Sample baseline outputs (no watermark):**

```
[z=-1.02] "the son of james, james, he was born in ireland. he studied 
           in new england, new york..."

[z=-0.76] "the film's star was directed plays a role in s. frayne's 
           novel, john of..."
```

Both sets look equally natural. The z-score difference is entirely statistical.

**The one miss (sample 16, z=1.41):**
```
[z=1.41] "in 2000, the second run, the last on the titles was to by on 
          abc, but it never c..."
```
This sample happened to contain fewer green tokens by random chance during sampling. This is expected — with 95% detection, approximately 1 in 20 samples will fall below threshold.

### 5.3 Bias Strength Tradeoff

The bias strength parameter controls how strongly we push the model toward green tokens. This creates a fundamental tradeoff between detection reliability and text quality.

| Bias Strength | Detection Rate | GPT-2 PPL | Quality Impact |
|--------------|---------------|-----------|----------------|
| 1.0 | 0% | 62 | Imperceptible — but watermark invisible |
| 2.0 | 73% | 65 | Minimal |
| **3.0** | **93%** | **65** | **Minimal — best** |
| 5.0 | 100% | 88 | Noticeable degradation |
| 7.0 | 100% | 91 | Significant degradation |
| 10.0 | 100% | 114 | Severe degradation |

**What each level looks like in practice:**

*Bias = 1.0 (undetectable):*
```
"he studied at the university of chicago and later became 
a professor of philosophy"
```
Perfectly natural. But green/red ratio is indistinguishable from chance. Z-score ~0.

*Bias = 3.0 (sweet spot):*
```
"naresh gained his undergraduate degree from oxford university, 
a graduate student who later became known for his research"
```
Still reads naturally. PPL barely changed (62→65). Z-score reliably above 4.0.

*Bias = 5.0 (noticeable):*
```
"anderson had his baby, john, in the united in, in 1947 
and was given his third c..."
```
"In the united in" is unnatural — the model picked a green-list token ("in") over the more natural "states" because "states" was on the red list. Quality degrades visibly.

*Bias = 10.0 (severe):*
```
"( g, ( d & v ;, g ), f and h, a, b, e, ( r, - h ), and (, ) ;"
```
The model abandons natural language entirely to satisfy the green-list constraint. Useless in practice.

**Conclusion:** Bias = 3.0 is optimal. It achieves 93% detection with only an 8% PPL increase (62→65) — an imperceptible quality cost for a reliable watermark.

---

## 6. Robustness Analysis

A watermark is only useful if it survives realistic editing. We test four attack types.

### 6.1 Truncation Attack

**What it is:** An adversary cuts off part of the text, keeping only the beginning.

**Why it might work:** Fewer tokens means less statistical signal, potentially dropping the z-score below threshold.

**Results:**

| Attack | Detection Rate | Interpretation |
|--------|---------------|----------------|
| Original (100%) | 93.3% | Baseline |
| Keep 70% | 93.3% | No degradation |
| Keep 50% | 86.7% | Slight reduction |

**Example:**

*Original (z=6.70, 64 tokens):*
```
"naresh gained his undergraduate degree from oxford university, a graduate 
and doctoral student in the department of philosophy. he later moved to 
cambridge where he continued his research into metaphysics and ethics."
```

*Truncated to 70% (z=5.60, 45 tokens):*
```
"naresh gained his undergraduate degree from oxford university, a graduate 
and doctoral student in the department of philosophy."
```

Still detected. The surviving tokens still carry the green-list bias. Think of it like cutting a watermarked banknote — the watermark pattern remains visible on the half you kept.

### 6.2 Word Swapping Attack

**What it is:** An adversary reorders words within sentences, perhaps to disguise the text's origin.

**Results:** 80–93% detection. The watermark survives well because the same words are present — just in different positions.

**Example:**
```
Original:  "naresh gained his degree from oxford university"
Swapped:   "naresh gained his oxford degree from university"
```
Most words are unchanged. Green tokens remain green. Z-score is largely preserved.

### 6.3 Word Deletion Attack — The Critical Vulnerability

**What it is:** An adversary randomly deletes 10–20% of words throughout the text.

**Results (standard mode): 3.3% detection** — the watermark is completely destroyed.

**Why this happens:**

In standard mode, token color depends on both the token identity and its position:
```python
is_green = hash(secret_key + str(token_id) + str(position))
```

**Before deletion:**
```
Position:  1        2       3      4        5       6
Word:      naresh   gained  his    degree   from    oxford
Color:     GREEN    GREEN   RED    GREEN    RED     GREEN
```

**After deleting "gained" (position 2) and "his" (position 3):**
```
Position:  1        2        3       4
Word:      naresh   degree   from    oxford
Color:     GREEN    RED      GREEN   RED   ← colors REASSIGNED!
```

"degree" was GREEN at position 4. Now it's at position 2, which hashes to RED. The carefully embedded signal is scrambled. Z-score collapses to near zero.

This is not a gradual degradation — it is catastrophic. Deletion of just 20% of words reduces detection from 93.3% to 3.3%.

### 6.4 Robust Mode — The Fix

**Core idea:** Make token color depend only on the token itself, not its position.

```python
# Standard mode (position-sensitive)
is_green = hash(secret_key + str(token_id) + str(position))

# Robust mode (position-independent)
is_green = hash(secret_key + str(token_id))                   
```

Now "degree" (token ID 8901) is always GREEN regardless of where it appears in the text.

**After deletion with robust mode:**
```
BEFORE: naresh(G)  gained(G)  his(R)  degree(G)  from(R)  oxford(G)
AFTER:  naresh(G)             degree(G)  from(R)  oxford(G)
```

Two words deleted, but the surviving words keep their colors. Green-list bias is preserved.

**Results:**

| Mode | Detection after 20% deletion | Mean Z-score |
|------|------------------------------|-------------|
| **Robust** | **100% (10/10)** | **6.11–7.07** |
| Standard | 0% (0/10) | −1.18 to +1.41 |

The vulnerability is completely fixed. Every watermarked sample is still detected after 20% deletion.

### 6.5 Robustness Summary

| Attack | Standard Mode | Robust Mode |
|--------|--------------|-------------|
| No attack | 93.3% | ~95% |
| Truncate 50% | 86.7% | ~90% |
| Word swap 20% | 80–93% | ~95% |
| **Word deletion 20%** | **3.3%** | **100%** |

**Remaining vulnerability:** Synonym substitution. If an adversary replaces "oxford" (GREEN) with "cambridge" (RED), the token identity changes and so does its color. This is the natural next attack to address in future work.

---

## 7. Theoretical Foundations

### 7.1 The Z-Test as a Statistical Hypothesis Test

Our detection method is a formal statistical hypothesis test:

- **Null hypothesis H₀:** The text was generated without watermarking (green fraction ≈ 0.5)
- **Alternative hypothesis H₁:** The text was generated with watermarking (green fraction > 0.5)

Under H₀, the number of green tokens follows a binomial distribution with parameters n and p=0.5. By the central limit theorem, for large n this is well-approximated by a normal distribution, giving us the z-test.

**False positive guarantee:**

For threshold z=4.0:
```
P(false positive) = P(Z > 4.0) = 3.2 × 10⁻⁵
```

This means a genuinely non-watermarked text would only be falsely flagged once in every 31,250 detection attempts.

### 7.2 Why Diffusion LMs Are Well-Suited for This

Autoregressive models (GPT-style) generate tokens sequentially, where each token depends on all previous ones. Injecting a bias at position 5 changes the context for position 6, which changes position 7, and so on. The watermark signal can cascade unpredictably through the sequence.

Diffusion LMs avoid this problem because:

1. **Parallel generation** — all positions are updated simultaneously at each step
2. **Independent predictions** — the bias at each position does not affect predictions at other positions within the same step
3. **Direct logit manipulation** — we can inject bias cleanly into the logits without disrupting the generation process

This makes diffusion LMs a more principled foundation for logit-bias watermarking.

---

## 8. Discussion

### 8.1 What We Demonstrated

| Objective | Result |
|-----------|--------|
| Reliable detection | 95% TPR, 0% FPR |
| Efficient detection | Single pass, O(n) time |
| Theoretical grounding | Z-test with p < 3×10⁻⁵ |
| Robustness to truncation | 87–93% detection at 50% truncation |
| Robustness to deletion | 100% detection (robust mode) |
| Minimal quality impact | 8% PPL increase at optimal bias |

### 8.2 Limitations

**Text quality at high bias.** At bias_strength=5.0 and above, generated text begins showing unnatural word choices. This limits practical deployments to bias ≤ 3.0, where quality impact is minimal.

**Short texts.** The z-test requires enough tokens to achieve statistical significance. For very short texts (under ~20 tokens), the test loses power. Minimum recommended length is ~40 tokens.

**Paraphrasing.** A complete rewrite of the text will defeat any token-level watermark. This is an inherent limitation of the approach, not a flaw in the implementation.

**Synonym substitution.** Robust mode's position-independence introduces sensitivity to token identity changes. Future work should explore hybrid schemes that combine both approaches.

### 8.3 Future Work

Several natural extensions follow from this work:

1. **Synonym-robust watermarking** — group semantically similar tokens into the same color class so substitution attacks fail
2. **Conditional generation** — extend the model to accept topic or style prompts, making the system more practically useful
3. **Multi-key watermarking** — embed different keys for different users to track which deployment generated a specific text
4. **Theoretical impossibility bounds** — characterize the fundamental limits of watermarking under strong adversaries, following the framework of Zhang et al. (2024)

---

## 9. Conclusion

We have presented a complete watermarking system for discrete diffusion language models. Starting from a BERT-based D3PM architecture trained with a 3-stage curriculum, we implemented a green/red list watermarking scheme with logit-bias injection during denoising. Our system achieves 95% detection at 0% false positives, with only an 8% quality degradation at optimal bias strength.

The key contribution beyond standard watermarking is the position-independent robust mode, which addresses a critical vulnerability: the standard scheme's detection drops from 93% to 3% under 20% word deletion, while robust mode maintains 100% detection under the same attack.

This work demonstrates that diffusion language models are not only suitable for watermarking — they may be better suited than autoregressive models, owing to their parallel generation structure which permits clean, non-cascading logit manipulation.

---

## References

- Gloaguen, T., Staab, R., Jovanović, N., and Vechev, M. "Watermarking Diffusion Language Models."
- Sahoo, S. S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J. T., Rush, A., and Kuleshov, V. "Simple and Effective Masked Diffusion Language Models."
- Zhou, Z., Chen, L., Tong, H., and Song, D. "dLLM: Simple Diffusion Language Modeling."
