"""
grpo_train.py — GRPO fine-tuning of opt-125m on the number-addition task.

Algorithm reference:
    DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
    https://arxiv.org/abs/2402.03300  (Section 3 — GRPO)

Structure
─────────
  GRPOConfig          — all hyperparameters in one place
  Reward functions    — imported from reward_functions.py
  Helper functions    — one per concept, each annotated with tensor shapes
  train_step()        — reads like the GRPO pseudocode
  train()             — outer loop, logging, checkpointing
"""

import os
import math
import json
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

from reward_function import (
    reward_think_tags,
    reward_think_content,
    reward_answer,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GRPOConfig:
    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str         = "EleutherAI/gpt-neo-125m"
    dataset_train_path: str = "/mnt/user-data/outputs/grpo_dataset/train"
    dataset_val_path: str   = "/mnt/user-data/outputs/grpo_dataset/val"
    output_dir: str         = "grpo_checkpoints"

    # ── GRPO core (maps directly to paper notation) ───────────────────────────
    group_size: int         = 4      # G  — completions sampled per prompt
    beta: float             = 0.04   # β  — KL penalty coefficient
    clip_eps: float         = 0.2    # ε  — PPO-style ratio clip

    # ── Generation ────────────────────────────────────────────────────────────
    max_prompt_len: int     = 96     # truncate prompts longer than this
    max_new_tokens: int     = 64     # max tokens the model may generate
    temperature: float      = 0.9   # sampling temperature during rollout
    top_p: float            = 0.95  # nucleus sampling cutoff

    # ── Training ──────────────────────────────────────────────────────────────
    lr: float               = 1e-5
    num_epochs: int         = 3
    batch_size: int         = 2      # prompts per step (completions = batch * G)
    grad_clip: float        = 1.0
    warmup_steps: int       = 50

    # ── Logging / Saving ──────────────────────────────────────────────────────
    log_every: int          = 10     # steps between console logs
    save_every: int         = 200    # steps between checkpoints
    val_every: int          = 100    # steps between validation runs
    val_batches: int        = 20     # number of validation batches to evaluate

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int               = 42
    device: str             = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class AdditionDataset(Dataset):
    """Thin wrapper around the HF dataset saved by create_dataset.py."""

    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "prompt":   row["prompt"],
            "numbers":  row["numbers"],   # list[int] — ground-truth operands
            "answer":   row["answer"],    # int       — ground-truth sum
        }


def collate_fn(batch: list[dict]) -> dict:
    """Keep strings as lists; tensors are built inside the training step."""
    return {
        "prompts":  [ex["prompt"]  for ex in batch],
        "numbers":  [ex["numbers"] for ex in batch],
        "answers":  [ex["answer"]  for ex in batch],
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# Each function does exactly one thing and documents its tensor shapes.
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Sample completions ─────────────────────────────────────────────────────

def sample_completions(
    model,
    tokenizer,
    prompts: list[str],         # len = batch_size
    cfg: GRPOConfig,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """
    Generate G completions per prompt using ancestral sampling.

    Left-padding is required here: the model's first generated token is
    conditioned on the last real token of the prompt. Right-padding would
    place pad tokens there instead.

    Returns
    ───────
    completion_texts : list[str]   len = batch_size * G
    input_ids        : Tensor      [batch_size * G, prompt_len]
    attention_mask   : Tensor      [batch_size * G, prompt_len]
    """
    # Each prompt is repeated G times so one .generate() call covers all groups
    repeated_prompts = [p for p in prompts for _ in range(cfg.group_size)]
    # len(repeated_prompts) == batch_size * G

    # ── Left-pad for generation ───────────────────────────────────────────────
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        repeated_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_prompt_len,
    ).to(cfg.device)
    # encoded.input_ids      : [batch*G, prompt_len]
    # encoded.attention_mask : [batch*G, prompt_len]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        # output_ids : [batch*G, prompt_len + completion_len]

    # Slice off the prompt tokens to get completion-only ids
    prompt_len = encoded.input_ids.shape[1]
    completion_ids = output_ids[:, prompt_len:]
    # completion_ids : [batch*G, completion_len]

    completion_texts = tokenizer.batch_decode(
        completion_ids,
        skip_special_tokens=True,
    )
    # completion_texts : list[str], len = batch*G

    return completion_texts, encoded.input_ids, encoded.attention_mask


# ── 2. Compute rewards ────────────────────────────────────────────────────────

def compute_rewards(
    completion_texts: list[str],  # len = batch*G
    numbers_batch: list[list[int]],  # len = batch  (one per prompt)
    answers_batch: list[int],        # len = batch  (one per prompt)
    cfg: GRPOConfig,
) -> torch.Tensor:
    """
    Apply all three reward functions and sum them per completion.

    Each prompt's ground-truth is replicated G times to align with completions.

    Returns
    ───────
    rewards : Tensor  [batch*G]
    """
    # Replicate ground-truth to match the G completions per prompt
    numbers_rep = [n for n in numbers_batch for _ in range(cfg.group_size)]
    answers_rep = [a for a in answers_batch for _ in range(cfg.group_size)]
    # len(numbers_rep) == len(answers_rep) == batch*G

    kwargs = {"numbers": numbers_rep, "answer": answers_rep}

    r_think_tags    = reward_think_tags(completion_texts,    **kwargs)
    r_think_content = reward_think_content(completion_texts, **kwargs)
    r_answer        = reward_answer(completion_texts,        **kwargs)
    # each: list[float], len = batch*G

    rewards = torch.tensor(
        [t + c + a for t, c, a in zip(r_think_tags, r_think_content, r_answer)],
        dtype=torch.float32,
        device=cfg.device,
    )
    # rewards : [batch*G]

    return rewards


# ── 3. Compute advantages ─────────────────────────────────────────────────────

def compute_advantages(
    rewards: torch.Tensor,  # [batch*G]
    cfg: GRPOConfig,
) -> torch.Tensor:
    """
    GRPO advantage: normalise rewards within each group of G completions.

        A_i = (r_i - mean(r_group)) / (std(r_group) + ε)

    This is the key GRPO insight: no value network is needed. The baseline
    is simply the mean reward of the other completions in the same group.

    Returns
    ───────
    advantages : Tensor  [batch*G]
    """
    # Reshape to expose the group dimension for per-group normalisation
    rewards_grouped = rewards.view(-1, cfg.group_size)
    # rewards_grouped : [batch, G]

    mean = rewards_grouped.mean(dim=1, keepdim=True)   # [batch, 1]
    std  = rewards_grouped.std(dim=1, keepdim=True)    # [batch, 1]

    advantages_grouped = (rewards_grouped - mean) / (std + 1e-8)
    # advantages_grouped : [batch, G]

    advantages = advantages_grouped.view(-1)
    # advantages : [batch*G]

    return advantages


# ── 4. Get per-token log-probabilities ────────────────────────────────────────

def get_log_probs(
    model,
    tokenizer,
    prompts: list[str],           # len = batch*G  (repeated prompts)
    completions: list[str],       # len = batch*G
    cfg: GRPOConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a forward pass to get per-token log-probs for the completion tokens.

    Right-padding is required here: we need real tokens at consistent position
    indices for positional embeddings to be meaningful. Padding goes on the
    right where the loss mask will zero it out anyway.

    Returns
    ───────
    log_probs : Tensor  [batch*G, completion_len]  — log π(token | context)
    mask      : Tensor  [batch*G, completion_len]  — 1 for real, 0 for pad
    """
    # Concatenate prompt + completion so the model sees full context
    full_texts = [p + c for p, c in zip(prompts, completions)]

    # ── Right-pad for forward pass ────────────────────────────────────────────
    tokenizer.padding_side = "right"
    encoded = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_prompt_len + cfg.max_new_tokens,
    ).to(cfg.device)
    # encoded.input_ids : [batch*G, total_len]

    # We need prompt lengths to know where completions start
    tokenizer.padding_side = "right"
    prompt_encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_prompt_len,
    ).to(cfg.device)
    # prompt lengths vary; we take the non-padded length per row
    prompt_lens = encoded.attention_mask.sum(dim=1) - (
        tokenizer(completions, return_tensors="pt", padding=True,
                  truncation=True, max_length=cfg.max_new_tokens)
        .attention_mask.to(cfg.device).sum(dim=1)
    )
    # prompt_lens : [batch*G]  — number of real prompt tokens per row

    logits = model(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
    ).logits
    # logits : [batch*G, total_len, vocab_size]

    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]          # [batch*G, total_len-1, vocab]
    shift_labels = encoded.input_ids[:, 1:]   # [batch*G, total_len-1]

    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    # log_probs_all : [batch*G, total_len-1, vocab]

    # Gather the log-prob of the actual token at each position
    token_log_probs = log_probs_all.gather(
        dim=2,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)
    # token_log_probs : [batch*G, total_len-1]

    # Build completion mask: 1 only for completion tokens (not prompt, not pad)
    total_len = encoded.input_ids.shape[1]
    position_ids = torch.arange(total_len - 1, device=cfg.device).unsqueeze(0)
    # position_ids : [1, total_len-1]

    completion_mask = (
        (position_ids >= prompt_lens.unsqueeze(1) - 1) &
        (encoded.attention_mask[:, 1:].bool())
    ).float()
    # completion_mask : [batch*G, total_len-1]
    # 1.0 for positions that are (a) past the prompt and (b) not padding

    # Trim both to max_new_tokens for consistent downstream shapes
    log_probs = token_log_probs[:, -cfg.max_new_tokens:]
    mask      = completion_mask[:, -cfg.max_new_tokens:]
    # log_probs : [batch*G, completion_len]
    # mask      : [batch*G, completion_len]

    return log_probs, mask


# ── 5. Compute K3 reverse KL ──────────────────────────────────────────────────

def compute_kl_k3(
    log_probs_policy: torch.Tensor,  # [batch*G, completion_len]
    log_probs_ref: torch.Tensor,     # [batch*G, completion_len]
) -> torch.Tensor:
    """
    K3 estimator of the reverse KL divergence D_KL(π_ref || π_θ).

    As proposed by Schulman and used in DeepSeek GRPO:

        D_KL ≈ 0.5 * (π_ref/π_θ  -  log(π_ref/π_θ)  -  1)

    Properties
    ──────────
    • Unbiased: computed from sampled tokens only, no full softmax needed
    • Non-negative: guaranteed by x - log(x) - 1 ≥ 0  for all x > 0
    • Conservative: over-penalises large drift, preventing policy collapse

    Returns
    ───────
    kl : Tensor  [batch*G, completion_len]  — per-token KL estimate
    """
    # log(π_ref / π_θ) = log_probs_ref - log_probs_policy
    log_ratio = log_probs_ref - log_probs_policy
    # log_ratio : [batch*G, completion_len]

    ratio = torch.exp(log_ratio)
    # ratio : [batch*G, completion_len]  — π_ref(token) / π_θ(token)

    kl = 0.5 * (ratio - log_ratio - 1)
    # kl : [batch*G, completion_len]

    return kl


# ── 6. Compute GRPO loss ──────────────────────────────────────────────────────

def compute_loss(
    log_probs_policy: torch.Tensor,  # [batch*G, completion_len]
    log_probs_old: torch.Tensor,     # [batch*G, completion_len]  — from rollout
    log_probs_ref: torch.Tensor,     # [batch*G, completion_len]  — frozen ref
    advantages: torch.Tensor,        # [batch*G]
    mask: torch.Tensor,              # [batch*G, completion_len]
    cfg: GRPOConfig,
) -> tuple[torch.Tensor, dict]:
    """
    GRPO objective (equation 3 in the DeepSeekMath paper):

        L = -E[ min(r·A, clip(r, 1±ε)·A) ] + β · D_KL(π_ref || π_θ)

    where  r = π_θ(token) / π_old(token)  is the per-token probability ratio.

    Returns
    ───────
    loss    : scalar Tensor
    metrics : dict of float — for logging
    """
    # ── Probability ratio ─────────────────────────────────────────────────────
    log_ratio = log_probs_policy - log_probs_old
    ratio = torch.exp(log_ratio)
    # ratio : [batch*G, completion_len]

    # ── Clipped surrogate objective ───────────────────────────────────────────
    A = advantages.unsqueeze(1)
    # A : [batch*G, 1]  — broadcast over completion tokens

    surrogate_unclipped = ratio * A
    surrogate_clipped   = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * A
    # both : [batch*G, completion_len]

    policy_loss = -torch.min(surrogate_unclipped, surrogate_clipped)
    # policy_loss : [batch*G, completion_len]  — negative because we minimise

    # ── K3 reverse KL penalty ─────────────────────────────────────────────────
    kl_k3 = compute_kl_k3(log_probs_policy, log_probs_ref)
    # kl_k3 : [batch*G, completion_len]

    # ── Combine and mask ──────────────────────────────────────────────────────
    total_per_token = policy_loss + cfg.beta * kl_k3
    # total_per_token : [batch*G, completion_len]

    # Average over real (non-pad, non-prompt) tokens only
    loss = (total_per_token * mask).sum() / mask.sum().clamp(min=1)
    # loss : scalar

    # ── Metrics for logging ───────────────────────────────────────────────────
    with torch.no_grad():
        metrics = {
            "loss":         loss.item(),
            "policy_loss":  ((policy_loss * mask).sum() / mask.sum()).item(),
            "kl":           ((kl_k3 * mask).sum() / mask.sum()).item(),
            "mean_ratio":   ((ratio * mask).sum() / mask.sum()).item(),
            "clip_frac":    (((ratio - 1).abs() > cfg.clip_eps).float() * mask
                             ).sum().item() / mask.sum().item(),
        }

    return loss, metrics


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN STEP
# Reads like the GRPO algorithm pseudocode.
# ══════════════════════════════════════════════════════════════════════════════

def train_step(
    policy_model,
    ref_model,
    tokenizer,
    optimizer,
    batch: dict,
    cfg: GRPOConfig,
) -> dict:
    prompts  = batch["prompts"]   # list[str], len = batch_size
    numbers  = batch["numbers"]   # list[list[int]], len = batch_size
    answers  = batch["answers"]   # list[int], len = batch_size

    # ── 1. SAMPLE COMPLETIONS ─────────────────────────────────────────────────
    # Generate G completions per prompt under the current policy.
    # Left-pad so the model's context ends flush with the first generated token.
    policy_model.eval()
    with torch.no_grad():
        completion_texts, _, _ = sample_completions(
            policy_model, tokenizer, prompts, cfg,
        )
    # completion_texts : list[str], len = batch*G

    # ── 2. COMPUTE REWARDS ────────────────────────────────────────────────────
    # Score each completion with our three reward functions.
    rewards = compute_rewards(completion_texts, numbers, answers, cfg)
    # rewards : [batch*G]

    # ── 3. COMPUTE ADVANTAGES ─────────────────────────────────────────────────
    # Normalise within each group of G — no value network needed.
    advantages = compute_advantages(rewards, cfg)
    # advantages : [batch*G]

    # ── 4. FORWARD PASS — policy + reference ──────────────────────────────────
    # Right-pad for the forward pass to keep position indices consistent.
    # We run both models on the same (prompt + completion) sequences.
    policy_model.train()

    repeated_prompts = [p for p in prompts for _ in range(cfg.group_size)]
    # repeated_prompts : list[str], len = batch*G

    log_probs_policy, mask = get_log_probs(
        policy_model, tokenizer, repeated_prompts, completion_texts, cfg,
    )
    # log_probs_policy : [batch*G, completion_len]
    # mask             : [batch*G, completion_len]

    # Free policy activations before the ref forward pass — both would
    # otherwise sit in VRAM simultaneously, doubling activation memory.
    torch.cuda.empty_cache()

    with torch.no_grad():
        log_probs_ref, _ = get_log_probs(
            ref_model, tokenizer, repeated_prompts, completion_texts, cfg,
        )
        # log_probs_ref : [batch*G, completion_len]

        # "Old" log-probs come from the policy *before* this gradient step.
        # Since we sample immediately before updating, policy ≈ old-policy,
        # but we record them explicitly to keep the PPO ratio well-defined.
        log_probs_old = log_probs_policy.detach()
        # log_probs_old : [batch*G, completion_len]

    # ── 5. COMPUTE LOSS ───────────────────────────────────────────────────────
    loss, metrics = compute_loss(
        log_probs_policy,
        log_probs_old,
        log_probs_ref,
        advantages,
        mask,
        cfg,
    )
    # loss : scalar

    # ── 6. BACKWARD + UPDATE ──────────────────────────────────────────────────
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), cfg.grad_clip)
    optimizer.step()

    # Attach reward stats for logging
    metrics["mean_reward"] = rewards.mean().item()
    metrics["std_reward"]  = rewards.std().item()

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    policy_model,
    ref_model,
    tokenizer,
    val_loader: DataLoader,
    cfg: GRPOConfig,
) -> dict:
    """
    Run a few batches of validation and return mean metrics.
    Uses greedy decoding to get the model's best guess answer.
    """
    policy_model.eval()

    total_correct = 0
    total_samples = 0
    total_reward  = 0.0

    for i, batch in enumerate(val_loader):
        if i >= cfg.val_batches:
            break

        prompts = batch["prompts"]
        numbers = batch["numbers"]
        answers = batch["answers"]

        # Greedy decode — one completion per prompt for val efficiency
        tokenizer.padding_side = "left"
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_len,
        ).to(cfg.device)

        output_ids = policy_model.generate(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,   # greedy
            pad_token_id=tokenizer.pad_token_id,
        )

        prompt_len = encoded.input_ids.shape[1]
        completion_ids = output_ids[:, prompt_len:]
        completions = tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True,
        )

        # Compute rewards (group_size=1 for val)
        rewards = compute_rewards(completions, numbers, answers,
                                  GRPOConfig(group_size=1))

        # Count exact-correct answers
        import re
        for comp, ans in zip(completions, answers):
            m = re.search(r"<answer>(\d+)</answer>", comp)
            if m and int(m.group(1)) == ans:
                total_correct += 1

        total_samples += len(prompts)
        total_reward  += rewards.sum().item()

    return {
        "val_accuracy":    total_correct / max(total_samples, 1),
        "val_mean_reward": total_reward  / max(total_samples, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(cfg: GRPOConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # OPT has no dedicated pad token — setting pad=eos is the common fix but
    # causes ambiguity: the model sees </s> as both padding and end-of-sequence.
    # Instead we add a clean <pad> token that carries no semantic meaning.
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # ── Load policy model ─────────────────────────────────────────────────────
    policy_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=torch.float32,
    ).to(cfg.device)

    # ── Load frozen reference model ───────────────────────────────────────────
    # The reference stays at the initial weights throughout training.
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=torch.float32,
    ).to(cfg.device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    # ── Resize embedding tables ──────────────────────────────────────────────
    # OPT's checkpoint already stores embed_tokens and lm_head as independent
    # tensors — no tying is in effect. We only need to grow both tables by one
    # row for the new <pad> token. Both models must be resized to stay in sync.
    policy_model.resize_token_embeddings(len(tokenizer))
    ref_model.resize_token_embeddings(len(tokenizer))

    # Gradient checkpointing trades ~20% speed for ~50% activation memory.
    # Activations are recomputed on the backward pass instead of stored.
    policy_model.gradient_checkpointing_enable()

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_hf = load_from_disk(cfg.dataset_train_path)
    val_hf   = load_from_disk(cfg.dataset_val_path)

    train_ds = AdditionDataset(train_hf)
    val_ds   = AdditionDataset(val_hf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn,
    )

    # ── Optimizer + linear warmup scheduler ──────────────────────────────────
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=cfg.lr)

    total_steps = len(train_loader) * cfg.num_epochs
    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        # linear decay after warmup
        progress = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ──────────────────────────────────────────────────────────────
    global_step = 0
    log_buffer: list[dict] = []

    print(f"\n{'═'*60}")
    print(f"  GRPO Training — {cfg.model_name}")
    print(f"  Device     : {cfg.device}")
    print(f"  Train size : {len(train_ds):,}  |  Val size: {len(val_ds):,}")
    print(f"  Steps/epoch: {len(train_loader):,}  |  Total: {total_steps:,}")
    print(f"  Group size : {cfg.group_size}  |  β={cfg.beta}  |  ε={cfg.clip_eps}")
    print(f"{'═'*60}\n")

    for epoch in range(1, cfg.num_epochs + 1):
        for batch in train_loader:
            global_step += 1

            metrics = train_step(
                policy_model, ref_model, tokenizer, optimizer, batch, cfg,
            )
            scheduler.step()
            metrics["lr"] = scheduler.get_last_lr()[0]
            log_buffer.append(metrics)

            # ── Console log ───────────────────────────────────────────────────
            if global_step % cfg.log_every == 0:
                avg = {k: sum(d[k] for d in log_buffer) / len(log_buffer)
                       for k in log_buffer[0]}
                print(
                    f"Ep {epoch:02d} | Step {global_step:05d} | "
                    f"loss={avg['loss']:.4f} | "
                    f"reward={avg['mean_reward']:.3f} ± {avg['std_reward']:.3f} | "
                    f"kl={avg['kl']:.4f} | "
                    f"clip%={avg['clip_frac']*100:.1f} | "
                    f"lr={avg['lr']:.2e}"
                )
                log_buffer.clear()

            # ── Validation ────────────────────────────────────────────────────
            if global_step % cfg.val_every == 0:
                val_metrics = validate(
                    policy_model, ref_model, tokenizer, val_loader, cfg,
                )
                print(
                    f"  ── VAL @ step {global_step} | "
                    f"acc={val_metrics['val_accuracy']:.3f} | "
                    f"reward={val_metrics['val_mean_reward']:.3f}"
                )
                policy_model.train()

            # ── Checkpoint ────────────────────────────────────────────────────
            if global_step % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.output_dir, f"step_{global_step:05d}")
                policy_model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                print(f"  ── Saved checkpoint → {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(cfg.output_dir, "final")
    policy_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nTraining complete. Final model saved → {final_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = GRPOConfig()
    train(cfg)
