"""
Training loop for the AttnRes LM.

Features:
  - Cosine learning rate schedule with warmup
  - Mixed precision (bf16/fp16) training
  - Gradient accumulation for effective large batch sizes
  - Gradient clipping for stability
  - Periodic checkpointing
  - WandB logging (optional)
  - MoE auxiliary loss integration
  - Memory-efficient: gradient checkpointing support

Designed to work on a single GPU (T4 16GB) for the prototype, with
clear upgrade paths to multi-GPU via accelerate/DeepSpeed.
"""

from __future__ import annotations

import json
import os
import time
import math
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from ..model import AttnResLM, AttnResConfig


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 10000
    lr_scheduler_type: str = "cosine"  # "cosine" or "linear"

    # Batch
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch = batch_size * grad_accum

    # Precision
    use_amp: bool = True  # mixed precision training
    amp_dtype: str = "bf16"  # "bf16" or "fp16"

    # Logging
    log_interval: int = 10  # steps between console logs
    eval_interval: int = 500  # steps between evaluations
    save_interval: int = 1000  # steps between checkpoint saves
    use_wandb: bool = False
    wandb_project: str = "attnres-lm"

    # Checkpointing
    output_dir: str = "./checkpoints"
    save_total_limit: int = 3  # max checkpoints to keep

    # MoE
    moe_aux_loss_weight: float = 0.01


class Trainer:
    """
    Training loop for the AttnRes LM.

    Handles optimization, scheduling, logging, and checkpointing.
    Designed to be simple and self-contained for the prototype.
    """

    def __init__(
        self,
        model: AttnResLM,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Mixed precision
        self.amp_dtype = (
            torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
        )
        self.use_amp = config.use_amp and self.device.type == "cuda"
        # GradScaler only needed for fp16, not bf16
        self.scaler = GradScaler() if (self.use_amp and config.amp_dtype == "fp16") else None

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Learning rate scheduler
        self.scheduler = self._build_scheduler()

        # WandB
        self.wandb_run = None
        if config.use_wandb:
            import wandb
            self.wandb_run = wandb.init(
                project=config.wandb_project,
                config={**vars(config), **vars(model.config)},
            )

        # State
        self.global_step = 0
        self.best_eval_loss = float("inf")        # Per-step loss history for the notebook's loss-curve cell. Each entry
        # is {"step": int, "loss": float, "lr": float, "tokens_per_sec": float}.
        # Only rank-0 writes the final losses.json sidecar (in save_checkpoint
        # + at the bottom of train()).
        self.loss_history: List[Dict[str, float]] = []


    def _inner_model(self) -> "AttnResLM":
        """Return the underlying AttnResLM, unwrapping DistributedDataParallel.

        DDP only proxies `forward`, `parameters`, `named_parameters`, `train`,
        `eval`, `to`, etc. — every AttnResLM-specific attribute (num_parameters,
        save_pretrained, the model's own `config` dataclass, etc.) needs to be
        read on `self.model.module`. Centralising the unwrap here keeps callers
        one-liner clean.
        """
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW optimizer with separate weight decay groups."""
        # No weight decay for norms, embeddings, and AttnRes pseudo-queries
        no_decay = ["norm", "embed", "attn_res", "router"]
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        return optimizer

    def _build_scheduler(self):
        """Build cosine learning rate scheduler with warmup."""
        warmup = self.config.warmup_steps
        max_steps = self.config.max_steps
        lr = self.config.learning_rate

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, max_steps - warmup)
            if self.config.lr_scheduler_type == "cosine":
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            else:  # linear
                return max(0.0, 1.0 - progress)

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """Main training loop."""
        self.model.train()
        accum_steps = self.config.gradient_accumulation_steps
        step_loss = 0.0
        step_aux_loss = 0.0
        data_iter = iter(self.train_dataloader)
        start_time = time.time()

        print(f"Starting training: {self.config.max_steps} steps, "
              f"effective batch size = {self.config.batch_size * accum_steps}")
        print(f"Device: {self.device}, AMP: {self.use_amp} ({self.config.amp_dtype})")
        print(f"Model parameters: {self._inner_model().num_parameters() / 1e6:.1f}M")

        for step in range(self.config.max_steps):
            # ── Forward + Backward with gradient accumulation ────
            self.optimizer.zero_grad()

            for _ in range(accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass with optional mixed precision
                if self.use_amp:
                    with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        output = self.model(input_ids, labels=labels)
                        loss = output.loss / accum_steps
                        if output.aux_loss is not None:
                            aux_loss = output.aux_loss * self.config.moe_aux_loss_weight / accum_steps
                            loss = loss + aux_loss
                            step_aux_loss += aux_loss.item() * accum_steps
                else:
                    output = self.model(input_ids, labels=labels)
                    loss = output.loss / accum_steps
                    if output.aux_loss is not None:
                        aux_loss = output.aux_loss * self.config.moe_aux_loss_weight / accum_steps
                        loss = loss + aux_loss
                        step_aux_loss += aux_loss.item() * accum_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                step_loss += output.loss.item()

            # ── Gradient clipping & optimizer step ───────────────
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1

            # ── Logging ──────────────────────────────────────────
            if self.global_step % self.config.log_interval == 0:
                avg_loss = step_loss / (self.config.log_interval * accum_steps)
                elapsed = time.time() - start_time
                tokens_per_sec = (
                    self.config.batch_size * accum_steps * input_ids.shape[1]
                    * self.config.log_interval / elapsed
                )
                lr = self.scheduler.get_last_lr()[0]

                aux_str = (
                    f" Aux: {step_aux_loss / (self.config.log_interval * accum_steps):.4f}"
                    if step_aux_loss > 0 else ""
                )
                print(
                    f"Step {self.global_step:>6}/{self.config.max_steps} | "
                    f"Loss: {avg_loss:.4f}"
                    f"{aux_str}"
                    f" | LR: {lr:.2e} | "
                    f"Tokens/s: {tokens_per_sec:.0f} | "
                    f"Time: {elapsed:.1f}s"
                )

                # Append to loss history for the notebook's loss-curve cell.
                # Rank-0 only — non-zero ranks stay quiet to avoid duplicate logs.
                if not (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0):
                    entry = {
                        "step": self.global_step,
                        "loss": avg_loss,
                        "lr": lr,
                        "tokens_per_sec": tokens_per_sec,
                    }
                    if step_aux_loss > 0:
                        entry["aux_loss"] = step_aux_loss / (
                            self.config.log_interval * accum_steps
                        )
                    self.loss_history.append(entry)

                if self.wandb_run:
                    self.wandb_run.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/step": self.global_step,
                    })

                step_loss = 0.0
                step_aux_loss = 0.0
                start_time = time.time()

            # ── Evaluation ───────────────────────────────────────
            if (
                self.eval_dataloader is not None
                and self.global_step % self.config.eval_interval == 0
            ):
                eval_loss = self.evaluate()
                print(f"  Eval loss: {eval_loss:.4f}")

                if self.wandb_run:
                    self.wandb_run.log({
                        "eval/loss": eval_loss,
                        "eval/step": self.global_step,
                    })

                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint("best")

            # ── Checkpointing ────────────────────────────────────
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}")

        # Final save — save_checkpoint now handles DDP rank-0 + unwrap.
        self.save_checkpoint("final")

        # Write per-step losses.json sidecar for the notebook loss-curve cell.
        # save_checkpoint already restricted itself to rank-0 on DDP, so this
        # only fires for rank-0 in DDP and once on single-GPU.
        self._write_losses_json()

        print(f"Training complete! Best eval loss: {self.best_eval_loss:.4f}")

        if self.wandb_run:
            self.wandb_run.finish()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on the eval dataset and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.use_amp:
                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    output = self.model(input_ids, labels=labels)
            else:
                output = self.model(input_ids, labels=labels)

            total_loss += output.loss.item()
            num_batches += 1

        self.model.train()
        return total_loss / max(1, num_batches)

    def save_checkpoint(self, name: str):
        """Save a model checkpoint.

        DDP-safe: returns early on non-zero ranks, and unwraps
        DistributedDataParallel before calling save_pretrained (which is a
        method on AttnResLM, not on DDP).
        """
        # Only rank 0 writes the checkpoint on DDP.
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        # Unwrap DDP wrapper if present.
        inner_model = (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )

        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, name)
        # Surface silent overwrites of `final` or `best` so the user is
        # aware when a re-run clobbers a previously saved checkpoint.
        if os.path.isdir(path):
            print(f"  NOTE: overwriting existing checkpoint at {path}")
        os.makedirs(path, exist_ok=True)

        # Save model (now guaranteed to be a bare AttnResLM with save_pretrained)
        inner_model.save_pretrained(path)

        # Save training state
        torch.save({
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_eval_loss": self.best_eval_loss,
        }, os.path.join(path, "training_state.pt"))

        print(f"  Saved checkpoint: {path}")

        # Clean up old checkpoints
        self._cleanup_checkpoints()
        return path

    def _write_losses_json(self):
        """Write per-step losses to <output_dir>/losses.json for the notebook."""
        # Only rank 0 writes on DDP. Single-GPU always writes.
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        if not self.loss_history:
            return  # Skip if no log step ran yet (sanity-only runs)
        os.makedirs(self.config.output_dir, exist_ok=True)
        losses_path = os.path.join(self.config.output_dir, "losses.json")
        with open(losses_path, "w") as f:
            json.dump({
                "config": {k: v for k, v in self.config.__dict__.items() if not k.startswith("_")},
                "step": self.loss_history,
            }, f, indent=2)
        print(f"  Wrote loss log: {losses_path} ({len(self.loss_history)} entries)")

    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        if self.config.save_total_limit <= 0:
            return

        ckpt_dir = self.config.output_dir
        if not os.path.exists(ckpt_dir):
            return

        # List all checkpoint dirs (exclude "best" and "final")
        ckpts = []
        for name in os.listdir(ckpt_dir):
            full_path = os.path.join(ckpt_dir, name)
            if os.path.isdir(full_path) and name.startswith("step_"):
                ckpts.append((name, full_path))

        # Sort by step number
        ckpts.sort(key=lambda x: int(x[0].split("_")[1]))

        # Remove oldest beyond limit
        while len(ckpts) > self.config.save_total_limit:
            name, path = ckpts.pop(0)
            import shutil
            shutil.rmtree(path)
            print(f"  Removed old checkpoint: {name}")
