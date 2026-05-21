"""
Transfer Learning Trainer for Feature 2.
Handles linear probe, fine-tuning, and scratch training.
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.metrics import compute_roc_auc


# ─────────────────────────────────────────────
# Transfer Early Stopping
# ─────────────────────────────────────────────

class TransferEarlyStopping:
    """Early stopping for transfer learning (same logic as Feature 1)."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_state = None
        self.ckpt_path = None

    def step(
        self,
        score: float,
        model: nn.Module,
        ckpt_path: Optional[str] = None,
    ) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            if ckpt_path is not None:
                os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                self.ckpt_path = ckpt_path
                self.best_state = None
            else:
                self.best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module, device: Optional[torch.device] = None):
        """Restore model to best checkpoint."""
        if self.ckpt_path is not None:
            if device is None:
                device = torch.device("cpu")
            model.load_state_dict(torch.load(self.ckpt_path, map_location=device))
        elif self.best_state is not None:
            model.load_state_dict(self.best_state)


# ─────────────────────────────────────────────
# Transfer Trainer
# ─────────────────────────────────────────────

class TransferTrainer:
    """
    Unified trainer for all transfer learning strategies:
    - linear_probe
    - finetune (top_layers / full)
    - scratch

    Handles:
    - Per-epoch training loop
    - Validation ROC-AUC computation
    - Early stopping
    - Checkpoint saving
    - Result logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        test_dataset,
        task_names: List[str],
        config: Dict,
        checkpoint_dir: str = "checkpoints/feature2",
        result_dir: str = "results/feature2/transfer_baselines",
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        if config.get("compile_model", False) and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self._checkpoint_model = getattr(self.model, "_orig_mod", self.model)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.verbose = verbose

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        # Training config
        self.lr = config.get("lr", 1e-3)
        self.weight_decay = config.get("weight_decay", 1e-5)
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 100)
        self.patience = config.get("patience", 20)
        self.grad_clip = config.get("grad_clip", 1.0)

        # Optimizer (only trainable params)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            trainable_params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=10
        )
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Early stopping
        self.early_stopping = TransferEarlyStopping(patience=self.patience)

        # History
        self.history = {
            "train_loss": [],
            "val_auc": [],
            "lr": [],
        }

    def _build_loader(self, dataset, shuffle: bool = False) -> PyGDataLoader:
        available_cpus = os.cpu_count() or 1
        # Cap at 4 workers to improve host-to-device throughput without
        # oversubscribing typical single-GPU training pods.
        num_workers = min(4, available_cpus)
        return PyGDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    def _train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss = self.model.compute_loss(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(self, loader) -> Tuple[float, Dict[str, float]]:
        """
        Compute ROC-AUC on a dataset split.

        Returns:
            (mean_auc, per_task_auc_dict)
        """
        self.model.eval()

        all_probs = []
        all_labels = []

        for batch in loader:
            batch = batch.to(self.device)
            probs = self.model.predict(batch)   # [B, num_tasks]
            labels = batch.y                    # [B, num_tasks]

            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            if probs.shape != labels.shape:
                labels = labels.view(probs.shape)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Per-task ROC-AUC
        per_task_auc = {}
        valid_aucs = []

        for t, name in enumerate(self.task_names):
            task_labels = all_labels[:, t] if all_labels.ndim > 1 else all_labels
            task_probs = all_probs[:, t] if all_probs.ndim > 1 else all_probs

            # Skip tasks with missing labels or single class
            mask = task_labels != -1
            if mask.sum() < 10:
                continue
            tl = task_labels[mask]
            tp = task_probs[mask]
            if len(np.unique(tl)) < 2:
                continue

            auc = compute_roc_auc(tl, tp)
            if not np.isnan(auc):
                per_task_auc[name] = auc
                valid_aucs.append(auc)

        mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.5
        return mean_auc, per_task_auc

    def train(
        self,
        experiment_name: str = "transfer_exp",
        save_checkpoint: bool = True,
    ) -> Dict:
        """
        Full training loop.

        Returns:
            results dict with train/val/test metrics
        """
        train_loader = self._build_loader(self.train_dataset, shuffle=True)
        val_loader = self._build_loader(self.val_dataset, shuffle=False)
        test_loader = self._build_loader(self.test_dataset, shuffle=False)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Transfer Training: {experiment_name}")
            print(f"  Train: {len(self.train_dataset)}, "
                  f"Val: {len(self.val_dataset)}, "
                  f"Test: {len(self.test_dataset)}")
            print(f"  Tasks: {self.num_tasks}, Device: {self.device}")
            print(f"{'='*60}")

        start_time = time.time()
        ckpt_path = os.path.join(self.checkpoint_dir, f"{experiment_name}_best.pt")

        for epoch in range(self.epochs):
            # Train
            train_loss = self._train_epoch(train_loader)

            # Validate
            val_auc, val_per_task = self._evaluate(val_loader)

            # Scheduler step
            self.scheduler.step(val_auc)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_auc"].append(val_auc)
            self.history["lr"].append(current_lr)

            if self.verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{self.epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Val AUC: {val_auc:.4f} | "
                      f"LR: {current_lr:.2e}")

            # Early stopping
            if self.early_stopping.step(
                val_auc,
                self._checkpoint_model,
                ckpt_path if save_checkpoint else None,
            ):
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        # Restore best model
        self.early_stopping.restore_best(self._checkpoint_model, self.device)

        # Final test evaluation
        test_auc, test_per_task = self._evaluate(test_loader)
        val_auc, val_per_task = self._evaluate(val_loader)

        elapsed = time.time() - start_time

        results = {
            "experiment_name": experiment_name,
            "val_auc": val_auc,
            "test_auc": test_auc,
            "val_per_task_auc": val_per_task,
            "test_per_task_auc": test_per_task,
            "best_val_auc": self.early_stopping.best_score,
            "history": self.history,
            "config": self.config,
            "elapsed_seconds": elapsed,
            "n_train": len(self.train_dataset),
            "n_val": len(self.val_dataset),
            "n_test": len(self.test_dataset),
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Final Results: {experiment_name}")
            print(f"  Val AUC:  {val_auc:.4f}")
            print(f"  Test AUC: {test_auc:.4f}")
            print(f"  Time: {elapsed:.1f}s")
            print(f"{'='*60}\n")

        # Save results
        result_path = os.path.join(
            self.result_dir, f"{experiment_name}_results.json"
        )
        with open(result_path, "w") as f:
            # Convert non-serializable objects
            json_results = {
                k: v for k, v in results.items()
                if k != "history"
            }
            json_results["history"] = {
                k: [float(x) for x in v]
                for k, v in self.history.items()
            }
            json.dump(json_results, f, indent=2)

        if self.verbose:
            print(f"  Results saved to {result_path}")

        return results
