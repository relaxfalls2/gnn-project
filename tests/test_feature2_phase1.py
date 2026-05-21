"""
Tests for Feature 2 Phase 1: Transfer Learning Infrastructure.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def mock_checkpoint_path(tmp_path):
    """Create a temporary mock checkpoint."""
    from feature2.models.pretrained_encoder import create_mock_checkpoint
    path = str(tmp_path / "mock_checkpoint.pt")
    create_mock_checkpoint(
        path, model_type="task_conditioned",
        hidden_dim=32, n_layers=2, task_dim=16, num_tasks=17
    )
    return path


@pytest.fixture
def small_transfer_dataset():
    """Create a small synthetic TransferDataset."""
    import torch
    from torch_geometric.data import Data

    class TinyDataset:
        def __init__(self, n=20, num_tasks=5):
            self.data_list = []
            for i in range(n):
                n_atoms = np.random.randint(5, 15)
                n_edges = n_atoms * 2
                data = Data(
                    x=torch.randn(n_atoms, 129),
                    edge_index=torch.randint(0, n_atoms, (2, n_edges)),
                    edge_attr=torch.randn(n_edges, 6),
                    pos=torch.randn(n_atoms, 3),
                    y=torch.randint(0, 2, (num_tasks,)).float(),
                )
                self.data_list.append(data)
            self.task_names = [f"task_{i}" for i in range(num_tasks)]
            self.num_tasks = num_tasks

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            return self.data_list[idx]

        def get_task_names(self):
            return self.task_names

    return TinyDataset(n=30, num_tasks=5)


# ─────────────────────────────────────────────
# Test 1: Dataset Info
# ─────────────────────────────────────────────

class TestTransferDatasetInfo:

    def test_sider_info_exists(self):
        from feature2.data.transfer_datasets import TRANSFER_DATASET_INFO
        assert "sider" in TRANSFER_DATASET_INFO
        info = TRANSFER_DATASET_INFO["sider"]
        assert info["num_tasks"] == 27
        assert len(info["task_names"]) == 27

    def test_muv_info_exists(self):
        from feature2.data.transfer_datasets import TRANSFER_DATASET_INFO
        assert "muv" in TRANSFER_DATASET_INFO
        info = TRANSFER_DATASET_INFO["muv"]
        assert info["num_tasks"] == 17
        assert len(info["task_names"]) == 17

    def test_get_transfer_task_names(self):
        from feature2.data.transfer_datasets import get_transfer_task_names
        names = get_transfer_task_names("sider")
        assert isinstance(names, list)
        assert len(names) == 27

    def test_get_num_transfer_tasks(self):
        from feature2.data.transfer_datasets import get_num_transfer_tasks
        assert get_num_transfer_tasks("sider") == 27
        assert get_num_transfer_tasks("muv") == 17

    def test_invalid_dataset_raises(self):
        from feature2.data.transfer_datasets import get_transfer_dataset_info
        with pytest.raises(ValueError):
            get_transfer_dataset_info("invalid_dataset")


# ─────────────────────────────────────────────
# Test 2: Low-Data Splits
# ─────────────────────────────────────────────

class TestLowDataSplits:

    def test_create_low_data_subset_fraction(self, small_transfer_dataset):
        from feature2.data.low_data_splits import create_low_data_subset
        subset = create_low_data_subset(small_transfer_dataset, fraction=0.5, seed=42)
        n_expected = max(32, int(0.5 * len(small_transfer_dataset)))
        n_expected = min(n_expected, len(small_transfer_dataset))
        assert len(subset) == n_expected

    def test_full_fraction_returns_all(self, small_transfer_dataset):
        from feature2.data.low_data_splits import create_low_data_subset
        subset = create_low_data_subset(small_transfer_dataset, fraction=1.0, seed=0)
        assert len(subset) == len(small_transfer_dataset)

    def test_create_all_splits_returns_dict(self, small_transfer_dataset):
        from feature2.data.low_data_splits import create_all_low_data_splits
        splits = create_all_low_data_splits(
            small_transfer_dataset,
            fractions=[1.0, 0.5, 0.25],
            seed=42,
        )
        assert set(splits.keys()) == {1.0, 0.5, 0.25}

    def test_smaller_fraction_gives_smaller_dataset(self, small_transfer_dataset):
        from feature2.data.low_data_splits import create_all_low_data_splits
        splits = create_all_low_data_splits(
            small_transfer_dataset,
            fractions=[1.0, 0.5, 0.25],
        )
        # Note: min size is 32, so not always strictly decreasing for tiny datasets
        assert len(splits[1.0]) >= len(splits[0.25])

    def test_reproducibility(self, small_transfer_dataset):
        from feature2.data.low_data_splits import create_low_data_subset
        from torch.utils.data import Subset
        s1 = create_low_data_subset(small_transfer_dataset, 0.5, seed=42)
        s2 = create_low_data_subset(small_transfer_dataset, 0.5, seed=42)
        if hasattr(s1, "indices") and hasattr(s2, "indices"):
            assert s1.indices == s2.indices

    def test_low_data_config(self):
        from feature2.data.low_data_splits import LowDataConfig
        cfg = LowDataConfig("sider", 0.5, 42, "linear_probe")
        assert cfg.fraction == 0.5
        d = cfg.to_dict()
        assert d["dataset_name"] == "sider"

    def test_generate_experiment_grid(self):
        from feature2.data.low_data_splits import generate_low_data_experiment_grid
        grid = generate_low_data_experiment_grid(
            datasets=["sider"],
            fractions=[1.0, 0.5],
            seeds=[0, 1],
            strategies=["scratch", "linear_probe"],
        )
        # 1 dataset × 2 fractions × 2 seeds × 2 strategies = 8
        assert len(grid) == 8


# ─────────────────────────────────────────────
# Test 3: Pretrained Encoder Loader
# ─────────────────────────────────────────────

class TestPretrainedEncoderLoader:

    def test_mock_checkpoint_creation(self, tmp_path):
        from feature2.models.pretrained_encoder import create_mock_checkpoint
        path = str(tmp_path / "mock.pt")
        create_mock_checkpoint(path, model_type="task_conditioned",
                               hidden_dim=32, n_layers=2)
        assert os.path.exists(path)
        ckpt = torch.load(path, map_location="cpu")
        assert "model_state_dict" in ckpt
        assert ckpt.get("is_mock", False)

    def test_load_mock_checkpoint(self, mock_checkpoint_path):
        from feature2.models.pretrained_encoder import load_feature1_checkpoint
        model = load_feature1_checkpoint(
            mock_checkpoint_path, "task_conditioned", "cpu", verbose=False
        )
        assert model is not None
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_frozen_encoder_creation(self, mock_checkpoint_path):
        from feature2.models.pretrained_encoder import (
            load_feature1_checkpoint, FrozenEncoder
        )
        model = load_feature1_checkpoint(
            mock_checkpoint_path, "task_conditioned", "cpu", verbose=False
        )
        encoder = FrozenEncoder(model, "task_conditioned")
        # All encoder params should be frozen
        for param in encoder.encoder.parameters():
            assert not param.requires_grad

    def test_frozen_encoder_output_dim(self, mock_checkpoint_path):
        from feature2.models.pretrained_encoder import (
            load_feature1_checkpoint, FrozenEncoder
        )
        model = load_feature1_checkpoint(
            mock_checkpoint_path, "task_conditioned", "cpu", verbose=False
        )
        encoder = FrozenEncoder(model, "task_conditioned")
        dim = encoder.get_output_dim()
        assert dim > 0

    def test_unfreeze_encoder(self, mock_checkpoint_path):
        from feature2.models.pretrained_encoder import (
            load_feature1_checkpoint, FrozenEncoder
        )
        model = load_feature1_checkpoint(
            mock_checkpoint_path, "task_conditioned", "cpu", verbose=False
        )
        encoder = FrozenEncoder(model, "task_conditioned")
        encoder.unfreeze_encoder()
        n_trainable = sum(
            p.numel() for p in encoder.encoder.parameters() if p.requires_grad
        )
        assert n_trainable > 0

    def test_missing_checkpoint_raises(self):
        from feature2.models.pretrained_encoder import load_feature1_checkpoint
        with pytest.raises(FileNotFoundError):
            load_feature1_checkpoint("/nonexistent/path.pt", "task_conditioned")


# ─────────────────────────────────────────────
# Test 4: Transfer Heads
# ─────────────────────────────────────────────

class TestTransferHeads:

    def _make_encoder_and_batch(self, mock_checkpoint_path):
        from feature2.models.pretrained_encoder import (
            load_feature1_checkpoint, FrozenEncoder
        )
        from torch_geometric.data import Data, Batch

        model = load_feature1_checkpoint(
            mock_checkpoint_path, "task_conditioned", "cpu", verbose=False
        )
        encoder = FrozenEncoder(model, "task_conditioned")

        # Create synthetic batch
        data_list = []
        for _ in range(4):
            n = 8
            data_list.append(Data(
                x=torch.randn(n, 129),
                edge_index=torch.randint(0, n, (2, n * 2)),
                edge_attr=torch.randn(n * 2, 6),
                pos=torch.randn(n, 3),
                y=torch.randint(0, 2, (5,)).float(),
            ))
        batch = Batch.from_data_list(data_list)
        return encoder, batch

    def test_linear_probe_forward(self, mock_checkpoint_path):
        from feature2.models.transfer_heads import LinearProbeClassifier
        encoder, batch = self._make_encoder_and_batch(mock_checkpoint_path)
        model = LinearProbeClassifier(encoder=encoder, num_tasks=5)
        logits = model.forward(batch)
        assert logits.shape == (4, 5)

    def test_linear_probe_loss(self, mock_checkpoint_path):
        from feature2.models.transfer_heads import LinearProbeClassifier
        encoder, batch = self._make_encoder_and_batch(mock_checkpoint_path)
        model = LinearProbeClassifier(encoder=encoder, num_tasks=5)
        loss = model.compute_loss(batch)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_linear_probe_predict(self, mock_checkpoint_path):
        from feature2.models.transfer_heads import LinearProbeClassifier
        encoder, batch = self._make_encoder_and_batch(mock_checkpoint_path)
        model = LinearProbeClassifier(encoder=encoder, num_tasks=5)
        probs = model.predict(batch)
        assert probs.shape == (4, 5)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_finetune_top_layers(self, mock_checkpoint_path):
        from feature2.models.transfer_heads import FineTuneClassifier
        from feature2.models.pretrained_encoder import (
            load_feature1_checkpoint, FrozenEncoder
        )
        from torch_geometric.data import Data, Batch

        model = load_feature1_checkpoint(
            mock_checkpoint_path, "task_conditioned", "cpu", verbose=False
        )
        encoder = FrozenEncoder(model, "task_conditioned")
        ft_model = FineTuneClassifier(
            encoder=encoder, num_tasks=5,
            strategy="top_layers", num_unfreeze_layers=1
        )
        n_trainable = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
        assert n_trainable > 0

    def test_scratch_classifier_forward(self):
        from feature2.models.transfer_heads import ScratchClassifier
        from torch_geometric.data import Data, Batch

        model = ScratchClassifier(
            node_dim=129, edge_dim=6, hidden_dim=32,
            n_layers=2, num_tasks=5
        )
        data_list = []
        for _ in range(4):
            n = 8
            data_list.append(Data(
                x=torch.randn(n, 129),
                edge_index=torch.randint(0, n, (2, 16)),
                edge_attr=torch.randn(16, 6),
                pos=torch.randn(n, 3),
                y=torch.randint(0, 2, (5,)).float(),
            ))
        batch = Batch.from_data_list(data_list)
        loss = model.compute_loss(batch)
        assert loss.item() > 0


# ─────────────────────────────────────────────
# Test 5: Transfer Trainer
# ─────────────────────────────────────────────

class TestTransferTrainer:

    def test_trainer_runs_one_epoch(self, mock_checkpoint_path, tmp_path):
        from feature2.models.pretrained_encoder import (
            load_feature1_checkpoint, FrozenEncoder
        )
        from feature2.models.transfer_heads import LinearProbeClassifier
        from feature2.training.transfer_trainer import TransferTrainer
        from torch_geometric.data import Data

        class TinyDS:
            def __init__(self, n=12, t=3):
                self.data_list = [
                    Data(
                        x=torch.randn(8, 129),
                        edge_index=torch.randint(0, 8, (2, 16)),
                        edge_attr=torch.randn(16, 6),
                        pos=torch.randn(8, 3),
                        y=torch.randint(0, 2, (t,)).float(),
                    ) for _ in range(n)
                ]
            def __len__(self): return len(self.data_list)
            def __getitem__(self, i): return self.data_list[i]

        model_f1 = load_feature1_checkpoint(
            mock_checkpoint_path, "task_conditioned", "cpu", verbose=False
        )
        encoder = FrozenEncoder(model_f1, "task_conditioned")
        model = LinearProbeClassifier(encoder=encoder, num_tasks=3)

        train_ds = TinyDS(12, 3)
        val_ds = TinyDS(6, 3)
        test_ds = TinyDS(6, 3)

        config = {"lr": 1e-3, "weight_decay": 0, "batch_size": 4,
                  "epochs": 2, "patience": 5, "grad_clip": 1.0}

        trainer = TransferTrainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            task_names=["t0", "t1", "t2"],
            config=config,
            checkpoint_dir=str(tmp_path),
            result_dir=str(tmp_path),
            device="cpu",
            verbose=False,
        )
        results = trainer.train(experiment_name="test_exp")
        assert "test_auc" in results
        assert 0.0 <= results["test_auc"] <= 1.0


# ─────────────────────────────────────────────
# Test 6: Transfer Metrics
# ─────────────────────────────────────────────

class TestTransferMetrics:

    def test_compute_transfer_gain_positive(self):
        from feature2.evaluation.transfer_metrics import compute_transfer_gain
        result = compute_transfer_gain(0.80, 0.70)
        assert result["absolute_gain"] == pytest.approx(0.10, abs=1e-5)
        assert result["is_positive"] is True

    def test_compute_transfer_gain_negative(self):
        from feature2.evaluation.transfer_metrics import compute_transfer_gain
        result = compute_transfer_gain(0.65, 0.70)
        assert result["is_positive"] is False

    def test_aggregate_results_across_seeds(self):
        from feature2.evaluation.transfer_metrics import aggregate_results_across_seeds
        results = [{"test_auc": 0.70}, {"test_auc": 0.72}, {"test_auc": 0.68}]
        agg = aggregate_results_across_seeds(results, "test_auc")
        assert agg["mean"] == pytest.approx(0.70, abs=1e-3)
        assert agg["n_seeds"] == 3

    def test_low_data_degradation(self):
        from feature2.evaluation.transfer_metrics import compute_low_data_degradation
        results = {1.0: 0.80, 0.5: 0.75, 0.25: 0.68, 0.10: 0.60}
        deg = compute_low_data_degradation(results)
        assert deg[1.0]["absolute_drop"] == pytest.approx(0.0, abs=1e-5)
        assert deg[0.10]["absolute_drop"] == pytest.approx(0.20, abs=1e-3)

    def test_build_comparison_table(self):
        from feature2.evaluation.transfer_metrics import build_comparison_table
        results = {
            "linear_probe_sider": {"val_auc": 0.72, "test_auc": 0.71},
            "scratch_sider": {"val_auc": 0.65, "test_auc": 0.64},
        }
        table = build_comparison_table(results)
        assert "linear_probe_sider" in table
        assert "scratch_sider" in table


class TestTransferTrainerOptimizations:

    def test_build_loader_uses_workers_and_pin_memory(self, mock_checkpoint_path, tmp_path):
        from feature2.models.pretrained_encoder import load_feature1_checkpoint, FrozenEncoder
        from feature2.models.transfer_heads import LinearProbeClassifier
        from feature2.training.transfer_trainer import TransferTrainer
        from torch_geometric.data import Data

        class TinyDS:
            def __init__(self, n=8, t=2):
                self.data_list = [
                    Data(
                        x=torch.randn(6, 129),
                        edge_index=torch.randint(0, 6, (2, 12)),
                        edge_attr=torch.randn(12, 6),
                        pos=torch.randn(6, 3),
                        y=torch.randint(0, 2, (t,)).float(),
                    ) for _ in range(n)
                ]
            def __len__(self): return len(self.data_list)
            def __getitem__(self, i): return self.data_list[i]

        model_f1 = load_feature1_checkpoint(
            mock_checkpoint_path, "task_conditioned", "cpu", verbose=False
        )
        encoder = FrozenEncoder(model_f1, "task_conditioned")
        model = LinearProbeClassifier(encoder=encoder, num_tasks=2)

        trainer = TransferTrainer(
            model=model,
            train_dataset=TinyDS(),
            val_dataset=TinyDS(),
            test_dataset=TinyDS(),
            task_names=["t0", "t1"],
            config={"batch_size": 4, "epochs": 2},
            checkpoint_dir=str(tmp_path),
            result_dir=str(tmp_path),
            device="cpu",
            verbose=False,
        )

        loader = trainer._build_loader(trainer.train_dataset, shuffle=True)
        assert loader.num_workers == min(4, os.cpu_count() or 1)
        assert loader.pin_memory == torch.cuda.is_available()
        assert loader.persistent_workers is True

    def test_transfer_early_stopping_can_restore_from_disk(self, tmp_path):
        from feature2.training.transfer_trainer import TransferEarlyStopping

        model = nn.Linear(3, 1)
        ckpt_path = tmp_path / "best.pt"
        es = TransferEarlyStopping(patience=2)

        with torch.no_grad():
            model.weight.fill_(1.5)
            model.bias.fill_(0.25)
        es.step(0.8, model, str(ckpt_path))

        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()

        es.restore_best(model, torch.device("cpu"))

        assert ckpt_path.exists()
        assert torch.allclose(model.weight, torch.full_like(model.weight, 1.5))
        assert torch.allclose(model.bias, torch.full_like(model.bias, 0.25))


# ─────────────────────────────────────────────
# Test 7: Embedding Extractor
# ─────────────────────────────────────────────

class TestEmbeddingExtractor:

    def test_extract_embeddings(self, mock_checkpoint_path):
        from feature2.models.pretrained_encoder import (
            load_feature1_checkpoint, FrozenEncoder
        )
        from feature2.evaluation.embedding_extractor import EmbeddingExtractor
        from torch_geometric.data import Data

        class TinyDS:
            def __init__(self):
                self.data_list = [
                    Data(
                        x=torch.randn(8, 129),
                        edge_index=torch.randint(0, 8, (2, 16)),
                        edge_attr=torch.randn(16, 6),
                        pos=torch.randn(8, 3),
                        y=torch.zeros(3),
                    ) for _ in range(10)
                ]
            def __len__(self): return len(self.data_list)
            def __getitem__(self, i): return self.data_list[i]

        model = load_feature1_checkpoint(
            mock_checkpoint_path, "task_conditioned", "cpu", verbose=False
        )
        encoder = FrozenEncoder(model, "task_conditioned")
        extractor = EmbeddingExtractor(encoder, device="cpu", batch_size=4)

        result = extractor.extract(TinyDS(), label="test")
        assert "embeddings" in result
        assert result["embeddings"].shape[0] == 10
        assert result["embedding_dim"] > 0

    def test_inter_dataset_similarity(self):
        from feature2.evaluation.embedding_extractor import compute_inter_dataset_similarity
        embeddings = {
            "sider": np.random.randn(20, 32),
            "muv": np.random.randn(15, 32),
        }
        sim_matrix, names = compute_inter_dataset_similarity(embeddings)
        assert sim_matrix.shape == (2, 2)
        # Diagonal should be 1.0 (self-similarity)
        assert abs(sim_matrix[0, 0] - 1.0) < 0.01
        assert abs(sim_matrix[1, 1] - 1.0) < 0.01
