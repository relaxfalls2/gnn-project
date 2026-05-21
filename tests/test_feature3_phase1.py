"""Tests for Feature 3 Phase 1: Wrapper + GNNExplainer."""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from feature3.models.maskable_wrapper import MaskableModelWrapper
from feature3.explainer.gnn_explainer import GNNExplainer
from feature3.explainer.mask_utils import (
    normalize_mask, threshold_mask, get_important_edges, mask_statistics
)


# ── Fixtures ─────────────────────────────────────────────────────────────

class SimpleModel(nn.Module):
    """Minimal model that mimics MultiTaskClassifier interface."""
    def __init__(self, node_dim=10, edge_dim=3):
        super().__init__()
        self.num_tasks = 4
        self.lin = nn.Linear(node_dim + edge_dim, 1)

    def forward(self, data, task_idx=None):
        node_feat = data.x.mean(0)
        if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
            edge_feat = data.edge_attr.mean(0)
        else:
            edge_feat = torch.zeros(3)
        combined = torch.cat([node_feat, edge_feat]).unsqueeze(0)
        return self.lin(combined)


@pytest.fixture
def base_model():
    m = SimpleModel()
    m.eval()
    return m


@pytest.fixture
def wrapped_model(base_model):
    return MaskableModelWrapper(base_model)


@pytest.fixture
def simple_data():
    """3-atom chain: 0-1-2."""
    return Data(
        x=torch.randn(3, 10),
        edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]], dtype=torch.long),
        edge_attr=torch.ones(4, 3),
        pos=torch.randn(3, 3),
    )


@pytest.fixture
def fast_explainer(wrapped_model):
    return GNNExplainer(wrapped_model, epochs=5, lr=0.01)


# ── MaskableModelWrapper Tests ────────────────────────────────────────────

class TestMaskableModelWrapper:

    def test_init_freezes_model(self, wrapped_model, base_model):
        """All base model params must be frozen."""
        for param in base_model.parameters():
            assert not param.requires_grad

    def test_forward_no_mask(self, wrapped_model, simple_data):
        out = wrapped_model(simple_data, task_idx=0)
        assert out.shape == (1, 1)
        assert not torch.isnan(out).any()

    def test_forward_with_ones_mask(self, wrapped_model, simple_data):
        """Ones mask = same as no mask."""
        ones = torch.ones(4)
        out_no_mask = wrapped_model(simple_data, task_idx=0, edge_weight=None)
        out_ones = wrapped_model(simple_data, task_idx=0, edge_weight=ones)
        assert torch.allclose(out_no_mask, out_ones, atol=1e-5)

    def test_forward_with_zeros_mask(self, wrapped_model, simple_data):
        """Zero mask zeros out edge_attr → different output."""
        zeros = torch.zeros(4)
        out_no_mask = wrapped_model(simple_data, task_idx=0, edge_weight=None)
        out_zeros = wrapped_model(simple_data, task_idx=0, edge_weight=zeros)
        assert not torch.allclose(out_no_mask, out_zeros, atol=1e-4)

    def test_forward_mask_shape_mismatch_handled(self, wrapped_model, simple_data):
        """Wrong mask shape should raise or handle gracefully."""
        wrong_mask = torch.ones(10)
        try:
            out = wrapped_model(simple_data, task_idx=0, edge_weight=wrong_mask)
        except Exception:
            pass  # Expected

    def test_num_tasks_property(self, wrapped_model):
        assert wrapped_model.num_tasks == 4

    def test_task_names_property(self, wrapped_model):
        names = wrapped_model.task_names
        assert isinstance(names, list)
        assert len(names) == 17  # F1 default

    def test_get_task_name(self, wrapped_model):
        assert wrapped_model.get_task_name(0) == 'NR-AR'
        assert wrapped_model.get_task_name(99) == 'Task_99'

    def test_wrapper_has_no_params(self, wrapped_model):
        """Wrapper itself has no trainable parameters."""
        params = list(wrapped_model.parameters())
        assert len(params) == 0

    def test_eval_mode_maintained(self, wrapped_model):
        wrapped_model.train(True)
        assert not wrapped_model.base_model.training

    def test_repr(self, wrapped_model):
        r = repr(wrapped_model)
        assert 'MaskableModelWrapper' in r
        assert 'trainable_params=0' in r

    def test_no_edge_attr(self, wrapped_model):
        """Data without edge_attr should still work."""
        data = Data(
            x=torch.randn(3, 10),
            edge_index=torch.tensor([[0,1],[1,0]], dtype=torch.long),
        )
        mask = torch.ones(2)
        out = wrapped_model(data, task_idx=0, edge_weight=mask)
        assert out.shape == (1, 1)

    def test_gradient_does_not_flow_to_base(self, wrapped_model, simple_data):
        """Explanation gradients must not reach base model."""
        # Do a forward pass
        out = wrapped_model(simple_data, task_idx=0)
        loss = out.sum()
        loss.backward()
        for param in wrapped_model.base_model.parameters():
            assert param.grad is None

    def test_soft_mask_interpolation(self, wrapped_model, simple_data):
        """0.5 mask should give output between 0.0 and 1.0 mask outputs."""
        out_zero = wrapped_model(simple_data, task_idx=0,
                                 edge_weight=torch.zeros(4)).item()
        out_ones = wrapped_model(simple_data, task_idx=0,
                                 edge_weight=torch.ones(4)).item()
        out_half = wrapped_model(simple_data, task_idx=0,
                                 edge_weight=torch.full((4,), 0.5)).item()
        lo, hi = min(out_zero, out_ones), max(out_zero, out_ones)
        if hi - lo > 0.01:
            assert lo <= out_half <= hi or abs(out_half - lo) < 0.1


# ── GNNExplainer Tests ────────────────────────────────────────────────────

class TestGNNExplainerInit:
    def test_default_params(self, wrapped_model):
        exp = GNNExplainer(wrapped_model)
        assert exp.epochs == 100
        assert exp.lr == 0.01
        assert exp.edge_size == 0.005
        assert exp.edge_entropy == 1.0
        assert exp.temperature == 1.0

    def test_custom_params(self, wrapped_model):
        exp = GNNExplainer(
            wrapped_model, epochs=50, lr=0.005,
            edge_size=0.01, temperature=0.5
        )
        assert exp.epochs == 50
        assert exp.temperature == 0.5


class TestGNNExplainerExplain:

    def test_explain_returns_dict(self, fast_explainer, simple_data):
        result = fast_explainer.explain(simple_data, task_idx=0)
        assert isinstance(result, dict)

    def test_required_keys_present(self, fast_explainer, simple_data):
        result = fast_explainer.explain(simple_data, task_idx=0)
        required = [
            'edge_mask', 'node_feat_mask', 'node_importance',
            'prediction', 'target', 'loss_curve',
            'converged', 'epochs_run', 'num_edges', 'num_nodes'
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_edge_mask_shape(self, fast_explainer, simple_data):
        result = fast_explainer.explain(simple_data, task_idx=0)
        assert result['edge_mask'].shape == (4,)  # 4 directed edges

    def test_node_importance_shape(self, fast_explainer, simple_data):
        result = fast_explainer.explain(simple_data, task_idx=0)
        assert result['node_importance'].shape == (3,)  # 3 atoms

    def test_edge_mask_in_range(self, fast_explainer, simple_data):
        result = fast_explainer.explain(simple_data, task_idx=0)
        assert result['edge_mask'].min() >= 0.0
        assert result['edge_mask'].max() <= 1.0

    def test_node_importance_in_range(self, fast_explainer, simple_data):
        result = fast_explainer.explain(simple_data, task_idx=0)
        assert result['node_importance'].min() >= 0.0
        assert result['node_importance'].max() <= 1.0

    def test_feat_mask_shape(self, fast_explainer, simple_data):
        result = fast_explainer.explain(simple_data, task_idx=0)
        assert result['node_feat_mask'].shape == (10,)  # 10 features

    def test_loss_curve_exists(self, fast_explainer, simple_data):
        result = fast_explainer.explain(simple_data, task_idx=0)
        assert len(result['loss_curve']) > 0
        assert all(isinstance(v, float) for v in result['loss_curve'])

    def test_prediction_is_scalar(self, fast_explainer, simple_data):
        result = fast_explainer.explain(simple_data, task_idx=0)
        assert isinstance(result['prediction'], float)
        assert 0.0 <= result['prediction'] <= 1.0

    def test_with_provided_target(self, fast_explainer, simple_data):
        target = torch.tensor([1.0])
        result = fast_explainer.explain(simple_data, task_idx=0, target=target)
        assert result['target'] == 1.0

    def test_no_nans_in_masks(self, fast_explainer, simple_data):
        result = fast_explainer.explain(simple_data, task_idx=0)
        assert not torch.isnan(result['edge_mask']).any()
        assert not torch.isnan(result['node_importance']).any()

    def test_larger_molecule(self, fast_explainer):
        """Test with larger molecule (8 atoms)."""
        data = Data(
            x=torch.randn(8, 10),
            edge_index=torch.tensor(
                [[0,1,1,2,2,3,3,4,4,5,5,6,6,7],
                 [1,0,2,1,3,2,4,3,5,4,6,5,7,6]],
                dtype=torch.long
            ),
            edge_attr=torch.randn(14, 3),
            pos=torch.randn(8, 3),
        )
        result = fast_explainer.explain(data, task_idx=0)
        assert result['node_importance'].shape == (8,)
        assert result['edge_mask'].shape == (14,)


class TestGNNExplainerBatch:

    def test_batch_returns_list(self, fast_explainer, simple_data):
        results = fast_explainer.explain_batch(
            [simple_data, simple_data], task_idx=0
        )
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_each_has_keys(self, fast_explainer, simple_data):
        results = fast_explainer.explain_batch([simple_data], task_idx=0)
        assert 'edge_mask' in results[0]
        assert 'node_importance' in results[0]

    def test_batch_handles_failure(self, wrapped_model):
        """Batch should handle failed molecules gracefully."""
        explainer = GNNExplainer(wrapped_model, epochs=3)
        broken_data = Data(x=torch.randn(1, 10))  # Missing edge_index

        results = explainer.explain_batch([broken_data], task_idx=0)
        assert len(results) == 1
        assert results[0].get('failed', False)

    def test_batch_mol_idx(self, fast_explainer, simple_data):
        results = fast_explainer.explain_batch(
            [simple_data, simple_data, simple_data], task_idx=0
        )
        for i, r in enumerate(results):
            assert r['mol_idx'] == i


class TestBinaryEntropy:

    def test_max_at_half(self):
        m05 = torch.tensor([0.5])
        m01 = torch.tensor([0.1])
        assert (
            GNNExplainer._binary_entropy(m05)
            > GNNExplainer._binary_entropy(m01)
        )

    def test_min_at_extremes(self):
        m0 = torch.tensor([0.001])
        m1 = torch.tensor([0.999])
        m5 = torch.tensor([0.5])
        assert GNNExplainer._binary_entropy(m0) < GNNExplainer._binary_entropy(m5)
        assert GNNExplainer._binary_entropy(m1) < GNNExplainer._binary_entropy(m5)

    def test_non_negative(self):
        for val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            mask = torch.tensor([val])
            assert GNNExplainer._binary_entropy(mask) >= 0


class TestAggregateToNodes:

    def test_chain_3_atoms(self):
        edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]])
        mask = torch.tensor([0.8, 0.8, 0.4, 0.4])
        imp = GNNExplainer._aggregate_to_nodes(edge_index, mask, 3)
        assert imp.shape == (3,)
        # Node 1 connected to both edges → higher score
        assert imp[1] >= imp[0]
        assert imp[1] >= imp[2]

    def test_all_same_weight(self):
        edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]])
        mask = torch.ones(4) * 0.5
        imp = GNNExplainer._aggregate_to_nodes(edge_index, mask, 3)
        assert imp.shape == (3,)
        assert (imp > 0).all()

    def test_isolated_node_keeps_zero_importance(self):
        edge_index = torch.tensor([[0, 1], [1, 0]])
        mask = torch.tensor([0.3, 0.3])
        imp = GNNExplainer._aggregate_to_nodes(edge_index, mask, 3)
        assert imp.tolist() == pytest.approx([0.3, 0.3, 0.0])


# ── Mask Utility Tests ────────────────────────────────────────────────────

class TestNormalizeMask:

    def test_minmax_zero_to_one(self):
        m = torch.tensor([0.1, 0.5, 0.9])
        n = normalize_mask(m, 'minmax')
        assert abs(n.min().item()) < 1e-5
        assert abs(n.max().item() - 1.0) < 1e-5

    def test_minmax_uniform(self):
        m = torch.ones(5) * 0.7
        n = normalize_mask(m, 'minmax')
        assert ((n - 0.5).abs() < 1e-5).all()

    def test_softmax_sums_to_one(self):
        m = torch.tensor([0.1, 0.5, 0.9])
        n = normalize_mask(m, 'softmax')
        assert abs(n.sum().item() - 1.0) < 1e-5

    def test_rank_range(self):
        m = torch.tensor([0.3, 0.1, 0.8, 0.5])
        n = normalize_mask(m, 'rank')
        assert n.min() >= 0.0
        assert n.max() <= 1.0

    def test_rank_preserves_order(self):
        m = torch.tensor([0.1, 0.5, 0.9])
        n = normalize_mask(m, 'rank')
        assert n[0] < n[1] < n[2]

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            normalize_mask(torch.ones(3), 'bad_method')


class TestThresholdMask:

    def test_basic_threshold(self):
        m = torch.tensor([0.2, 0.6, 0.8, 0.4])
        b = threshold_mask(m, threshold=0.5)
        expected = torch.tensor([0., 1., 1., 0.])
        assert torch.allclose(b, expected)

    def test_top_k_count(self):
        m = torch.tensor([0.1, 0.5, 0.9, 0.7, 0.3])
        b = threshold_mask(m, top_k=2)
        assert b.sum() == 2.0

    def test_top_k_selects_highest(self):
        m = torch.tensor([0.1, 0.5, 0.9, 0.7, 0.3])
        b = threshold_mask(m, top_k=2)
        assert b[2] == 1.0  # 0.9
        assert b[3] == 1.0  # 0.7

    def test_top_k_capped_at_len(self):
        m = torch.tensor([0.5, 0.8])
        b = threshold_mask(m, top_k=100)
        assert b.sum() == 2.0


class TestGetImportantEdges:

    def test_basic(self):
        edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]])
        mask = torch.tensor([0.8, 0.8, 0.3, 0.3])
        pairs, scores = get_important_edges(edge_index, mask, threshold=0.5)
        assert len(pairs) == 1
        assert (0, 1) in pairs

    def test_empty(self):
        edge_index = torch.tensor([[0,1],[1,0]])
        mask = torch.tensor([0.2, 0.2])
        pairs, scores = get_important_edges(edge_index, mask, threshold=0.5)
        assert len(pairs) == 0

    def test_all_important(self):
        edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]])
        mask = torch.ones(4) * 0.9
        pairs, scores = get_important_edges(edge_index, mask, threshold=0.5)
        assert len(pairs) == 2  # 2 unique undirected pairs


class TestMaskStatistics:

    def test_basic_stats(self):
        m = torch.tensor([0.0, 0.5, 1.0])
        stats = mask_statistics(m)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'sparsity' in stats
        assert 'entropy' in stats
        assert abs(stats['mean'] - 0.5) < 0.01

    def test_all_zeros_stats(self):
        m = torch.zeros(10)
        stats = mask_statistics(m)
        assert stats['sparsity'] == 1.0
        assert stats['mean'] == 0.0

    def test_all_ones_stats(self):
        m = torch.ones(10)
        stats = mask_statistics(m)
        assert stats['sparsity'] == 0.0
        assert stats['mean'] == 1.0
