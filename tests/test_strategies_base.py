import numpy as np
import pytest

from src.strategies import base


def test_weighted_average_handles_empty_and_zero_examples():
    assert base.weighted_average([]) == {}
    assert base.weighted_average([(0, {"acc": 1.0})]) == {}


def test_weighted_average_aggregates_metrics():
    metrics = [
        (5, {"acc": 0.8, "loss": 0.5}),
        (15, {"acc": 0.6, "loss": 0.3}),
    ]
    aggregated = base.weighted_average(metrics)
    assert pytest.approx(0.65) == aggregated["acc"]
    assert pytest.approx(0.35) == aggregated["loss"]


def test_aggregate_parameters_computes_weighted_sum():
    weights_results = [
        ([np.array([1.0, 2.0]), np.array([3.0])], 2),
        ([np.array([3.0, 4.0]), np.array([5.0])], 1),
    ]
    aggregated = base.aggregate_parameters(weights_results)

    expected_layer0 = np.array([(2 * 1.0 + 1 * 3.0) / 3, (2 * 2.0 + 1 * 4.0) / 3])
    expected_layer1 = np.array([(2 * 3.0 + 1 * 5.0) / 3])
    np.testing.assert_allclose(aggregated[0], expected_layer0)
    np.testing.assert_allclose(aggregated[1], expected_layer1)


def test_aggregate_parameters_empty_results():
    with pytest.raises(ValueError):
        base.aggregate_parameters([])


def test_compute_update_and_add_inplace():
    old = [np.array([1.0, 1.0], dtype=np.float32)]
    new = [np.array([2.5, 0.5], dtype=np.float32)]
    update = base.compute_update(new, old)

    assert np.allclose(update[0], np.array([1.5, -0.5], dtype=np.float32))

    dst = base.zeros_like(old)
    base.add_inplace(dst, update, alpha=2.0)
    assert np.allclose(dst[0], np.array([3.0, -1.0], dtype=np.float32))
