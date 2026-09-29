"""Unit tests for Benchmark Topologies in MetaBench."""

import pytest
import numpy as np
from metabench.benchmarks import (
    create_benchmark,
    get_all_benchmark_names,
    get_benchmark_suite
)


def test_all_20_benchmarks_exist():
    names = get_all_benchmark_names()
    assert len(names) == 20
    suite = get_benchmark_suite(dim=10)
    assert len(suite) == 20


@pytest.mark.parametrize("name", get_all_benchmark_names())
def test_benchmark_evaluation_and_bounds(name):
    bench = create_benchmark(name, dim=5)
    d = bench.dimension

    # 1D single evaluation
    low, high = bench.get_bounds(d)
    mid = 0.5 * (low + high)
    val_1d = bench.evaluate(mid)
    assert isinstance(val_1d, float)
    assert np.isfinite(val_1d)

    # 2D batch evaluation
    batch = np.tile(mid, (10, 1))
    vals_batch = bench.evaluate(batch)
    assert vals_batch.shape == (10,)
    assert np.all(np.isfinite(vals_batch))


def test_global_optima_evaluation():
    # Sphere
    sphere = create_benchmark("Sphere", dim=10)
    assert sphere.evaluate(np.zeros(10)) == pytest.approx(0.0, abs=1e-7)

    # Booth
    booth = create_benchmark("Booth")
    assert booth.evaluate(np.array([1.0, 3.0])) == pytest.approx(0.0, abs=1e-7)

    # Matyas
    matyas = create_benchmark("Matyas")
    assert matyas.evaluate(np.array([0.0, 0.0])) == pytest.approx(0.0, abs=1e-7)

    # Rosenbrock
    rosen = create_benchmark("Rosenbrock", dim=5)
    assert rosen.evaluate(np.ones(5)) == pytest.approx(0.0, abs=1e-7)


def test_100_dimensional_scalability():
    """Verify that scalable functions handle 100-D search spaces seamlessly."""
    scalable_names = [
        "Sphere", "Sum of Squares", "Zakharov", "Dixon-Price",
        "Rosenbrock", "Rastrigin", "Ackley", "Griewank", "Schwefel", "Levy"
    ]
    for name in scalable_names:
        bench = create_benchmark(name, dim=100)
        assert bench.dimension == 100
        low, high = bench.get_bounds()
        assert len(low) == 100
        assert len(high) == 100
        test_x = np.random.uniform(low, high, size=(5, 100))
        res = bench.evaluate(test_x)
        assert res.shape == (5,)
        assert np.all(np.isfinite(res))
