"""Unit tests for evaluation/perf.py metrics computation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from evaluation.perf import perf_metrics


class TestPerfMetrics:
    def test_empty_latencies(self):
        m = perf_metrics([], total_wall_s=1.0)
        assert m["n"] == 0.0
        assert m["throughput_req_per_s"] == 0.0

    def test_single_value(self):
        m = perf_metrics([100.0], total_wall_s=0.1)
        assert m["n"] == 1.0
        assert m["latency_ms_mean"] == 100.0
        assert m["latency_ms_min"] == 100.0
        assert m["latency_ms_max"] == 100.0
        assert m["throughput_req_per_s"] == pytest.approx(10.0)

    def test_multiple_values(self):
        latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
        m = perf_metrics(latencies, total_wall_s=1.5)
        assert m["n"] == 5.0
        assert m["latency_ms_mean"] == 300.0
        assert m["latency_ms_min"] == 100.0
        assert m["latency_ms_max"] == 500.0
        assert m["latency_ms_p50"] == 300.0
        assert m["throughput_req_per_s"] == pytest.approx(5.0 / 1.5)

    def test_p95(self):
        latencies = list(range(1, 101))  # 1 to 100
        m = perf_metrics([float(x) for x in latencies], total_wall_s=10.0)
        assert m["latency_ms_p95"] == pytest.approx(95.05, abs=1.0)

    def test_cv_zero_mean(self):
        m = perf_metrics([0.0, 0.0, 0.0], total_wall_s=1.0)
        assert m["latency_ms_cv"] == 0.0

    def test_cv_nonzero(self):
        m = perf_metrics([100.0, 100.0, 100.0], total_wall_s=1.0)
        assert m["latency_ms_cv"] == pytest.approx(0.0)

    def test_zero_wall_time(self):
        m = perf_metrics([100.0], total_wall_s=0.0)
        assert m["throughput_req_per_s"] == 0.0


class TestPerfMetricsConsistency:
    def test_std_positive(self):
        m = perf_metrics([100.0, 200.0, 300.0], total_wall_s=1.0)
        assert m["latency_ms_std"] > 0

    def test_min_le_mean_le_max(self):
        m = perf_metrics([50.0, 150.0, 250.0], total_wall_s=1.0)
        assert m["latency_ms_min"] <= m["latency_ms_mean"] <= m["latency_ms_max"]

    def test_p50_between_min_max(self):
        m = perf_metrics([10.0, 20.0, 30.0, 40.0, 50.0], total_wall_s=1.0)
        assert m["latency_ms_min"] <= m["latency_ms_p50"] <= m["latency_ms_max"]
