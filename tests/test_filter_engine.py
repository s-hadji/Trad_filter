"""
Unit tests for filter_engine: DSL parsing, condition evaluation,
and cross-TF zone interpolation.
"""
import numpy as np
import pandas as pd
import pytest

from filter_engine import (
    Condition,
    FilterEngine,
    FilterSegment,
    IndicatorRef,
    LogicExpr,
    extract_segments,
    parse_filter,
    project_to_target,
)
from tests.conftest import make_ohlcv, MockProvider


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParser:
    def test_simple_gt(self):
        ast = parse_filter("RSI(M15,14) > 30")
        assert isinstance(ast, Condition)
        assert isinstance(ast.left, IndicatorRef)
        assert ast.left.name == "RSI"
        assert ast.left.timeframe == "M15"
        assert ast.left.params == [14.0]
        assert ast.op == ">"
        assert ast.right == 30.0

    def test_and_expression(self):
        ast = parse_filter("RSI(M15,14) > 30 AND RSI(M15,14) < 50")
        assert isinstance(ast, LogicExpr)
        assert ast.operator == "AND"
        assert len(ast.children) == 2

    def test_or_expression(self):
        ast = parse_filter("RSI(M5,9) < 40 OR EMA(H1,20) > 1.05")
        assert isinstance(ast, LogicExpr)
        assert ast.operator == "OR"

    def test_nested_parens(self):
        ast = parse_filter("(RSI(M15,14) > 30 AND RSI(M15,14) < 50) OR EMA(H1,20) > 1.0")
        assert isinstance(ast, LogicExpr)
        assert ast.operator == "OR"
        left = ast.children[0]
        assert isinstance(left, LogicExpr)
        assert left.operator == "AND"

    def test_indicator_vs_indicator(self):
        ast = parse_filter("EMA(H1,20) > SMA(H1,50)")
        assert isinstance(ast, Condition)
        assert isinstance(ast.right, IndicatorRef)
        assert ast.right.name == "SMA"

    def test_close_indicator(self):
        ast = parse_filter("CLOSE(H1) > EMA(H1,20)")
        assert isinstance(ast, Condition)
        assert ast.left.name == "CLOSE"
        assert ast.left.params == []

    def test_macd_multi_param(self):
        ast = parse_filter("MACD_HIST(H1,12,26,9) > 0")
        assert isinstance(ast, Condition)
        assert ast.left.params == [12.0, 26.0, 9.0]

    def test_invalid_expression_raises(self):
        with pytest.raises((SyntaxError, Exception)):
            parse_filter("RSI(M15,14) >> 30")

    def test_missing_closing_paren_raises(self):
        with pytest.raises((SyntaxError, Exception)):
            parse_filter("(RSI(M15,14) > 30 AND EMA(H1,20) > 1.0")


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

class TestEvaluator:
    def _make_data(self, n=100):
        df = make_ohlcv(n, freq_minutes=15, seed=1)
        return {"M15": df}

    def test_rsi_gt_constant(self):
        data = self._make_data()
        provider = MockProvider()
        engine = FilterEngine(provider)
        engine.add_filter("f1", "RSI(M15,14) > 30")
        mask = engine.evaluate(data, "M15")
        assert mask.dtype == bool
        assert len(mask) == 100
        # RSI is always between 0 and 100 — expect some True values
        assert mask.any()

    def test_always_false_condition(self):
        data = self._make_data()
        provider = MockProvider()
        engine = FilterEngine(provider)
        engine.add_filter("f1", "RSI(M15,14) > 999")
        mask = engine.evaluate(data, "M15")
        assert not mask.any()

    def test_always_true_condition(self):
        # RSI has a 14-bar warm-up; bars before that are NaN → False.
        # After warm-up, RSI < 999 must be True for every bar.
        data = self._make_data()
        provider = MockProvider()
        engine = FilterEngine(provider)
        engine.add_filter("f1", "RSI(M15,14) < 999")
        mask = engine.evaluate(data, "M15")
        warmup = 14
        assert mask[warmup:].all()

    def test_and_logic(self):
        data = self._make_data()
        provider = MockProvider()
        engine = FilterEngine(provider)
        engine.add_filter("f1", "RSI(M15,14) > 0 AND RSI(M15,14) < 999")
        mask = engine.evaluate(data, "M15")
        warmup = 14
        assert mask[warmup:].all()

    def test_or_logic(self):
        data = self._make_data()
        provider = MockProvider()
        engine = FilterEngine(provider)
        engine.add_filter("f1", "RSI(M15,14) > 999 OR RSI(M15,14) < 999")
        mask = engine.evaluate(data, "M15")
        warmup = 14
        assert mask[warmup:].all()

    def test_multiple_filters_combined_with_and(self):
        data = self._make_data()
        provider = MockProvider()
        engine = FilterEngine(provider)
        engine.add_filter("f1", "RSI(M15,14) > 0")
        engine.add_filter("f2", "RSI(M15,14) > 999")
        mask = engine.evaluate(data, "M15")
        assert not mask.any()

    def test_disabled_filter_ignored(self):
        data = self._make_data()
        provider = MockProvider()
        engine = FilterEngine(provider)
        engine.add_filter("f1", "RSI(M15,14) > 999")
        engine.set_filter_enabled("f1", False)
        mask = engine.evaluate(data, "M15")
        assert mask.all()  # no active filters → all True

    def test_no_filters_returns_all_true(self):
        data = self._make_data()
        provider = MockProvider()
        engine = FilterEngine(provider)
        mask = engine.evaluate(data, "M15")
        assert mask.all()


# ---------------------------------------------------------------------------
# Cross-TF projection tests
# ---------------------------------------------------------------------------

class TestCrossTFProjection:
    def _aligned_data(self, n_m15=40, n_m5=120):
        """
        Create M15 and M5 DataFrames with aligned timestamps.
        M15 bar i covers [i*15min, (i+1)*15min).
        Each M15 bar aligns with exactly 3 M5 bars.
        """
        start = pd.Timestamp("2024-01-01 00:00")
        m15_times = pd.date_range(start=start, periods=n_m15, freq="15min")
        m5_times  = pd.date_range(start=start, periods=n_m5,  freq="5min")

        def _df(times, freq_m):
            rng = np.random.default_rng(99)
            n = len(times)
            c = 1.1 + np.cumsum(rng.normal(0, 0.0002, n))
            return pd.DataFrame({
                "time": times,
                "open": c - 0.0001,
                "high": c + 0.0005,
                "low":  c - 0.0005,
                "close": c,
                "volume": np.ones(n) * 100,
            })

        return _df(m15_times, 15), _df(m5_times, 5)

    def test_projection_length(self):
        m15, m5 = self._aligned_data(40, 120)
        mask_m15 = np.ones(40, dtype=bool)
        result = project_to_target(mask_m15, m15, m5, "M15", "M5")
        assert len(result) == 120

    def test_all_true_projects_all_true(self):
        m15, m5 = self._aligned_data(40, 120)
        mask_m15 = np.ones(40, dtype=bool)
        result = project_to_target(mask_m15, m15, m5, "M15", "M5")
        assert result.all()

    def test_all_false_projects_all_false(self):
        m15, m5 = self._aligned_data(40, 120)
        mask_m15 = np.zeros(40, dtype=bool)
        result = project_to_target(mask_m15, m15, m5, "M15", "M5")
        assert not result.any()

    def test_first_bar_true_rest_false(self):
        """Only first M15 bar valid → first 3 M5 bars should be True."""
        m15, m5 = self._aligned_data(40, 120)
        mask_m15 = np.zeros(40, dtype=bool)
        mask_m15[0] = True
        result = project_to_target(mask_m15, m15, m5, "M15", "M5")
        assert result[:3].all(), "First 3 M5 bars should be True"
        assert not result[3:].any(), "Rest should be False"

    def test_last_bar_true(self):
        """Last M15 bar valid → last 3 M5 bars should be True."""
        m15, m5 = self._aligned_data(40, 120)
        mask_m15 = np.zeros(40, dtype=bool)
        mask_m15[-1] = True
        result = project_to_target(mask_m15, m15, m5, "M15", "M5")
        assert result[-3:].all()
        assert not result[:-3].any()

    def test_alternating_projection(self):
        """Alternating M15 bars: M5 triplets should mirror the pattern."""
        m15, m5 = self._aligned_data(6, 18)
        mask_m15 = np.array([True, False, True, False, True, False])
        result = project_to_target(mask_m15, m15, m5, "M15", "M5")
        expected = np.array([True]*3 + [False]*3 + [True]*3 + [False]*3 + [True]*3 + [False]*3)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Segment extraction tests
# ---------------------------------------------------------------------------

class TestSegmentExtraction:
    def test_all_true_single_segment(self):
        df = make_ohlcv(10, freq_minutes=15)
        mask = np.ones(10, dtype=bool)
        segs = extract_segments(mask, df, "M15")
        assert len(segs) == 1
        assert segs[0].valid is True

    def test_all_false_single_segment(self):
        df = make_ohlcv(10, freq_minutes=15)
        mask = np.zeros(10, dtype=bool)
        segs = extract_segments(mask, df, "M15")
        assert len(segs) == 1
        assert segs[0].valid is False

    def test_alternating_produces_n_segments(self):
        df = make_ohlcv(6, freq_minutes=15)
        mask = np.array([True, False, True, False, True, False])
        segs = extract_segments(mask, df, "M15")
        assert len(segs) == 6
        for i, seg in enumerate(segs):
            assert seg.valid == (i % 2 == 0)

    def test_segment_timestamps_correct(self):
        df = make_ohlcv(4, start="2024-01-01 00:00", freq_minutes=15)
        mask = np.array([True, True, False, False])
        segs = extract_segments(mask, df, "M15")
        assert len(segs) == 2
        assert segs[0].valid is True
        assert segs[0].t_start == pd.Timestamp("2024-01-01 00:00")
        assert segs[0].t_end   == pd.Timestamp("2024-01-01 00:30")  # 2 bars × 15min
        assert segs[1].valid is False

    def test_empty_mask(self):
        df = make_ohlcv(0, freq_minutes=15)
        mask = np.array([], dtype=bool)
        segs = extract_segments(mask, df, "M15")
        assert segs == []


# ---------------------------------------------------------------------------
# Preset persistence tests
# ---------------------------------------------------------------------------

class TestPresetPersistence:
    def test_save_and_load(self, tmp_path):
        provider = MockProvider()
        engine = FilterEngine(provider)
        engine.add_filter("A", "RSI(M15,14) > 30")
        engine.add_filter("B", "EMA(H1,20) > SMA(H1,50)")

        p = tmp_path / "test_preset.json"
        engine.save_preset(p, "My Preset")

        engine2 = FilterEngine(provider)
        name = engine2.load_preset(p)
        assert name == "My Preset"
        assert len(engine2.filters) == 2
        assert engine2.filters[0]["name"] == "A"
        assert engine2.filters[1]["expression"] == "EMA(H1,20) > SMA(H1,50)"

    def test_load_sets_enabled(self, tmp_path):
        provider = MockProvider()
        engine = FilterEngine(provider)
        engine.add_filter("A", "RSI(M15,14) > 30")
        engine.set_filter_enabled("A", False)

        p = tmp_path / "preset.json"
        engine.save_preset(p, "Test")

        engine2 = FilterEngine(provider)
        engine2.load_preset(p)
        assert engine2.filters[0]["enabled"] is False
