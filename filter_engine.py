"""
Filter Engine: DSL parser, condition evaluator, cross-TF zone interpolation.

DSL grammar (simplified):
    expr     := or_expr
    or_expr  := and_expr ( 'OR' and_expr )*
    and_expr := atom ( 'AND' atom )*
    atom     := '(' expr ')' | condition
    condition := operand operator operand
    operand   := indicator_call | NUMBER
    indicator_call := NAME '(' NAME ',' NUMBER (',' NUMBER)* ')'
    operator  := '>' | '<' | '>=' | '<=' | '==' | '!='
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_provider import DataProvider, TF_ORDER, tf_minutes


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------

@dataclass
class IndicatorRef:
    name: str           # e.g. "RSI"
    timeframe: str      # e.g. "M15"
    params: List[Any]   # e.g. [14]


@dataclass
class Condition:
    left: Any   # IndicatorRef | float
    op: str
    right: Any  # IndicatorRef | float


@dataclass
class LogicExpr:
    operator: str           # "AND" | "OR"
    children: List[Any]     # Condition | LogicExpr


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(>=|<=|==|!=|>|<)"           # operators
    r"|([A-Za-z_][A-Za-z0-9_]*)"  # names / keywords
    r"|(-?\d+(?:\.\d+)?)"          # numbers
    r"|([(),])"                    # punctuation
    r")\s*"
)


def _tokenize(expr: str) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise SyntaxError(f"Unexpected character at position {pos}: {expr[pos]!r}")
        pos = m.end()
        if m.group(1):
            tokens.append(("OP", m.group(1)))
        elif m.group(2):
            word = m.group(2).upper()
            if word in ("AND", "OR"):
                tokens.append(("LOGIC", word))
            else:
                tokens.append(("NAME", m.group(2).upper()))
        elif m.group(3):
            tokens.append(("NUM", m.group(3)))
        elif m.group(4):
            tokens.append(("PUNCT", m.group(4)))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, tokens: List[Tuple[str, str]]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> Optional[Tuple[str, str]]:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self, kind: Optional[str] = None, value: Optional[str] = None) -> Tuple[str, str]:
        tok = self._peek()
        if tok is None:
            raise SyntaxError("Unexpected end of expression")
        if kind and tok[0] != kind:
            raise SyntaxError(f"Expected {kind}, got {tok}")
        if value and tok[1] != value:
            raise SyntaxError(f"Expected {value!r}, got {tok[1]!r}")
        self._pos += 1
        return tok

    def parse(self) -> Any:
        node = self._parse_or()
        if self._peek() is not None:
            raise SyntaxError(f"Unexpected token: {self._peek()}")
        return node

    def _parse_or(self) -> Any:
        left = self._parse_and()
        children = [left]
        while self._peek() == ("LOGIC", "OR"):
            self._consume()
            children.append(self._parse_and())
        if len(children) == 1:
            return left
        return LogicExpr("OR", children)

    def _parse_and(self) -> Any:
        left = self._parse_atom()
        children = [left]
        while self._peek() == ("LOGIC", "AND"):
            self._consume()
            children.append(self._parse_atom())
        if len(children) == 1:
            return left
        return LogicExpr("AND", children)

    def _parse_atom(self) -> Any:
        if self._peek() == ("PUNCT", "("):
            self._consume()
            node = self._parse_or()
            self._consume("PUNCT", ")")
            return node
        return self._parse_condition()

    def _parse_condition(self) -> Condition:
        left = self._parse_operand()
        op_tok = self._consume("OP")
        right = self._parse_operand()
        return Condition(left=left, op=op_tok[1], right=right)

    def _parse_operand(self) -> Any:
        tok = self._peek()
        if tok is None:
            raise SyntaxError("Expected operand")

        if tok[0] == "NUM":
            self._consume()
            return float(tok[1])

        if tok[0] == "NAME":
            name = tok[1]
            self._consume()
            # Check if it's an indicator call: NAME(TF, param, ...)
            if self._peek() == ("PUNCT", "("):
                self._consume("PUNCT", "(")
                tf_tok = self._consume("NAME")
                timeframe = tf_tok[1]
                params: List[Any] = []
                while self._peek() == ("PUNCT", ","):
                    self._consume()
                    p = self._consume("NUM")
                    params.append(float(p[1]))
                self._consume("PUNCT", ")")
                return IndicatorRef(name=name, timeframe=timeframe, params=params)
            # Bare name treated as a constant (shouldn't happen in well-formed DSL)
            raise SyntaxError(f"Unexpected bare name: {name}")

        raise SyntaxError(f"Unexpected token: {tok}")


def parse_filter(expr: str) -> Any:
    """Parse a DSL filter expression and return an AST node."""
    tokens = _tokenize(expr)
    return _Parser(tokens).parse()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def _eval_operand(
    operand: Any,
    data: Dict[str, pd.DataFrame],
    provider: DataProvider,
) -> Tuple[pd.Series, str]:
    """
    Evaluate an operand against data dict { timeframe: df }.
    Returns (series, timeframe) — timeframe is "" for scalar constants.
    """
    if isinstance(operand, float):
        # Scalar: return a constant-valued series (no specific TF)
        return operand, ""

    if isinstance(operand, IndicatorRef):
        tf = operand.timeframe
        if tf not in data:
            raise KeyError(f"Timeframe {tf} not loaded in data")
        df = data[tf]
        params: Dict[str, Any] = {}
        if operand.params:
            params["period"] = int(operand.params[0])
            if len(operand.params) > 1:
                params["fast"] = int(operand.params[0])
                params["slow"] = int(operand.params[1])
            if len(operand.params) > 2:
                params["signal"] = int(operand.params[2])
        series = provider.compute_indicator(df, operand.name, **params)
        series = series.values  # numpy array aligned with df index
        return series, tf

    raise TypeError(f"Unknown operand type: {type(operand)}")


def _apply_op(left: Any, op: str, right: Any) -> np.ndarray:
    ops = {
        ">":  np.greater,
        "<":  np.less,
        ">=": np.greater_equal,
        "<=": np.less_equal,
        "==": np.equal,
        "!=": np.not_equal,
    }
    fn = ops.get(op)
    if fn is None:
        raise ValueError(f"Unknown operator: {op}")
    return fn(left, right)


def evaluate_node(
    node: Any,
    data: Dict[str, pd.DataFrame],
    provider: DataProvider,
    target_tf: str,
) -> np.ndarray:
    """
    Evaluate an AST node for a given target timeframe.
    Returns a boolean numpy array aligned with data[target_tf].
    """
    if isinstance(node, Condition):
        lv, ltf = _eval_operand(node.left, data, provider)
        rv, rtf = _eval_operand(node.right, data, provider)

        # Determine source TF (whichever side has a TF)
        src_tf = ltf if ltf else rtf
        if not src_tf:
            src_tf = target_tf

        n_target = len(data[target_tf])

        # Scalar comparison
        if isinstance(lv, float) and isinstance(rv, float):
            result = _apply_op(np.array([lv]), node.op, np.array([rv]))
            return np.full(n_target, result[0], dtype=bool)

        # Compute raw boolean on source TF
        if isinstance(lv, float):
            raw = _apply_op(np.full(len(rv), lv), node.op, rv)
        elif isinstance(rv, float):
            raw = _apply_op(lv, node.op, np.full(len(lv), rv))
        else:
            # Both are arrays — must be same TF
            raw = _apply_op(lv, node.op, rv)

        if src_tf == target_tf:
            return raw.astype(bool)

        # Cross-TF projection
        return project_to_target(
            raw, data[src_tf], data[target_tf], src_tf, target_tf
        )

    if isinstance(node, LogicExpr):
        child_results = [
            evaluate_node(c, data, provider, target_tf) for c in node.children
        ]
        result = child_results[0]
        for cr in child_results[1:]:
            if node.operator == "AND":
                result = result & cr
            else:
                result = result | cr
        return result

    raise TypeError(f"Unknown AST node: {type(node)}")


# ---------------------------------------------------------------------------
# Cross-TF projection
# ---------------------------------------------------------------------------

def project_to_target(
    source_mask: np.ndarray,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_tf: str,
    target_tf: str,
) -> np.ndarray:
    """
    Project a boolean mask from source_tf bars onto target_tf bars.

    Each source bar covers [open_time, open_time + bar_duration).
    Each target bar's status is determined by which source bar contains
    the target bar's open_time (per spec: open_time of the target bar).
    If the source is a higher TF than the target, we project downward.
    """
    src_times = source_df["time"].values.astype("datetime64[ns]")
    tgt_times = target_df["time"].values.astype("datetime64[ns]")

    src_minutes = tf_minutes(source_tf)
    src_duration = np.timedelta64(src_minutes * 60, "s")

    n_tgt = len(tgt_times)
    result = np.zeros(n_tgt, dtype=bool)

    # Build source intervals as (start, end, valid)
    src_starts = src_times
    src_ends = src_times + src_duration

    # For each target bar, find which source bar it falls into
    # Use searchsorted for O(n log n) performance
    idx = np.searchsorted(src_starts, tgt_times, side="right") - 1
    valid_idx = (idx >= 0) & (idx < len(src_starts))

    # Check target bar open is strictly within source bar interval
    in_range = valid_idx & (tgt_times[valid_idx.nonzero()] < src_ends[idx[valid_idx]])

    # Build output
    result_valid = source_mask[idx[valid_idx]]
    result[valid_idx] = result_valid

    return result


# ---------------------------------------------------------------------------
# Segment extraction (for rendering)
# ---------------------------------------------------------------------------

@dataclass
class FilterSegment:
    t_start: pd.Timestamp
    t_end: pd.Timestamp
    valid: bool


def extract_segments(
    mask: np.ndarray,
    df: pd.DataFrame,
    timeframe: str,
) -> List[FilterSegment]:
    """
    Convert a boolean mask to a list of contiguous time segments.
    """
    segments: List[FilterSegment] = []
    if len(mask) == 0:
        return segments

    bar_dur = pd.Timedelta(minutes=tf_minutes(timeframe))
    times = df["time"].values
    i = 0
    while i < len(mask):
        j = i + 1
        while j < len(mask) and mask[j] == mask[i]:
            j += 1
        t_start = pd.Timestamp(times[i])
        t_end = pd.Timestamp(times[j - 1]) + bar_dur
        segments.append(FilterSegment(t_start=t_start, t_end=t_end, valid=bool(mask[i])))
        i = j
    return segments


# ---------------------------------------------------------------------------
# FilterEngine — top-level coordinator
# ---------------------------------------------------------------------------

class FilterEngine:
    """
    Holds a list of filter expressions and evaluates them against loaded data.
    """

    def __init__(self, provider: DataProvider) -> None:
        self._provider = provider
        self._filters: List[Dict[str, Any]] = []  # {"name", "expr", "ast", "enabled"}
        self._cross_tf_projection = True

    # ------------------------------------------------------------------
    # Filter management
    # ------------------------------------------------------------------

    def add_filter(self, name: str, expression: str) -> None:
        ast = parse_filter(expression)
        self._filters.append({
            "name": name,
            "expression": expression,
            "ast": ast,
            "enabled": True,
        })

    def remove_filter(self, name: str) -> None:
        self._filters = [f for f in self._filters if f["name"] != name]

    def set_filter_enabled(self, name: str, enabled: bool) -> None:
        for f in self._filters:
            if f["name"] == name:
                f["enabled"] = enabled

    def clear_filters(self) -> None:
        self._filters.clear()

    @property
    def filters(self) -> List[Dict[str, Any]]:
        return list(self._filters)

    def set_cross_tf_projection(self, enabled: bool) -> None:
        self._cross_tf_projection = enabled

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        data: Dict[str, pd.DataFrame],
        target_tf: str,
    ) -> np.ndarray:
        """
        Evaluate all enabled filters combined with AND on target_tf.
        Returns boolean mask aligned with data[target_tf].
        """
        enabled = [f for f in self._filters if f["enabled"]]
        if not enabled:
            n = len(data.get(target_tf, pd.DataFrame()))
            return np.ones(n, dtype=bool)

        n = len(data[target_tf])
        result = np.ones(n, dtype=bool)

        for f in enabled:
            try:
                mask = evaluate_node(f["ast"], data, self._provider, target_tf)
                result = result & mask
            except Exception:
                pass  # skip broken filters silently

        return result

    def get_segments(
        self,
        data: Dict[str, pd.DataFrame],
        target_tf: str,
    ) -> List[FilterSegment]:
        mask = self.evaluate(data, target_tf)
        return extract_segments(mask, data[target_tf], target_tf)

    # ------------------------------------------------------------------
    # Preset persistence
    # ------------------------------------------------------------------

    def save_preset(self, path: Path, preset_name: str) -> None:
        payload = {
            "name": preset_name,
            "filters": [
                {"name": f["name"], "expression": f["expression"], "enabled": f["enabled"]}
                for f in self._filters
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def load_preset(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self._filters.clear()
        for f in payload.get("filters", []):
            self.add_filter(f["name"], f["expression"])
            self.set_filter_enabled(f["name"], f.get("enabled", True))
        return payload.get("name", path.stem)
