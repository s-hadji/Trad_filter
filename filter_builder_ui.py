"""
Filter Builder UI — condition builder, filter list with inline delete,
preset load/save. Emits filter_applied and filters_changed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from filter_engine import FilterEngine, parse_filter

TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
INDICATORS = [
    "RSI", "EMA", "SMA", "MACD_HIST", "ATR",
    "BBANDS_UPPER", "BBANDS_MID", "BBANDS_LOWER",
    "STOCH_K", "STOCH_D",
    "CLOSE", "OPEN", "HIGH", "LOW", "VOLUME",
]
OPERATORS = [">", "<", ">=", "<=", "==", "!="]
LOGICS    = ["AND", "OR"]

PRESETS_DIR = Path(__file__).parent / "presets"

BG0    = "#0d0d1a"
BG1    = "#12121f"
BG2    = "#1a1a2e"
BG3    = "#22223a"
ACCENT = "#4f6ef7"
SUCCESS= "#10b981"
DANGER = "#ef4444"
TEXT1  = "#e2e2f0"
TEXT2  = "#9090b0"
BORDER = "#2a2a45"


# ── Condition row ──────────────────────────────────────────────────────────────

class ConditionRow(QWidget):
    removed = pyqtSignal(object)
    changed = pyqtSignal()

    def __init__(self, logic: str = "AND", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("condRow")
        self.setStyleSheet(f"""
            #condRow {{ background:{BG2}; border-radius:6px; border:1px solid {BORDER}; }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background:{BG3}; border:1px solid {BORDER};
                border-radius:4px; padding:3px 5px; color:{TEXT1};
                min-height:24px;
            }}
            QPushButton#rmBtn {{
                background:{DANGER}; border:none; border-radius:4px;
                color:#fff; font-weight:700; font-size:10pt;
            }}
            QPushButton#rmBtn:hover {{ background:#f87171; }}
        """)
        self._build(logic)

    def _build(self, logic: str) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 5, 8, 5)
        row.setSpacing(5)

        self._logic_cb = QComboBox()
        self._logic_cb.addItems(LOGICS)
        self._logic_cb.setCurrentText(logic)
        self._logic_cb.setFixedWidth(60)
        self._logic_cb.setStyleSheet(
            f"QComboBox {{ background:{ACCENT}; color:#fff; font-weight:700; border:none; border-radius:4px; }}"
        )
        self._logic_cb.currentTextChanged.connect(self.changed)
        row.addWidget(self._logic_cb)

        self._left_ind = QComboBox(); self._left_ind.addItems(INDICATORS)
        self._left_ind.setFixedWidth(120)
        self._left_ind.currentTextChanged.connect(self._on_left_ind)
        self._left_ind.currentTextChanged.connect(self.changed)
        row.addWidget(self._left_ind)

        self._left_tf = QComboBox(); self._left_tf.addItems(TIMEFRAMES)
        self._left_tf.setCurrentText("M15"); self._left_tf.setFixedWidth(58)
        self._left_tf.currentTextChanged.connect(self.changed)
        row.addWidget(self._left_tf)

        self._left_p = QSpinBox()
        self._left_p.setRange(1, 500); self._left_p.setValue(14)
        self._left_p.setFixedWidth(56); self._left_p.setPrefix("p:")
        self._left_p.valueChanged.connect(self.changed)
        row.addWidget(self._left_p)

        self._op = QComboBox(); self._op.addItems(OPERATORS)
        self._op.setFixedWidth(56)
        self._op.setStyleSheet(
            f"QComboBox {{ background:{BG0}; color:{ACCENT}; font-weight:700; border:1px solid {ACCENT}; border-radius:4px; }}"
        )
        self._op.currentTextChanged.connect(self.changed)
        row.addWidget(self._op)

        self._right_mode = QComboBox(); self._right_mode.addItems(["Value"] + INDICATORS)
        self._right_mode.setFixedWidth(120)
        self._right_mode.currentTextChanged.connect(self._on_right_mode)
        self._right_mode.currentTextChanged.connect(self.changed)
        row.addWidget(self._right_mode)

        self._right_tf = QComboBox(); self._right_tf.addItems(TIMEFRAMES)
        self._right_tf.setCurrentText("M15"); self._right_tf.setFixedWidth(58)
        self._right_tf.setVisible(False)
        self._right_tf.currentTextChanged.connect(self.changed)
        row.addWidget(self._right_tf)

        self._right_p = QSpinBox()
        self._right_p.setRange(1, 500); self._right_p.setValue(14)
        self._right_p.setFixedWidth(56); self._right_p.setPrefix("p:")
        self._right_p.setVisible(False)
        self._right_p.valueChanged.connect(self.changed)
        row.addWidget(self._right_p)

        self._right_val = QDoubleSpinBox()
        self._right_val.setRange(-1e9, 1e9); self._right_val.setDecimals(4)
        self._right_val.setValue(30.0); self._right_val.setFixedWidth(96)
        self._right_val.valueChanged.connect(self.changed)
        row.addWidget(self._right_val)

        btn = QPushButton("✕")
        btn.setObjectName("rmBtn")
        btn.setFixedSize(28, 28)
        btn.setToolTip("Remove condition")
        btn.clicked.connect(lambda: self.removed.emit(self))
        row.addWidget(btn)

    def _on_left_ind(self, ind: str) -> None:
        no_p = {"CLOSE", "OPEN", "HIGH", "LOW", "VOLUME"}
        self._left_p.setEnabled(ind not in no_p)

    def _on_right_mode(self, mode: str) -> None:
        is_val = mode == "Value"
        self._right_val.setVisible(is_val)
        self._right_tf.setVisible(not is_val)
        self._right_p.setVisible(not is_val)

    def set_logic_visible(self, v: bool) -> None:
        self._logic_cb.setVisible(v)

    def logic(self) -> str:
        return self._logic_cb.currentText()

    def to_expression_part(self) -> str:
        ind = self._left_ind.currentText()
        tf  = self._left_tf.currentText()
        p   = self._left_p.value()
        op  = self._op.currentText()
        no_p = {"CLOSE", "OPEN", "HIGH", "LOW", "VOLUME"}
        left = f"{ind}({tf})" if ind in no_p else f"{ind}({tf},{p})"

        mode = self._right_mode.currentText()
        if mode == "Value":
            right = str(self._right_val.value())
        else:
            right = f"{mode}({self._right_tf.currentText()},{self._right_p.value()})"
        return f"{left} {op} {right}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logic": self._logic_cb.currentText(),
            "left_ind": self._left_ind.currentText(),
            "left_tf": self._left_tf.currentText(),
            "left_period": self._left_p.value(),
            "op": self._op.currentText(),
            "right_mode": self._right_mode.currentText(),
            "right_value": self._right_val.value(),
            "right_tf": self._right_tf.currentText(),
            "right_period": self._right_p.value(),
        }

    def from_dict(self, d: Dict[str, Any]) -> None:
        self._logic_cb.setCurrentText(d.get("logic", "AND"))
        self._left_ind.setCurrentText(d.get("left_ind", "RSI"))
        self._left_tf.setCurrentText(d.get("left_tf", "M15"))
        self._left_p.setValue(d.get("left_period", 14))
        self._op.setCurrentText(d.get("op", ">"))
        mode = d.get("right_mode", "Value")
        self._right_mode.setCurrentText(mode)
        self._on_right_mode(mode)
        self._right_val.setValue(d.get("right_value", 30.0))
        self._right_tf.setCurrentText(d.get("right_tf", "M15"))
        self._right_p.setValue(d.get("right_period", 14))


# ── Filter badge (active list) ─────────────────────────────────────────────────

class FilterBadge(QWidget):
    """
    Compact card showing filter name + expression.
    Delete button is always visible (red, on the right).
    Checkbox to enable/disable.
    """
    sig_toggle = pyqtSignal(str, bool)
    sig_remove = pyqtSignal(str)

    def __init__(
        self, name: str, expression: str, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.filter_name = name
        self.setObjectName("filterBadge")
        self.setStyleSheet(f"""
            #filterBadge {{
                background:{BG2};
                border:1px solid {BORDER};
                border-radius:7px;
            }}
            QLabel  {{ background:transparent; border:none; }}
            QCheckBox {{ background:transparent; border:none; color:{TEXT1}; }}
            QPushButton#delBtn {{
                background:{DANGER};
                border:none;
                border-radius:5px;
                color:#fff;
                font-weight:900;
                font-size:12pt;
                min-width:32px;
                min-height:32px;
            }}
            QPushButton#delBtn:hover {{ background:#f87171; }}
        """)
        self._build(name, expression)

    def _build(self, name: str, expression: str) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 6, 8, 6)
        row.setSpacing(8)

        chk = QCheckBox()
        chk.setChecked(True)
        chk.setFixedWidth(20)
        chk.toggled.connect(lambda v: self.sig_toggle.emit(self.filter_name, v))
        row.addWidget(chk)

        dot = QLabel("◆")
        dot.setStyleSheet(f"color:{ACCENT}; font-size:7pt;")
        row.addWidget(dot)

        vbox = QVBoxLayout()
        vbox.setSpacing(1)
        vbox.setContentsMargins(0, 0, 0, 0)

        name_lbl = QLabel(f"<b>{name}</b>")
        name_lbl.setStyleSheet(f"color:{TEXT1}; font-size:9pt;")
        vbox.addWidget(name_lbl)

        expr_lbl = QLabel(expression[:62] + ("…" if len(expression) > 62 else ""))
        expr_lbl.setStyleSheet(f"color:{TEXT2}; font-size:7.5pt;")
        vbox.addWidget(expr_lbl)

        row.addLayout(vbox)
        row.addStretch()

        # Large, always-visible delete button
        del_btn = QPushButton("✕")
        del_btn.setObjectName("delBtn")
        del_btn.setFixedSize(32, 32)
        del_btn.setToolTip(f'Delete filter "{name}"')
        del_btn.clicked.connect(lambda: self.sig_remove.emit(self.filter_name))
        row.addWidget(del_btn)


# ── Filter Builder Panel ───────────────────────────────────────────────────────

class FilterBuilderPanel(QWidget):
    filter_applied  = pyqtSignal()
    filters_changed = pyqtSignal()

    def __init__(self, engine: FilterEngine, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._rows: List[ConditionRow] = []
        self._counter = 0
        self.setStyleSheet(f"QWidget {{ background:{BG1}; }}")
        self._build_ui()
        self._load_presets_list()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Condition builder ──────────────────────────────────
        builder = QGroupBox("Condition Builder")
        bl = QVBoxLayout(builder)
        bl.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(250)
        scroll.setStyleSheet(
            f"background:{BG2}; border:1px solid {BORDER}; border-radius:6px;"
        )
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._rows_w = QWidget()
        self._rows_w.setStyleSheet(f"background:{BG2};")
        self._rows_l = QVBoxLayout(self._rows_w)
        self._rows_l.setContentsMargins(6, 6, 6, 6)
        self._rows_l.setSpacing(4)
        self._rows_l.addStretch()
        scroll.setWidget(self._rows_w)
        bl.addWidget(scroll)

        btn_add_row = QPushButton("＋  Add Condition")
        btn_add_row.setFixedHeight(30)
        btn_add_row.clicked.connect(self._add_row)
        bl.addWidget(btn_add_row)

        # Expression preview
        lbl_prev = QLabel("Expression")
        lbl_prev.setStyleSheet(f"color:{TEXT2}; font-size:8pt; font-weight:600;")
        bl.addWidget(lbl_prev)

        self._preview = QLineEdit()
        self._preview.setReadOnly(True)
        self._preview.setFont(QFont("Cascadia Code, Consolas, Monospace", 8))
        self._preview.setStyleSheet(
            f"background:{BG0}; border:1px solid {BORDER}; border-radius:5px; "
            f"color:#86efac; padding:4px 8px;"
        )
        self._preview.setPlaceholderText("Build conditions above…")
        bl.addWidget(self._preview)

        # Name + single-click apply
        name_row = QHBoxLayout()
        lbl_n = QLabel("Name")
        lbl_n.setStyleSheet(f"color:{TEXT2}; font-size:8pt;")
        lbl_n.setFixedWidth(42)
        name_row.addWidget(lbl_n)
        self._name_in = QLineEdit("My Filter")
        self._name_in.setFixedHeight(28)
        name_row.addWidget(self._name_in)
        bl.addLayout(name_row)

        # ── Single-click Apply CTA (inside builder card) ───────
        self._apply_btn = QPushButton("▶  Apply")
        self._apply_btn.setProperty("primary", True)
        self._apply_btn.setFixedHeight(40)
        self._apply_btn.setEnabled(False)
        self._apply_btn.setToolTip(
            "Validate the expression, add it to the filter list and\n"
            "immediately apply zones to all chart panels."
        )
        self._apply_btn.clicked.connect(self._apply)
        bl.addWidget(self._apply_btn)

        root.addWidget(builder)

        # ── Active filters ─────────────────────────────────────
        active = QGroupBox("Active Filters")
        al = QVBoxLayout(active)
        al.setSpacing(4)

        self._badges_l = QVBoxLayout()
        self._badges_l.setSpacing(4)
        al.addLayout(self._badges_l)

        self._empty_lbl = QLabel("No filters yet.")
        self._empty_lbl.setStyleSheet(
            f"color:{TEXT2}; font-style:italic; padding:10px; qproperty-alignment:AlignCenter;"
        )
        self._badges_l.addWidget(self._empty_lbl)

        # Clear all — always visible
        self._clear_btn = QPushButton("🗑  Clear All Filters")
        self._clear_btn.setFixedHeight(30)
        self._clear_btn.setProperty("danger", True)
        self._clear_btn.clicked.connect(self._clear_all)
        al.addWidget(self._clear_btn)

        root.addWidget(active)

        # ── Presets ────────────────────────────────────────────
        preset = QGroupBox("Presets")
        pl = QHBoxLayout(preset)
        pl.setSpacing(6)

        self._preset_cb = QComboBox()
        self._preset_cb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        pl.addWidget(self._preset_cb)

        b_load = QPushButton("Load"); b_load.setFixedWidth(54)
        b_load.clicked.connect(self._load_preset)
        pl.addWidget(b_load)

        b_save = QPushButton("Save…"); b_save.setFixedWidth(54)
        b_save.clicked.connect(self._save_preset)
        pl.addWidget(b_save)

        root.addWidget(preset)
        root.addStretch()

        self._add_row()  # start with one condition

    # ── Condition rows ─────────────────────────────────────────────────────────

    def _add_row(self) -> None:
        row = ConditionRow("AND")
        row.removed.connect(self._rm_row)
        row.changed.connect(self._update_preview)
        if not self._rows:
            row.set_logic_visible(False)
        self._rows.append(row)
        self._rows_l.insertWidget(self._rows_l.count() - 1, row)
        self._update_preview()

    def _rm_row(self, row: ConditionRow) -> None:
        if row in self._rows:
            self._rows.remove(row)
            self._rows_l.removeWidget(row)
            row.setParent(None)   # detaches from widget tree → visible + garbage collected
        if self._rows:
            self._rows[0].set_logic_visible(False)
        self._update_preview()

    def _update_preview(self) -> None:
        if not self._rows:
            self._preview.setText("")
            self._apply_btn.setEnabled(False)
            return
        parts = []
        for i, r in enumerate(self._rows):
            if i > 0:
                parts.append(r.logic())
            parts.append(r.to_expression_part())
        expr = " ".join(parts)
        self._preview.setText(expr)
        self._apply_btn.setEnabled(bool(expr.strip()))

    # ── Filter list ────────────────────────────────────────────────────────────

    def _refresh_badges(self) -> None:
        # Clear existing badges
        while self._badges_l.count():
            item = self._badges_l.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        filters = self._engine.filters
        if not filters:
            self._badges_l.addWidget(self._empty_lbl)
            self._apply_btn.setEnabled(False)
            return

        for f in filters:
            badge = FilterBadge(f["name"], f["expression"])
            badge.sig_toggle.connect(self._on_toggle)
            badge.sig_remove.connect(self._on_remove)
            self._badges_l.addWidget(badge)

    def _on_toggle(self, name: str, enabled: bool) -> None:
        self._engine.set_filter_enabled(name, enabled)
        self.filters_changed.emit()

    def _on_remove(self, name: str) -> None:
        self._engine.remove_filter(name)
        self._refresh_badges()
        self.filters_changed.emit()

    def _clear_all(self) -> None:
        self._engine.clear_filters()
        self._refresh_badges()
        self.filters_changed.emit()

    # ── Apply (validate + register + render — one click) ──────────────────────

    def _apply(self) -> None:
        """
        Single-step action:
        1. Validate the current expression in the preview.
        2. Register (or replace) it in the engine under the chosen name.
        3. Emit filter_applied so MainWindow renders zones immediately.
        4. Show a brief ✓ confirmation on the button.
        5. Advance the default name counter so the next filter gets a fresh name.
        """
        expr = self._preview.text().strip()
        if not expr:
            return
        name = self._name_in.text().strip() or f"Filter {self._counter + 1}"
        try:
            parse_filter(expr)   # syntax check only
        except Exception as e:
            QMessageBox.warning(self, "Invalid expression", str(e))
            return

        # Register in engine (replace if same name)
        self._engine.remove_filter(name)
        self._engine.add_filter(name, expr)
        self._counter += 1

        # Prepare a fresh name for the next filter
        self._name_in.setText(f"Filter {self._counter + 1}")

        self._refresh_badges()
        self.filters_changed.emit()
        self.filter_applied.emit()

        # Visual confirmation
        self._apply_btn.setText("✓  Applied!")
        self._apply_btn.setStyleSheet(
            f"QPushButton {{ background:{SUCCESS}; border:none; color:#fff; "
            f"font-weight:700; font-size:9.5pt; border-radius:5px; padding:7px 18px; }}"
        )
        QTimer.singleShot(1600, self._reset_apply_btn)

    def _reset_apply_btn(self) -> None:
        self._apply_btn.setText("▶  Apply")
        self._apply_btn.setStyleSheet("")

    # ── Presets ────────────────────────────────────────────────────────────────

    def _load_presets_list(self) -> None:
        self._preset_cb.clear()
        PRESETS_DIR.mkdir(exist_ok=True)
        for p in sorted(PRESETS_DIR.glob("*.json")):
            self._preset_cb.addItem(p.stem, userData=p)

    def _load_preset(self) -> None:
        path = self._preset_cb.currentData()
        if not path:
            return
        try:
            self._engine.clear_filters()
            self._engine.load_preset(Path(path))
            self._refresh_badges()
            self._apply_btn.setEnabled(bool(self._engine.filters))
            self.filters_changed.emit()
        except Exception as e:
            QMessageBox.warning(self, "Load error", str(e))

    def _save_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save preset", str(PRESETS_DIR), "JSON Files (*.json)"
        )
        if path:
            p = Path(path)
            self._engine.save_preset(p, p.stem)
            self._load_presets_list()
