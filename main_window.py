"""
Main window: modern dark UI, multi-TF panels, parallel data loading, fast filter.
GPU acceleration via cuDF when CUDA is available (gpu_support.py).
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd

from PyQt6.QtCore import QDateTime, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from chart_widget import ChartPanel
from data_provider import DataProvider, TF_ORDER
from filter_builder_ui import FilterBuilderPanel
from filter_engine import FilterEngine
from gpu_support import GPU_STATE, gpu_init, hardware_report

# ── Palette ──────────────────────────────────────────────────────────────────
BG0      = "#0d0d1a"   # deepest background
BG1      = "#12121f"   # panel background
BG2      = "#1a1a2e"   # card / input background
BG3      = "#22223a"   # hover
ACCENT   = "#4f6ef7"   # indigo blue
ACCENT2  = "#7c3aed"   # violet (gradient partner)
SUCCESS  = "#10b981"
WARNING  = "#f59e0b"
DANGER   = "#ef4444"
TEXT1    = "#e2e2f0"
TEXT2    = "#9090b0"
BORDER   = "#2a2a45"

STYLESHEET = f"""
/* ── Global ──────────────────────────────────────────────── */
QWidget {{
    background-color: {BG1};
    color: {TEXT1};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 9pt;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background-color: {BG0};
    border: none;
}}

/* ── Toolbar ─────────────────────────────────────────────── */
QToolBar {{
    background: {BG0};
    border-bottom: 1px solid {BORDER};
    padding: 4px 8px;
    spacing: 6px;
}}
QToolBar QLabel {{
    color: {TEXT2};
    font-size: 8pt;
}}

/* ── Inputs ──────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px 8px;
    color: {TEXT1};
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {BG2};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    outline: none;
}}

/* ── Buttons ─────────────────────────────────────────────── */
QPushButton {{
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px 14px;
    color: {TEXT1};
}}
QPushButton:hover  {{ background: {BG3}; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {BG0}; }}

QPushButton[primary="true"] {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 {ACCENT}, stop:1 {ACCENT2});
    border: none;
    color: #ffffff;
    font-weight: 600;
    font-size: 9.5pt;
    padding: 7px 18px;
}}
QPushButton[primary="true"]:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #667aff, stop:1 #9b59e8);
}}
QPushButton[primary="true"]:pressed {{ opacity: 0.85; }}

QPushButton[danger="true"] {{
    background: {DANGER};
    border: none;
    color: #fff;
    font-weight: 600;
}}
QPushButton[danger="true"]:hover {{ background: #f87171; }}

/* ── Before/After toggle ─────────────────────────────────── */
QPushButton#beforeAfterBtn {{
    background: {BG2};
    border: 1px solid {ACCENT};
    border-radius: 5px;
    padding: 5px 16px;
    color: {ACCENT};
    font-weight: 600;
}}
QPushButton#beforeAfterBtn:checked {{
    background: {ACCENT};
    color: #fff;
}}
QPushButton#beforeAfterBtn:hover {{ background: {BG3}; }}

/* ── Group boxes ─────────────────────────────────────────── */
QGroupBox {{
    background: {BG1};
    border: 1px solid {BORDER};
    border-radius: 7px;
    margin-top: 10px;
    padding-top: 6px;
    font-weight: 600;
    font-size: 8.5pt;
    color: {TEXT2};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {ACCENT};
}}

/* ── Checkboxes ──────────────────────────────────────────── */
QCheckBox {{
    color: {TEXT1};
    spacing: 5px;
}}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {BG2};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}

/* ── Dock ────────────────────────────────────────────────── */
QDockWidget {{
    color: {TEXT1};
    font-weight: 600;
}}
QDockWidget::title {{
    background: {BG0};
    padding: 6px 10px;
    border-bottom: 1px solid {BORDER};
    text-align: left;
    font-size: 9pt;
    font-weight: 700;
    color: {TEXT1};
}}

/* ── Status bar ──────────────────────────────────────────── */
QStatusBar {{
    background: {BG0};
    border-top: 1px solid {BORDER};
    color: {TEXT2};
    font-size: 8pt;
}}

/* ── Scrollbars ──────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {BG1};
    width: 6px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(BG1))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(TEXT1))
    palette.setColor(QPalette.ColorRole.Base,            QColor(BG2))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(BG0))
    palette.setColor(QPalette.ColorRole.Text,            QColor(TEXT1))
    palette.setColor(QPalette.ColorRole.Button,          QColor(BG2))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(TEXT1))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)



# ── Background workers ────────────────────────────────────────────────────────

class DataFetchWorker(QThread):
    """
    Fetches OHLCV for all requested timeframes in parallel
    (one MT5/mock call per thread) then emits the result dict.

    Supports two modes:
    - **Bar count**: ``from_date=None`` → calls ``provider.get_ohlcv(symbol, tf, nb_bars)``
    - **Date range**: ``from_date`` and ``to_date`` set → calls ``provider.get_ohlcv_range(…)``
    """
    data_ready = pyqtSignal(dict)   # {tf: pd.DataFrame}

    def __init__(
        self, provider, symbol: str, tfs: List[str],
        nb_bars: int = 500, force: bool = False,
        from_date: Optional[pd.Timestamp] = None,
        to_date:   Optional[pd.Timestamp] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._provider   = provider
        self._symbol     = symbol
        self._tfs        = tfs
        self._nb_bars    = nb_bars
        self._force      = force
        self._from_date  = from_date
        self._to_date    = to_date

    def run(self) -> None:
        loaded: Dict[str, object] = {}
        use_range = self._from_date is not None and self._to_date is not None
        n_workers = min(len(self._tfs), 8)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            if use_range:
                futures = {
                    ex.submit(
                        self._provider.get_ohlcv_range,
                        self._symbol, tf, self._from_date, self._to_date, self._force,
                    ): tf
                    for tf in self._tfs
                }
            else:
                futures = {
                    ex.submit(
                        self._provider.get_ohlcv,
                        self._symbol, tf, self._nb_bars, self._force,
                    ): tf
                    for tf in self._tfs
                }
            for fut in as_completed(futures):
                tf = futures[fut]
                try:
                    loaded[tf] = fut.result()
                except Exception:
                    pass
        self.data_ready.emit(loaded)


class FilterWorker(QThread):
    """
    Computes filter segments for all timeframes in the background.
    Emits one signal per TF as soon as it's ready.
    """
    segments_ready = pyqtSignal(str, list)  # (tf, List[FilterSegment])

    def __init__(self, engine, data: Dict, tfs: List[str], parent=None):
        super().__init__(parent)
        self._engine = engine
        self._data   = data
        self._tfs    = tfs

    def run(self) -> None:
        for tf in self._tfs:
            if tf not in self._data:
                continue
            try:
                segs = self._engine.get_segments(self._data, tf)
                self.segments_ready.emit(tf, segs)
            except Exception:
                self.segments_ready.emit(tf, [])


# ── TF Selector ───────────────────────────────────────────────────────────────

class TFSelector(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._checks: Dict[str, QCheckBox] = {}
        defaults = {"M5", "M15", "H1"}
        for tf in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
            chk = QCheckBox(tf)
            chk.setChecked(tf in defaults)
            layout.addWidget(chk)
            self._checks[tf] = chk

    def active_timeframes(self) -> List[str]:
        return [tf for tf, chk in self._checks.items() if chk.isChecked()]

    def connect_changed(self, slot) -> None:
        for chk in self._checks.values():
            chk.toggled.connect(slot)


# ── Separator ─────────────────────────────────────────────────────────────────

class VLine(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(1)
        self.setStyleSheet(f"background:{BORDER};")


# ── Status dot ────────────────────────────────────────────────────────────────

class StatusDot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self._color = DANGER
        self.setStyleSheet(f"border-radius:5px; background:{self._color};")

    def set_connected(self, ok: bool) -> None:
        self._color = SUCCESS if ok else WARNING
        self.setStyleSheet(f"border-radius:5px; background:{self._color};")


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MT5 Chart Analyzer")
        self.resize(1680, 960)

        # ── GPU init (before anything else so status bar reflects result) ────
        gpu_init()

        self._provider = DataProvider()
        self._engine   = FilterEngine(self._provider)

        self._symbol        = "EURUSD"
        self._nb_bars       = 300
        self._show_zones    = False   # zones hidden until filter applied
        self._use_date_range = False  # toolbar toggle
        self._panels: Dict[str, ChartPanel] = {}
        self._data:   Dict[str, object]     = {}

        self._build_toolbar()
        self._build_central()
        self._build_filter_dock()
        self._build_statusbar()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._auto_refresh)

        self._try_connect()
        self._rebuild_panels()

    # ── Toolbar ──────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        tb.setIconSize(__import__("PyQt6.QtCore", fromlist=["QSize"]).QSize(16, 16))
        self.addToolBar(tb)

        # Symbol
        lbl_sym = QLabel("  Symbol")
        lbl_sym.setStyleSheet(f"color:{TEXT2}; font-size:8pt;")
        tb.addWidget(lbl_sym)

        self._symbol_edit = QLineEdit(self._symbol)
        self._symbol_edit.setFixedWidth(90)
        self._symbol_edit.returnPressed.connect(self._on_symbol_changed)
        tb.addWidget(self._symbol_edit)

        tb.addWidget(VLine())

        # TF checkboxes
        lbl_tf = QLabel("  Timeframes")
        lbl_tf.setStyleSheet(f"color:{TEXT2}; font-size:8pt;")
        tb.addWidget(lbl_tf)

        self._tf_selector = TFSelector()
        self._tf_selector.connect_changed(self._rebuild_panels)
        tb.addWidget(self._tf_selector)

        tb.addWidget(VLine())

        # Bar count — editable spinbox (100 → 50 000)
        self._lbl_bars = QLabel("  Bars")
        self._lbl_bars.setStyleSheet(f"color:{TEXT2}; font-size:8pt;")
        tb.addWidget(self._lbl_bars)

        self._bars_spin = QSpinBox()
        self._bars_spin.setRange(100, 50_000)
        self._bars_spin.setValue(self._nb_bars)
        self._bars_spin.setSingleStep(500)
        self._bars_spin.setFixedWidth(90)
        self._bars_spin.setToolTip(
            "Number of bars to load (100–50 000).\n"
            "Press Enter or click Refresh to apply."
        )
        self._bars_spin.editingFinished.connect(self._on_bars_changed)
        tb.addWidget(self._bars_spin)

        # Quick-load presets
        self._bars_quick = QComboBox()
        self._bars_quick.addItems(["500", "1 000", "2 000", "5 000", "10 000", "20 000"])
        self._bars_quick.setCurrentIndex(-1)
        self._bars_quick.setPlaceholderText("Quick")
        self._bars_quick.setFixedWidth(78)
        self._bars_quick.setToolTip("Quick-select a common bar count")
        self._bars_quick.currentTextChanged.connect(self._on_quick_bars)
        tb.addWidget(self._bars_quick)

        tb.addWidget(VLine())

        # ── Date-range picker ──────────────────────────────────────────────
        self._date_range_chk = QCheckBox("📅 Date range")
        self._date_range_chk.setToolTip(
            "Switch between 'last N bars' mode and a fixed date range.\n"
            "When checked the From/To calendar pickers are used instead."
        )
        self._date_range_chk.toggled.connect(self._on_date_range_toggled)
        tb.addWidget(self._date_range_chk)

        lbl_from = QLabel("From")
        lbl_from.setStyleSheet(f"color:{TEXT2}; font-size:8pt;")
        lbl_from.setVisible(False)
        tb.addWidget(lbl_from)
        self._lbl_from = lbl_from

        self._from_dt = QDateTimeEdit()
        self._from_dt.setCalendarPopup(True)
        self._from_dt.setDisplayFormat("yyyy-MM-dd HH:mm")
        self._from_dt.setDateTime(
            QDateTime.currentDateTime().addMonths(-3)
        )
        self._from_dt.setFixedWidth(148)
        self._from_dt.setVisible(False)
        self._from_dt.dateTimeChanged.connect(self._on_date_range_changed)
        tb.addWidget(self._from_dt)

        lbl_to = QLabel("To")
        lbl_to.setStyleSheet(f"color:{TEXT2}; font-size:8pt;")
        lbl_to.setVisible(False)
        tb.addWidget(lbl_to)
        self._lbl_to = lbl_to

        self._to_dt = QDateTimeEdit()
        self._to_dt.setCalendarPopup(True)
        self._to_dt.setDisplayFormat("yyyy-MM-dd HH:mm")
        self._to_dt.setDateTime(QDateTime.currentDateTime())
        self._to_dt.setFixedWidth(148)
        self._to_dt.setVisible(False)
        self._to_dt.dateTimeChanged.connect(self._on_date_range_changed)
        tb.addWidget(self._to_dt)

        tb.addWidget(VLine())

        # Refresh + auto
        self._btn_refresh = QPushButton("⟳  Refresh")
        self._btn_refresh.clicked.connect(self._manual_refresh)
        tb.addWidget(self._btn_refresh)

        lbl_auto = QLabel("  Auto")
        lbl_auto.setStyleSheet(f"color:{TEXT2}; font-size:8pt;")
        tb.addWidget(lbl_auto)

        self._auto_combo = QComboBox()
        self._auto_combo.addItems(["Off", "5s", "10s", "30s", "60s"])
        self._auto_combo.currentTextChanged.connect(self._on_auto_refresh_changed)
        self._auto_combo.setFixedWidth(65)
        tb.addWidget(self._auto_combo)

        tb.addWidget(VLine())

        # Cross-TF projection toggle
        self._cross_chk = QCheckBox("Cross-TF Proj.")
        self._cross_chk.setChecked(True)
        self._cross_chk.toggled.connect(self._on_cross_tf_toggle)
        tb.addWidget(self._cross_chk)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

    # ── Central ──────────────────────────────────────────────────────────────

    def _build_central(self) -> None:
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(f"background:{BG0}; border:none;")

        self._panels_container = QWidget()
        self._panels_container.setStyleSheet(f"background:{BG0};")
        self._panels_layout = QVBoxLayout(self._panels_container)
        self._panels_layout.setContentsMargins(6, 6, 6, 6)
        self._panels_layout.setSpacing(6)

        self._scroll.setWidget(self._panels_container)
        self.setCentralWidget(self._scroll)

    # ── Dock ─────────────────────────────────────────────────────────────────

    def _build_filter_dock(self) -> None:
        self._filter_panel = FilterBuilderPanel(self._engine)
        self._filter_panel.filter_applied.connect(self._on_filter_applied)
        self._filter_panel.filters_changed.connect(self._on_filters_changed)

        dock = QDockWidget("  Filter Builder", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setWidget(self._filter_panel)
        dock.setMinimumWidth(400)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self) -> None:
        sb = QStatusBar(self)
        self.setStatusBar(sb)

        # MT5 connection status
        self._dot = StatusDot()
        self._status_conn = QLabel("MT5: Disconnected")
        self._status_conn.setStyleSheet(f"color:{DANGER}; font-weight:600;")
        self._status_sym = QLabel("")

        sb.addWidget(self._dot)
        sb.addWidget(self._status_conn)
        sb.addPermanentWidget(self._status_sym)

        # ── GPU status ──────────────────────────────────────────
        self._gpu_dot = QWidget()
        self._gpu_dot.setFixedSize(10, 10)

        if GPU_STATE["enabled"]:
            gpu_color = SUCCESS
            dev = GPU_STATE["device"]
            mem = GPU_STATE["memory_gb"]
            gpu_text = f"GPU: {dev}" + (f"  {mem} GB" if mem else "")
        else:
            gpu_color = "#6b7280"   # neutral grey → CPU only
            gpu_text  = "GPU: CPU"

        self._gpu_dot.setStyleSheet(f"border-radius:5px; background:{gpu_color};")

        self._gpu_lbl = QLabel(gpu_text)
        self._gpu_lbl.setStyleSheet(f"color:{TEXT2}; font-size:8pt; padding-left:2px;")
        self._gpu_lbl.setToolTip(hardware_report())

        sb.addPermanentWidget(self._gpu_dot)
        sb.addPermanentWidget(self._gpu_lbl)

    # ── MT5 ──────────────────────────────────────────────────────────────────

    def _try_connect(self) -> None:
        ok = self._provider.connect()
        self._dot.set_connected(ok)
        if ok:
            self._status_conn.setText("MT5: Connected")
            self._status_conn.setStyleSheet(f"color:{SUCCESS}; font-weight:600;")
        else:
            self._status_conn.setText("MT5: Demo mode")
            self._status_conn.setStyleSheet(f"color:{WARNING}; font-weight:600;")

    # ── Panels ────────────────────────────────────────────────────────────────

    def _rebuild_panels(self) -> None:
        active_tfs = self._tf_selector.active_timeframes()

        for tf in list(self._panels.keys()):
            if tf not in active_tfs:
                w = self._panels.pop(tf)
                self._panels_layout.removeWidget(w)
                w.deleteLater()

        for tf in active_tfs:
            if tf not in self._panels:
                panel = ChartPanel(tf)
                panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                panel.setMinimumHeight(240)
                self._panels[tf] = panel
                self._panels_layout.addWidget(panel)

        ordered = [tf for tf in TF_ORDER if tf in self._panels]
        for i, tf in enumerate(ordered):
            self._panels_layout.insertWidget(i, self._panels[tf])

        if len(ordered) > 1:
            first = self._panels[ordered[0]]
            for tf in ordered[1:]:
                self._panels[tf].link_x(first)

        self._manual_refresh()

    # ── Data ─────────────────────────────────────────────────────────────────

    def _manual_refresh(self) -> None:
        self._start_fetch(force=True)

    def _auto_refresh(self) -> None:
        self._start_fetch(force=False)

    # ── Async fetch ───────────────────────────────────────────────────────────

    def _start_fetch(self, force: bool = False) -> None:
        """Launch parallel OHLCV fetch; UI stays responsive."""
        tfs = list(self._panels.keys())
        if not tfs:
            return

        # Prevent double-launch
        if hasattr(self, "_fetch_worker") and self._fetch_worker.isRunning():
            return

        # Validate date range before launching
        from_date: Optional[pd.Timestamp] = None
        to_date:   Optional[pd.Timestamp] = None
        if self._use_date_range:
            from_dt_q = self._from_dt.dateTime()
            to_dt_q   = self._to_dt.dateTime()
            if from_dt_q >= to_dt_q:
                QMessageBox.warning(
                    self, "Invalid date range",
                    "'From' must be earlier than 'To'."
                )
                return
            from_date = pd.Timestamp(from_dt_q.toPyDateTime())
            to_date   = pd.Timestamp(to_dt_q.toPyDateTime())

        self._btn_refresh.setEnabled(False)
        self._btn_refresh.setText("⟳ Loading…")
        self._status_sym.setText(f"Fetching {len(tfs)} timeframe(s)…")

        self._fetch_worker = DataFetchWorker(
            self._provider, self._symbol, tfs,
            nb_bars=self._nb_bars, force=force,
            from_date=from_date, to_date=to_date,
            parent=self,
        )
        self._fetch_worker.data_ready.connect(self._on_data_ready)
        self._fetch_worker.start()

    def _on_data_ready(self, loaded: dict) -> None:
        """Called on main thread once all OHLCV is ready."""
        self._data = loaded

        # Render candles immediately
        for tf, panel in self._panels.items():
            if tf in loaded:
                panel.set_data(loaded[tf])

        if self._use_date_range:
            from_str = self._from_dt.dateTime().toString("yyyy-MM-dd")
            to_str   = self._to_dt.dateTime().toString("yyyy-MM-dd")
            # show actual bar count from first loaded TF
            first_df = next(iter(loaded.values()), None)
            n_bars   = len(first_df) if first_df is not None else 0
            self._status_sym.setText(
                f"{self._symbol}  {from_str} → {to_str}  ({n_bars:,} bars)  ×{len(loaded)} TF"
            )
        else:
            self._status_sym.setText(
                f"{self._symbol}   {self._nb_bars:,} bars   ×{len(loaded)} TF"
            )

        self._btn_refresh.setEnabled(True)
        self._btn_refresh.setText("⟳  Refresh")

        # Launch filter computation in background only if zones are active
        if self._show_zones and self._engine.filters:
            self._start_filter_worker()

    # ── Async filter ──────────────────────────────────────────────────────────

    def _start_filter_worker(self) -> None:
        if hasattr(self, "_filter_worker") and self._filter_worker.isRunning():
            self._filter_worker.quit()

        self._filter_worker = FilterWorker(
            self._engine, self._data, list(self._panels.keys()), parent=self
        )
        self._filter_worker.segments_ready.connect(self._on_segments_ready)
        self._filter_worker.start()

    def _on_segments_ready(self, tf: str, segs: list) -> None:
        """Apply zone segments as each TF finishes (incremental update)."""
        if tf in self._panels:
            self._panels[tf].set_filter_segments(segs, show=True)

    def _apply_zones_to_panel(self, tf: str, panel: ChartPanel) -> None:
        """Synchronous zone apply (used right after Apply Filter click)."""
        if self._show_zones and self._data:
            try:
                segs = self._engine.get_segments(self._data, tf)
                panel.set_filter_segments(segs, show=True)
            except Exception:
                panel.set_filter_segments([], show=False)
        else:
            panel.set_filter_segments([], show=False)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_date_range_toggled(self, checked: bool) -> None:
        """Show/hide date pickers; show/hide bars spinbox."""
        self._use_date_range = checked

        # Bars controls ↔ date pickers
        self._lbl_bars.setVisible(not checked)
        self._bars_spin.setVisible(not checked)
        self._bars_quick.setVisible(not checked)
        self._lbl_from.setVisible(checked)
        self._from_dt.setVisible(checked)
        self._lbl_to.setVisible(checked)
        self._to_dt.setVisible(checked)

        # Auto-refresh doesn't make sense for fixed date ranges
        if checked:
            self._auto_combo.setCurrentText("Off")
            self._auto_combo.setEnabled(False)
        else:
            self._auto_combo.setEnabled(True)

        self._manual_refresh()

    def _on_date_range_changed(self) -> None:
        """Called when either QDateTimeEdit value changes."""
        if self._use_date_range:
            self._manual_refresh()

    def _on_symbol_changed(self) -> None:
        self._symbol = self._symbol_edit.text().strip().upper()
        self._provider.invalidate_cache(self._symbol)
        self._manual_refresh()

    def _on_bars_changed(self) -> None:
        self._nb_bars = self._bars_spin.value()
        self._manual_refresh()

    def _on_quick_bars(self, text: str) -> None:
        try:
            v = int(text.replace(" ", "").replace(" ", ""))
            self._bars_spin.setValue(v)
            self._nb_bars = v
            self._bars_quick.setCurrentIndex(-1)   # reset placeholder
            self._manual_refresh()
        except ValueError:
            pass

    def _on_filter_applied(self) -> None:
        """Called when user clicks Apply Filter in the builder."""
        self._show_zones = True
        if self._data:
            # Run filter computation in background → zones appear TF-by-TF
            self._start_filter_worker()
        else:
            # No data yet — fetch first (filter will run after)
            self._start_fetch(force=True)

    def _on_filters_changed(self) -> None:
        # Filter list changed but not yet applied — keep current display
        pass

    def _on_cross_tf_toggle(self, checked: bool) -> None:
        self._engine.set_cross_tf_projection(checked)
        if self._show_zones:
            for tf, panel in self._panels.items():
                self._apply_zones_to_panel(tf, panel)

    def _on_auto_refresh_changed(self, value: str) -> None:
        self._refresh_timer.stop()
        intervals = {"5s": 5000, "10s": 10000, "30s": 30000, "60s": 60000}
        if value in intervals:
            self._refresh_timer.start(intervals[value])

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._refresh_timer.stop()
        self._provider.disconnect()
        super().closeEvent(event)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("MT5 Chart Analyzer")
    app.setFont(QFont("Segoe UI", 9))
    apply_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
