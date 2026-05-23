"""
Main window: modern dark UI, multi-TF panels, before/after filter toggle.
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QIcon, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
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

        self._provider = DataProvider()
        self._engine   = FilterEngine(self._provider)

        self._symbol        = "EURUSD"
        self._nb_bars       = 300
        self._show_zones = False   # zones hidden until filter applied
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
        lbl_bars = QLabel("  Bars")
        lbl_bars.setStyleSheet(f"color:{TEXT2}; font-size:8pt;")
        tb.addWidget(lbl_bars)

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

        # Refresh + auto
        btn_refresh = QPushButton("⟳  Refresh")
        btn_refresh.clicked.connect(self._manual_refresh)
        tb.addWidget(btn_refresh)

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

        self._dot = StatusDot()
        self._status_conn = QLabel("MT5: Disconnected")
        self._status_conn.setStyleSheet(f"color:{DANGER}; font-weight:600;")
        self._status_sym = QLabel("")

        sb.addWidget(self._dot)
        sb.addWidget(self._status_conn)
        sb.addPermanentWidget(self._status_sym)

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
        self._fetch_and_render(force=True)

    def _auto_refresh(self) -> None:
        self._fetch_and_render(force=False)

    def _fetch_and_render(self, force: bool = False) -> None:
        symbol = self._symbol
        nb     = self._nb_bars
        loaded: Dict[str, object] = {}

        for tf in list(self._panels.keys()):
            try:
                df = self._provider.get_ohlcv(symbol, tf, nb, force_refresh=force)
                loaded[tf] = df
            except Exception:
                pass

        self._data = loaded

        for tf, panel in self._panels.items():
            if tf not in loaded:
                continue
            panel.set_data(loaded[tf])
            self._apply_zones_to_panel(tf, panel)

        self._status_sym.setText(f"{symbol}   {nb} bars")

    def _apply_zones_to_panel(self, tf: str, panel: ChartPanel) -> None:
        if self._show_zones and self._data:
            try:
                segs = self._engine.get_segments(self._data, tf)
                panel.set_filter_segments(segs, show=True)
            except Exception:
                panel.set_filter_segments([], show=False)
        else:
            panel.set_filter_segments([], show=False)

    # ── Slots ────────────────────────────────────────────────────────────────

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
        for tf, panel in self._panels.items():
            self._apply_zones_to_panel(tf, panel)

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
