"""
Chart widget — trading-style navigation + hover crosshair + info HUD.

Navigation:
  • Left drag     → pan
  • Scroll wheel  → zoom X around cursor (Y auto-adapts)
  • Double-click  → reset to last 200 bars

Hover:
  • Over a candle → HUD shows O/H/L/C + volume + timestamp
  • Over a zone   → HUD appends zone start/end + status
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QCursor, QFont, QPainter, QPen, QPicture, QBrush
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from filter_engine import FilterSegment

# ── Palette ────────────────────────────────────────────────────────────────────
BG_PANEL = QColor(13, 13, 26)
BG_PLOT  = QColor(16, 16, 28)
COL_BULL = QColor(16, 185, 129)
COL_BEAR = QColor(239, 68, 68)
COL_WICK = QColor(110, 110, 140)
COL_TEXT = QColor(130, 130, 160)

ZONE_VALID   = QColor(16, 185, 129, 38)
ZONE_INVALID = QColor(239, 68,  68,  38)

DEFAULT_VISIBLE = 200
MIN_VISIBLE     = 10


# ── Candlestick item ────────────────────────────────────────────────────────────

class CandlestickItem(pg.GraphicsObject):
    def __init__(self) -> None:
        super().__init__()
        self._pic: Optional[QPicture] = None
        self._df:  Optional[pd.DataFrame] = None
        self._bounds = QRectF()

    def set_data(self, df: pd.DataFrame) -> None:
        self._df = df.reset_index(drop=True)
        self._redraw()
        self.informViewBoundsChanged()

    def _redraw(self) -> None:
        pic = QPicture()
        if self._df is None or len(self._df) == 0:
            self._pic = pic
            return

        df     = self._df
        n      = len(df)
        opens  = df["open"].values.astype(float)
        highs  = df["high"].values.astype(float)
        lows   = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)

        self._bounds = QRectF(
            -0.5, float(lows.min()),
            n, float(highs.max() - lows.min()) or 1e-9,
        )

        p = QPainter(pic)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        wick_pen = QPen(COL_WICK); wick_pen.setWidth(0)
        w = 0.38

        for i in range(n):
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            color = COL_BULL if c >= o else COL_BEAR
            p.setPen(wick_pen)
            p.drawLine(pg.Point(i, l), pg.Point(i, h))
            bpen = QPen(color); bpen.setWidth(0)
            p.setPen(bpen); p.setBrush(QBrush(color))
            top = max(o, c); bot = min(o, c)
            p.drawRect(QRectF(i - w, bot, 2 * w, max(top - bot, 1e-9)))

        p.end()
        self._pic = pic

    def paint(self, p: QPainter, *args) -> None:
        if self._pic:
            self._pic.play(p)

    def boundingRect(self) -> QRectF:
        return self._bounds

    def ohlc_in_range(self, x0: float, x1: float) -> Optional[tuple]:
        if self._df is None or len(self._df) == 0:
            return None
        i0 = max(0, int(np.floor(x0)))
        i1 = min(len(self._df) - 1, int(np.ceil(x1)))
        if i0 > i1:
            return None
        sub = self._df.iloc[i0 : i1 + 1]
        return float(sub["low"].min()), float(sub["high"].max())


# ── Filter zone item ────────────────────────────────────────────────────────────

class FilterZoneItem(pg.GraphicsObject):
    def __init__(self) -> None:
        super().__init__()
        self._pic: Optional[QPicture] = None
        self._bounds = QRectF()

    def set_zones(
        self, segments: List[FilterSegment],
        df: pd.DataFrame, y_min: float, y_max: float,
    ) -> None:
        self._redraw(segments, df, y_min, y_max)
        self.informViewBoundsChanged()

    def _ts_to_idx(self, ts: pd.Timestamp, times: np.ndarray) -> float:
        return float(np.searchsorted(times, np.datetime64(ts, "ns"), side="left"))

    def _redraw(
        self, segments: List[FilterSegment],
        df: pd.DataFrame, y_min: float, y_max: float,
    ) -> None:
        pic = QPicture()
        if not segments or df is None or len(df) == 0:
            self._pic = pic
            return
        times  = df["time"].values.astype("datetime64[ns]")
        n      = len(df)
        height = (y_max - y_min) or 1.0
        self._bounds = QRectF(-0.5, y_min, n, height)
        p = QPainter(pic)
        p.setPen(QPen(Qt.PenStyle.NoPen))
        for seg in segments:
            p.setBrush(QBrush(ZONE_VALID if seg.valid else ZONE_INVALID))
            x0 = self._ts_to_idx(seg.t_start, times) - 0.5
            x1 = self._ts_to_idx(seg.t_end,   times) - 0.5
            p.drawRect(QRectF(x0, y_min, x1 - x0, height))
        p.end()
        self._pic = pic

    def paint(self, p: QPainter, *args) -> None:
        if self._pic:
            self._pic.play(p)

    def boundingRect(self) -> QRectF:
        return self._bounds


# ── Time axis ──────────────────────────────────────────────────────────────────

class TimeAxis(pg.AxisItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._times: Optional[np.ndarray] = None

    def set_times(self, times: np.ndarray) -> None:
        self._times = times

    def tickStrings(self, values, scale, spacing):
        if self._times is None:
            return [str(int(v)) for v in values]
        out = []
        for v in values:
            idx = int(round(v))
            if 0 <= idx < len(self._times):
                out.append(pd.Timestamp(self._times[idx]).strftime("%H:%M\n%m/%d"))
            else:
                out.append("")
        return out


# ── Trading ViewBox ─────────────────────────────────────────────────────────────

class TradingViewBox(pg.ViewBox):
    """Left-drag pan · scroll zoom X · double-click reset."""

    def __init__(self, panel: "ChartPanel", **kwargs):
        super().__init__(**kwargs)
        self._panel = panel
        self.setMouseMode(pg.ViewBox.PanMode)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def wheelEvent(self, ev, axis=None):
        try:
            delta = ev.angleDelta().y()
        except AttributeError:
            delta = ev.delta()
        if delta == 0:
            ev.ignore(); return

        factor  = 0.80 if delta > 0 else 1.25
        x_range = self.viewRange()[0]
        width   = x_range[1] - x_range[0]

        try:
            sp = ev.position()
        except AttributeError:
            sp = ev.pos()
        cx = self.mapSceneToView(sp).x()

        frac      = (cx - x_range[0]) / width if width > 0 else 0.5
        new_width = width * factor
        new_x0    = cx - frac * new_width
        new_x1    = new_x0 + new_width

        if self._panel._df is not None:
            n = len(self._panel._df)
            if new_width < MIN_VISIBLE:
                ev.accept(); return
            new_x0 = max(-0.5, new_x0)
            new_x1 = min(n - 0.5, new_x1)

        self.setXRange(new_x0, new_x1, padding=0)
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        if self._panel._df is not None:
            n = len(self._panel._df)
            v = min(DEFAULT_VISIBLE, n)
            self.setXRange(n - v - 0.5, n - 0.5, padding=0)
        ev.accept()


# ── Info HUD label ─────────────────────────────────────────────────────────────

HUD_STYLE = """
    QLabel {
        background: rgba(13, 13, 26, 210);
        color: #e2e2f0;
        border: 1px solid #2a2a45;
        border-radius: 6px;
        padding: 6px 11px;
        font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
        font-size: 8pt;
        line-height: 160%;
    }
"""


# ── Chart panel ────────────────────────────────────────────────────────────────

class ChartPanel(QWidget):
    def __init__(self, timeframe: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.timeframe  = timeframe
        self._df:       Optional[pd.DataFrame]   = None
        self._segments: List[FilterSegment]      = []

        self.setStyleSheet("background: rgb(13,13,26); border-radius:8px;")
        self._build_ui()

    # ── UI setup ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(BG_PANEL)
        layout.addWidget(self._glw)

        self._time_axis  = TimeAxis(orientation="bottom")
        self._vb         = TradingViewBox(panel=self)
        self._price_plot = self._glw.addPlot(
            row=0, col=0,
            axisItems={"bottom": self._time_axis},
            viewBox=self._vb,
        )
        self._style_plot(self._price_plot, self.timeframe)

        self._candle_item = CandlestickItem()
        self._price_plot.addItem(self._candle_item)

        self._zone_item = FilterZoneItem()
        self._price_plot.addItem(self._zone_item)

        self._overlay_curves: Dict[str, pg.PlotDataItem] = {}
        self._sub_plots:  Dict[str, pg.PlotItem]     = {}
        self._sub_curves: Dict[str, pg.PlotDataItem] = {}

        self._glw.ci.layout.setRowStretchFactor(0, 4)

        # ── Crosshair ─────────────────────────────────────────
        ch_pen = pg.mkPen("#4f6ef7", width=1, style=Qt.PenStyle.DashLine)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=ch_pen)
        self._hline = pg.InfiniteLine(angle=0,  movable=False, pen=ch_pen)
        self._price_plot.addItem(self._vline, ignoreBounds=True)
        self._price_plot.addItem(self._hline, ignoreBounds=True)
        self._vline.hide(); self._hline.hide()

        # ── HUD label (overlay on top of GLW) ─────────────────
        self._hud = QLabel("", self._glw)
        self._hud.setStyleSheet(HUD_STYLE)
        self._hud.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._hud.hide()

        # ── Mouse tracking ────────────────────────────────────
        self._proxy = pg.SignalProxy(
            self._price_plot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved,
        )

        # Adaptive Y on every X change
        self._vb.sigXRangeChanged.connect(self._adapt_y)

    def _style_plot(self, plot: pg.PlotItem, title: str) -> None:
        lp = pg.mkPen(COL_TEXT)
        ap = pg.mkPen(QColor(40, 40, 65))
        for ax in ("left", "right", "bottom", "top"):
            a = plot.getAxis(ax)
            a.setPen(ap); a.setTextPen(lp)
            a.setStyle(tickFont=QFont("Segoe UI", 7))
        plot.setTitle(
            f"<span style='color:#7070a0; font-size:8pt; font-weight:700;'>{title}</span>"
        )
        plot.showGrid(x=True, y=True, alpha=0.10)
        plot.setMenuEnabled(False); plot.hideButtons()
        vb = plot.getViewBox()
        vb.setBackgroundColor(BG_PLOT)
        vb.enableAutoRange(axis="y", enable=False)
        vb.setAutoVisible(y=False)

    # ── Data ──────────────────────────────────────────────────────────────────

    def set_data(self, df: pd.DataFrame) -> None:
        self._df = df.reset_index(drop=True)
        self._candle_item.set_data(self._df)
        self._time_axis.set_times(self._df["time"].values)
        n = len(self._df)
        v = min(DEFAULT_VISIBLE, n)
        self._vb.setXRange(n - v - 0.5, n - 0.5, padding=0)
        self._adapt_y()
        self._refresh_zones()

    def set_filter_segments(
        self, segments: List[FilterSegment], show: bool = True
    ) -> None:
        self._segments = segments
        self._refresh_zones(show=show)

    def _refresh_zones(self, show: bool = True) -> None:
        if self._df is None or len(self._df) == 0:
            return
        lo  = float(self._df["low"].min())
        hi  = float(self._df["high"].max())
        pad = (hi - lo) * 0.10
        if show and self._segments:
            self._zone_item.set_zones(self._segments, self._df, lo - pad, hi + pad)
        else:
            self._zone_item.set_zones([], self._df, lo - pad, hi + pad)

    # ── Adaptive Y ────────────────────────────────────────────────────────────

    def _adapt_y(self) -> None:
        if self._df is None or len(self._df) == 0:
            return
        x0, x1 = self._vb.viewRange()[0]
        res = self._candle_item.ohlc_in_range(x0, x1)
        if res is None:
            return
        lo, hi = res
        pad = (hi - lo) * 0.07 or abs(lo) * 0.01 or 0.001
        self._vb.setYRange(lo - pad, hi + pad, padding=0)

    # ── Hover / crosshair ─────────────────────────────────────────────────────

    def _on_mouse_moved(self, event) -> None:
        pos = event[0]

        # Check if cursor is inside the data viewport (not on axes/title)
        if not self._vb.sceneBoundingRect().contains(pos):
            self._vline.hide()
            self._hline.hide()
            self._hud.hide()
            return

        mp = self._vb.mapSceneToView(pos)
        x, y = mp.x(), mp.y()

        # Crosshair
        self._vline.setPos(x); self._vline.show()
        self._hline.setPos(y); self._hline.show()

        if self._df is None or len(self._df) == 0:
            return

        # Bar info
        idx = int(round(x))
        idx = max(0, min(len(self._df) - 1, idx))
        bar = self._df.iloc[idx]
        ts  = pd.Timestamp(bar["time"])

        bull      = bar["close"] >= bar["open"]
        arrow     = "▲" if bull else "▼"
        col_arrow = "#10b981" if bull else "#ef4444"
        t_str     = ts.strftime("%Y-%m-%d  %H:%M")
        pips      = abs(bar["close"] - bar["open"])

        # Build rich-text HUD
        html = (
            f"<span style='color:#9090b0'>{self.timeframe}</span>"
            f"&nbsp;&nbsp;"
            f"<span style='color:#c8c8d2'>{t_str}</span>"
            f"&nbsp;&nbsp;"
            f"<span style='color:{col_arrow}; font-weight:700'>{arrow}</span>"
            f"<br>"
            f"<span style='color:#9090b0'>O</span> <b>{bar['open']:.5f}</b>"
            f"&nbsp;&nbsp;"
            f"<span style='color:#10b981'>H</span> <b>{bar['high']:.5f}</b>"
            f"&nbsp;&nbsp;"
            f"<span style='color:#ef4444'>L</span> <b>{bar['low']:.5f}</b>"
            f"&nbsp;&nbsp;"
            f"<span style='color:#9090b0'>C</span> <b>{bar['close']:.5f}</b>"
            f"&nbsp;&nbsp;"
            f"<span style='color:#9090b0'>Vol</span> {int(bar['volume']):,}"
        )

        # Zone info
        zone = self._zone_at(ts)
        if zone:
            status_col = "#10b981" if zone.valid else "#ef4444"
            status_txt = "✔ Valide" if zone.valid else "✘ Invalide"
            t0 = zone.t_start.strftime("%Y-%m-%d  %H:%M")
            t1 = zone.t_end.strftime("%Y-%m-%d  %H:%M")
            html += (
                f"<br>"
                f"<span style='color:#4f6ef7'>Zone</span>"
                f"&nbsp;&nbsp;"
                f"{t0}"
                f"&nbsp;<span style='color:#9090b0'>→</span>&nbsp;"
                f"{t1}"
                f"&nbsp;&nbsp;"
                f"<span style='color:{status_col}; font-weight:700'>{status_txt}</span>"
            )

        self._hud.setText(html)
        self._hud.adjustSize()

        # Position HUD: top-left by default; shift right if cursor is too close
        hud_w = self._hud.width()
        hud_h = self._hud.height()
        glw_w = self._glw.width()
        margin = 12

        # Convert scene pos to widget pos
        scene_x = pos.x()
        widget_x = self._glw.mapFromGlobal(
            self._glw.mapToGlobal(
                self._glw.rect().topLeft()
            )
        ).x()

        # Always top-left corner of the chart, safe from cursor overlap
        hud_x = margin
        hud_y = margin

        self._hud.move(hud_x, hud_y)
        self._hud.show()
        self._hud.raise_()

    def _zone_at(self, ts: pd.Timestamp) -> Optional[FilterSegment]:
        for seg in self._segments:
            if seg.t_start <= ts < seg.t_end:
                return seg
        return None

    # ── Overlays / sub-panes ──────────────────────────────────────────────────

    def add_overlay(
        self, name: str, values: np.ndarray,
        color: str = "#f59e0b", width: int = 1,
    ) -> None:
        x = np.arange(len(values), dtype=float)
        if name in self._overlay_curves:
            self._overlay_curves[name].setData(x=x, y=values)
        else:
            self._overlay_curves[name] = self._price_plot.plot(
                x=x, y=values, pen=pg.mkPen(color, width=width), name=name,
            )

    def remove_overlay(self, name: str) -> None:
        if name in self._overlay_curves:
            self._price_plot.removeItem(self._overlay_curves.pop(name))

    def add_sub_indicator(
        self, name: str, values: np.ndarray, color: str = "#818cf8",
    ) -> None:
        if name not in self._sub_plots:
            row  = self._glw.ci.layout.rowCount()
            plot = self._glw.addPlot(row=row, col=0)
            self._style_plot(plot, name)
            plot.setXLink(self._price_plot)
            self._glw.ci.layout.setRowStretchFactor(row, 1)
            if "RSI" in name.upper():
                for lvl, lc in [(30, "#10b981"), (50, "#6b7280"), (70, "#ef4444")]:
                    plot.addLine(y=lvl, pen=pg.mkPen(lc, style=Qt.PenStyle.DashLine))
            self._sub_plots[name] = plot
        x = np.arange(len(values), dtype=float)
        if name in self._sub_curves:
            self._sub_curves[name].setData(x=x, y=values)
        else:
            self._sub_curves[name] = self._sub_plots[name].plot(
                x=x, y=values, pen=pg.mkPen(color, width=1), name=name,
            )

    def remove_sub_indicator(self, name: str) -> None:
        if name in self._sub_plots:
            self._glw.removeItem(self._sub_plots.pop(name))
            self._sub_curves.pop(name, None)

    # ── X sync ────────────────────────────────────────────────────────────────

    def price_view(self) -> pg.ViewBox:
        return self._vb

    def link_x(self, other: "ChartPanel") -> None:
        self._price_plot.setXLink(other._price_plot)
        for sp in self._sub_plots.values():
            sp.setXLink(other._price_plot)
