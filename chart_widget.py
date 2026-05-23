"""
Chart widget — visible-only rendering + LOD + hover HUD + crosshair.

Rendering strategy
──────────────────
paint() draws ONLY the bars currently in view (no QPicture pre-bake of 10 k bars).
• ≤ 1500 visible bars → full Japanese candlesticks
  Bodies: one QPainterPath per colour → single fillPath call each
  Wicks : list[QLineF] → single drawLines call
• > 1500 visible bars → LOD: coloured vertical OHLC lines, no bodies
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QLineF, QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush, QColor, QCursor, QFont, QPainter,
    QPainterPath, QPen, QPicture,
)
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from filter_engine import FilterSegment

# ── Palette ───────────────────────────────────────────────────────────────────
BG_PANEL = QColor(13, 13, 26)
BG_PLOT  = QColor(16, 16, 28)
COL_BULL = QColor(16, 185, 129)
COL_BEAR = QColor(239, 68, 68)
COL_WICK = QColor(100, 100, 130)
COL_TEXT = QColor(130, 130, 160)

ZONE_VALID   = QColor(16, 185, 129, 40)
ZONE_INVALID = QColor(239, 68,  68,  40)

DEFAULT_VISIBLE = 200
MIN_VISIBLE     = 10
LOD_THRESHOLD   = 1500   # bars above which we switch to simple lines


# ── Candlestick item (visible-range paint, LOD) ────────────────────────────────

class CandlestickItem(pg.GraphicsObject):
    def __init__(self) -> None:
        super().__init__()
        self._df: Optional[pd.DataFrame] = None
        self._bounds = QRectF()

        # Pre-computed numpy columns (updated when df changes, not on every paint)
        self._xs:     Optional[np.ndarray] = None
        self._opens:  Optional[np.ndarray] = None
        self._highs:  Optional[np.ndarray] = None
        self._lows:   Optional[np.ndarray] = None
        self._closes: Optional[np.ndarray] = None

    def set_data(self, df: pd.DataFrame) -> None:
        self._df     = df.reset_index(drop=True)
        n            = len(self._df)
        self._xs     = np.arange(n, dtype=np.float64)
        self._opens  = self._df["open"].values.astype(np.float64)
        self._highs  = self._df["high"].values.astype(np.float64)
        self._lows   = self._df["low"].values.astype(np.float64)
        self._closes = self._df["close"].values.astype(np.float64)
        self._bounds = QRectF(
            -0.5, float(self._lows.min()),
            n, float(self._highs.max() - self._lows.min()) or 1e-9,
        )
        self.informViewBoundsChanged()
        self.update()

    # ── paint: visible-range only ──────────────────────────────────────────────

    def paint(self, p: QPainter, *args) -> None:
        if self._df is None or len(self._df) == 0:
            return

        # Determine visible bar range from the ViewBox
        vb = self.getViewBox()
        if vb is not None:
            x0, x1 = vb.viewRange()[0]
        else:
            x0, x1 = -0.5, len(self._df) - 0.5

        n  = len(self._df)
        i0 = max(0, int(np.floor(x0)))
        i1 = min(n - 1, int(np.ceil(x1)))
        if i0 > i1:
            return

        visible = i1 - i0 + 1
        if visible > LOD_THRESHOLD:
            self._paint_lod(p, i0, i1)
        else:
            self._paint_candles(p, i0, i1)

    # ── Full candlesticks ──────────────────────────────────────────────────────

    def _paint_candles(self, p: QPainter, i0: int, i1: int) -> None:
        xs     = self._xs[i0:i1+1]
        opens  = self._opens[i0:i1+1]
        highs  = self._highs[i0:i1+1]
        lows   = self._lows[i0:i1+1]
        closes = self._closes[i0:i1+1]

        bull = closes >= opens
        bear = ~bull

        body_top = np.maximum(opens, closes)
        body_bot = np.minimum(opens, closes)
        body_h   = np.maximum(body_top - body_bot, 1e-9)
        w = 0.38

        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # ── Wicks (one drawLines call) ──────────────────────────
        wick_pen = QPen(COL_WICK); wick_pen.setWidth(0)
        p.setPen(wick_pen)
        wick_lines = [
            QLineF(float(xs[j]), float(lows[j]), float(xs[j]), float(highs[j]))
            for j in range(len(xs))
        ]
        p.drawLines(wick_lines)

        # ── Bull bodies ─────────────────────────────────────────
        if bull.any():
            path = QPainterPath()
            for j in np.where(bull)[0]:
                path.addRect(QRectF(xs[j] - w, body_bot[j], 2 * w, body_h[j]))
            p.fillPath(path, QBrush(COL_BULL))
            p.setPen(QPen(COL_BULL, 0))
            p.drawPath(path)

        # ── Bear bodies ─────────────────────────────────────────
        if bear.any():
            path = QPainterPath()
            for j in np.where(bear)[0]:
                path.addRect(QRectF(xs[j] - w, body_bot[j], 2 * w, body_h[j]))
            p.fillPath(path, QBrush(COL_BEAR))
            p.setPen(QPen(COL_BEAR, 0))
            p.drawPath(path)

    # ── LOD: simple OHLC lines ─────────────────────────────────────────────────

    def _paint_lod(self, p: QPainter, i0: int, i1: int) -> None:
        xs     = self._xs[i0:i1+1]
        opens  = self._opens[i0:i1+1]
        highs  = self._highs[i0:i1+1]
        lows   = self._lows[i0:i1+1]
        closes = self._closes[i0:i1+1]

        bull = closes >= opens

        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Bull bars: high-low line
        bull_pen = QPen(COL_BULL); bull_pen.setWidth(0)
        p.setPen(bull_pen)
        bull_lines = [
            QLineF(float(xs[j]), float(lows[j]), float(xs[j]), float(highs[j]))
            for j in np.where(bull)[0]
        ]
        if bull_lines:
            p.drawLines(bull_lines)

        # Bear bars
        bear_pen = QPen(COL_BEAR); bear_pen.setWidth(0)
        p.setPen(bear_pen)
        bear_lines = [
            QLineF(float(xs[j]), float(lows[j]), float(xs[j]), float(highs[j]))
            for j in np.where(~bull)[0]
        ]
        if bear_lines:
            p.drawLines(bear_lines)

    # ── Bounds ────────────────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return self._bounds

    def ohlc_in_range(self, x0: float, x1: float) -> Optional[tuple]:
        if self._df is None or len(self._df) == 0:
            return None
        i0 = max(0, int(np.floor(x0)))
        i1 = min(len(self._df) - 1, int(np.ceil(x1)))
        if i0 > i1:
            return None
        return float(self._lows[i0:i1+1].min()), float(self._highs[i0:i1+1].max())


# ── Filter zone item ───────────────────────────────────────────────────────────

class FilterZoneItem(pg.GraphicsObject):
    def __init__(self) -> None:
        super().__init__()
        self._pic: Optional[QPicture] = None
        self._bounds = QRectF()

    def set_zones(
        self, segments: List[FilterSegment],
        df: pd.DataFrame, y_min: float, y_max: float,
    ) -> None:
        self._bake(segments, df, y_min, y_max)
        self.informViewBoundsChanged()
        self.update()

    def _ts_to_idx(self, ts: pd.Timestamp, times: np.ndarray) -> float:
        return float(np.searchsorted(times, np.datetime64(ts, "ns"), side="left"))

    def _bake(
        self, segments: List[FilterSegment],
        df: pd.DataFrame, y_min: float, y_max: float,
    ) -> None:
        pic = QPicture()
        if not segments or df is None or len(df) == 0:
            self._pic = pic; return
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

    def set_times(self, t: np.ndarray) -> None:
        self._times = t

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


# ── Trading ViewBox ────────────────────────────────────────────────────────────

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
        x0, x1 = self.viewRange()[0]
        width   = x1 - x0
        try:
            sp = ev.position()
        except AttributeError:
            sp = ev.pos()
        cx    = self.mapSceneToView(sp).x()
        frac  = (cx - x0) / width if width > 0 else 0.5
        nw    = width * factor
        nx0   = cx - frac * nw
        nx1   = nx0 + nw

        if self._panel._df is not None:
            n = len(self._panel._df)
            if nw < MIN_VISIBLE:
                ev.accept(); return
            nx0 = max(-0.5, nx0)
            nx1 = min(n - 0.5, nx1)

        self.setXRange(nx0, nx1, padding=0)
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        if self._panel._df is not None:
            n = len(self._panel._df)
            v = min(DEFAULT_VISIBLE, n)
            self.setXRange(n - v - 0.5, n - 0.5, padding=0)
        ev.accept()


# ── HUD style ─────────────────────────────────────────────────────────────────

HUD_STYLE = """
QLabel {
    background: rgba(13,13,26,215);
    color: #e2e2f0;
    border: 1px solid #2a2a45;
    border-radius: 6px;
    padding: 6px 12px;
    font-family: 'Cascadia Code','Consolas','Courier New',monospace;
    font-size: 8pt;
    line-height: 155%;
}
"""


# ── Chart panel ────────────────────────────────────────────────────────────────

class ChartPanel(QWidget):
    def __init__(self, timeframe: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.timeframe  = timeframe
        self._df:       Optional[pd.DataFrame] = None
        self._segments: List[FilterSegment]    = []
        self.setStyleSheet("background:rgb(13,13,26); border-radius:8px;")
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(BG_PANEL)
        lay.addWidget(self._glw)

        self._tax = TimeAxis(orientation="bottom")
        self._vb  = TradingViewBox(panel=self)
        self._pp  = self._glw.addPlot(
            row=0, col=0, axisItems={"bottom": self._tax}, viewBox=self._vb,
        )
        self._style_plot(self._pp, self.timeframe)

        self._ci = CandlestickItem()
        self._pp.addItem(self._ci)
        self._zi = FilterZoneItem()
        self._pp.addItem(self._zi)

        self._overlay_curves: Dict[str, pg.PlotDataItem] = {}
        self._sub_plots:  Dict[str, pg.PlotItem]     = {}
        self._sub_curves: Dict[str, pg.PlotDataItem] = {}

        self._glw.ci.layout.setRowStretchFactor(0, 4)

        # Crosshair
        ch = pg.mkPen("#4f6ef7", width=1, style=Qt.PenStyle.DashLine)
        self._vl = pg.InfiniteLine(angle=90, movable=False, pen=ch)
        self._hl = pg.InfiniteLine(angle=0,  movable=False, pen=ch)
        self._pp.addItem(self._vl, ignoreBounds=True)
        self._pp.addItem(self._hl, ignoreBounds=True)
        self._vl.hide(); self._hl.hide()

        # HUD
        self._hud = QLabel("", self._glw)
        self._hud.setStyleSheet(HUD_STYLE)
        self._hud.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._hud.hide()

        # Mouse proxy (rate-limited to 60 Hz)
        self._proxy = pg.SignalProxy(
            self._pp.scene().sigMouseMoved,
            rateLimit=60, slot=self._on_mouse,
        )

        self._vb.sigXRangeChanged.connect(self._adapt_y)

    def _style_plot(self, plot: pg.PlotItem, title: str) -> None:
        lp = pg.mkPen(COL_TEXT)
        ap = pg.mkPen(QColor(40, 40, 65))
        for ax in ("left", "right", "bottom", "top"):
            a = plot.getAxis(ax)
            a.setPen(ap); a.setTextPen(lp)
            a.setStyle(tickFont=QFont("Segoe UI", 7))
        plot.setTitle(
            f"<span style='color:#7070a0;font-size:8pt;font-weight:700;'>{title}</span>"
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
        self._ci.set_data(self._df)
        self._tax.set_times(self._df["time"].values)
        n = len(self._df)
        v = min(DEFAULT_VISIBLE, n)
        self._vb.setXRange(n - v - 0.5, n - 0.5, padding=0)
        self._adapt_y()
        self._refresh_zones()

    def set_filter_segments(self, segments: List[FilterSegment], show: bool = True) -> None:
        self._segments = segments
        self._refresh_zones(show=show)

    def _refresh_zones(self, show: bool = True) -> None:
        if self._df is None or len(self._df) == 0:
            return
        lo  = float(self._df["low"].min())
        hi  = float(self._df["high"].max())
        pad = (hi - lo) * 0.10
        if show and self._segments:
            self._zi.set_zones(self._segments, self._df, lo - pad, hi + pad)
        else:
            self._zi.set_zones([], self._df, lo - pad, hi + pad)

    # ── Adaptive Y ────────────────────────────────────────────────────────────

    def _adapt_y(self) -> None:
        if self._df is None or len(self._df) == 0:
            return
        x0, x1 = self._vb.viewRange()[0]
        res = self._ci.ohlc_in_range(x0, x1)
        if res is None:
            return
        lo, hi = res
        pad = (hi - lo) * 0.07 or abs(lo) * 0.01 or 0.001
        self._vb.setYRange(lo - pad, hi + pad, padding=0)

    # ── Hover ─────────────────────────────────────────────────────────────────

    def _on_mouse(self, event) -> None:
        pos = event[0]
        if not self._vb.sceneBoundingRect().contains(pos):
            self._vl.hide(); self._hl.hide(); self._hud.hide()
            return

        mp = self._vb.mapSceneToView(pos)
        x, y = mp.x(), mp.y()
        self._vl.setPos(x); self._vl.show()
        self._hl.setPos(y); self._hl.show()

        if self._df is None or len(self._df) == 0:
            return

        idx = int(round(x))
        idx = max(0, min(len(self._df) - 1, idx))
        bar = self._df.iloc[idx]
        ts  = pd.Timestamp(bar["time"])

        bull      = bar["close"] >= bar["open"]
        arrow     = "▲" if bull else "▼"
        acol      = "#10b981" if bull else "#ef4444"
        t_str     = ts.strftime("%Y-%m-%d  %H:%M")

        html = (
            f"<span style='color:#9090b0'>{self.timeframe}</span>"
            f"&nbsp;&nbsp;<span style='color:#c8c8d2'>{t_str}</span>"
            f"&nbsp;&nbsp;<span style='color:{acol};font-weight:700'>{arrow}</span><br>"
            f"<span style='color:#9090b0'>O</span> <b>{bar['open']:.5f}</b>&nbsp;&nbsp;"
            f"<span style='color:#10b981'>H</span> <b>{bar['high']:.5f}</b>&nbsp;&nbsp;"
            f"<span style='color:#ef4444'>L</span> <b>{bar['low']:.5f}</b>&nbsp;&nbsp;"
            f"<span style='color:#9090b0'>C</span> <b>{bar['close']:.5f}</b>"
            f"&nbsp;&nbsp;<span style='color:#9090b0'>Vol</span> {int(bar['volume']):,}"
        )

        zone = self._zone_at(ts)
        if zone:
            sc  = "#10b981" if zone.valid else "#ef4444"
            st  = "✔ Valide" if zone.valid else "✘ Invalide"
            t0  = zone.t_start.strftime("%Y-%m-%d  %H:%M")
            t1  = zone.t_end.strftime("%Y-%m-%d  %H:%M")
            html += (
                f"<br><span style='color:#4f6ef7'>Zone</span>"
                f"&nbsp;&nbsp;{t0}"
                f"&nbsp;<span style='color:#9090b0'>→</span>&nbsp;{t1}"
                f"&nbsp;&nbsp;<span style='color:{sc};font-weight:700'>{st}</span>"
            )

        self._hud.setText(html)
        self._hud.adjustSize()
        self._hud.move(12, 12)
        self._hud.show()
        self._hud.raise_()

    def _zone_at(self, ts: pd.Timestamp) -> Optional[FilterSegment]:
        for seg in self._segments:
            if seg.t_start <= ts < seg.t_end:
                return seg
        return None

    # ── Overlays ──────────────────────────────────────────────────────────────

    def add_overlay(self, name: str, values: np.ndarray,
                    color: str = "#f59e0b", width: int = 1) -> None:
        x = np.arange(len(values), dtype=float)
        if name in self._overlay_curves:
            self._overlay_curves[name].setData(x=x, y=values)
        else:
            self._overlay_curves[name] = self._pp.plot(
                x=x, y=values, pen=pg.mkPen(color, width=width), name=name)

    def remove_overlay(self, name: str) -> None:
        if name in self._overlay_curves:
            self._pp.removeItem(self._overlay_curves.pop(name))

    def add_sub_indicator(self, name: str, values: np.ndarray,
                          color: str = "#818cf8") -> None:
        if name not in self._sub_plots:
            row  = self._glw.ci.layout.rowCount()
            plot = self._glw.addPlot(row=row, col=0)
            self._style_plot(plot, name)
            plot.setXLink(self._pp)
            self._glw.ci.layout.setRowStretchFactor(row, 1)
            if "RSI" in name.upper():
                for lvl, lc in [(30,"#10b981"),(50,"#6b7280"),(70,"#ef4444")]:
                    plot.addLine(y=lvl, pen=pg.mkPen(lc, style=Qt.PenStyle.DashLine))
            self._sub_plots[name] = plot
        x = np.arange(len(values), dtype=float)
        if name in self._sub_curves:
            self._sub_curves[name].setData(x=x, y=values)
        else:
            self._sub_curves[name] = self._sub_plots[name].plot(
                x=x, y=values, pen=pg.mkPen(color, width=1), name=name)

    def remove_sub_indicator(self, name: str) -> None:
        if name in self._sub_plots:
            self._glw.removeItem(self._sub_plots.pop(name))
            self._sub_curves.pop(name, None)

    # ── X-axis sync ───────────────────────────────────────────────────────────

    def price_view(self) -> pg.ViewBox:
        return self._vb

    def link_x(self, other: "ChartPanel") -> None:
        self._pp.setXLink(other._pp)
        for sp in self._sub_plots.values():
            sp.setXLink(other._pp)
