"""커스텀 위젯: 장면 타임라인, 통계 카드."""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QFrame, QLabel, QToolTip, QVBoxLayout, QWidget

from ..analyzer import Timeline
from ..planner import scene_interp_params
from .style import ACCENT, MUTED, WARN


def fmt_time(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:06.3f}"


class StatCard(QFrame):
    def __init__(self, name: str):
        super().__init__()
        self.setObjectName("statCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)
        self.value = QLabel("-")
        self.value.setObjectName("statValue")
        n = QLabel(name)
        n.setObjectName("statName")
        lay.addWidget(self.value)
        lay.addWidget(n)

    def set(self, text: str, tip: str = ""):
        self.value.setText(text)
        self.setToolTip(tip)


class TimelineView(QWidget):
    """장면(컷)을 블록으로 표시. 색: 보라=보간, 회색=보간 안 함, 주황=AI가 끈 장면. 높이=움직임 양."""

    sceneClicked = Signal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(74)
        self.setMouseTracking(True)
        self.tl: Timeline | None = None
        self.settings = None
        self.selected = -1

    def set_timeline(self, tl: Timeline | None, settings):
        self.tl, self.settings = tl, settings
        self.update()

    def select(self, scene_id: int):
        self.selected = scene_id
        self.update()

    def _scene_at(self, x: float) -> int:
        if not self.tl or self.width() <= 0:
            return -1
        t = x / self.width() * self.tl.duration
        for sc in self.tl.scenes:
            if sc.start <= t < sc.end:
                return sc.id
        return -1

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#0e1015"))
        if not self.tl:
            p.setPen(QColor(MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "영상을 분석하면 장면 타임라인이 여기에 표시됩니다")
            return
        tl = self.tl
        enabled, _ = scene_interp_params(tl, self.settings)
        max_motion = max((sc.motion for sc in tl.scenes), default=0) or 1.0
        base_y = h - 16
        for sc in tl.scenes:
            x0 = sc.start / tl.duration * w
            x1 = max(x0 + 1.0, sc.end / tl.duration * w)
            frac = 0.25 + 0.75 * min(1.0, sc.motion / max_motion)
            bh = (base_y - 6) * frac
            if enabled[sc.id]:
                col = QColor(ACCENT)
            elif sc.ai and not sc.ai.get("interpolate", True) and sc.manual_interp is None:
                col = QColor(WARN)
            else:
                col = QColor(MUTED)
            rect = QRectF(x0, base_y - bh, x1 - x0 - (0.6 if x1 - x0 > 3 else 0), bh)
            p.fillRect(rect, col)
            if sc.id == self.selected:
                p.setPen(QPen(QColor("#ffffff"), 2))
                p.drawRect(rect.adjusted(1, 1, -1, -1))
        # 시간 눈금
        p.setPen(QPen(QColor("#5b6172")))
        steps = 6
        for i in range(steps + 1):
            x = i / steps * (w - 1)
            p.drawLine(int(x), base_y, int(x), base_y + 4)
            txt = fmt_time(tl.duration * i / steps)[:-4]
            if i == 0:
                rect, align = QRectF(x, base_y + 2, 80, 14), Qt.AlignLeft
            elif i == steps:
                rect, align = QRectF(x - 80, base_y + 2, 80, 14), Qt.AlignRight
            else:
                rect, align = QRectF(x - 40, base_y + 2, 80, 14), Qt.AlignHCenter
            p.drawText(rect, align | Qt.AlignTop, txt)

    def mouseMoveEvent(self, e):
        sid = self._scene_at(e.position().x())
        if sid >= 0 and self.tl:
            sc = self.tl.scenes[sid]
            ai = f"\nAI: {sc.ai.get('scene_type')} · {sc.ai.get('reason', '')}" if sc.ai else ""
            QToolTip.showText(e.globalPosition().toPoint(),
                              f"장면 {sid + 1}  {fmt_time(sc.start)} ~ {fmt_time(sc.end)}\n"
                              f"그림 {sc.n_keys}장 · 움직임 {sc.motion:.3f}{ai}", self)

    def mousePressEvent(self, e):
        sid = self._scene_at(e.position().x())
        if sid >= 0:
            self.sceneClicked.emit(sid)
