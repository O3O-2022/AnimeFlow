"""'진행 상황' 탭: 단계 표시, 실시간 미리보기, 상세 통계."""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..planner import KIND_GEN, KIND_LABELS, Plan
from .widgets import StatCard, fmt_time

STAGES = [
    ("analyze", "① 프레임 분석"),
    ("ai", "② AI 장면 감독"),
    ("plan", "③ 프레임 구조"),
    ("render", "④ 생성 · 효과 · 인코딩"),
    ("dub", "⑤ 한국어 더빙"),
]
STATE_TEXT = {"wait": "대기", "run": "진행 중", "done": "완료", "skip": "건너뜀", "fail": "중단"}


def fmt_dur(sec: float | None) -> str:
    if sec is None or sec != sec or sec < 0:
        return "계산 중…"
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}시간 {m:02d}분" if h else (f"{m}분 {s:02d}초" if m else f"{s}초")


def fmt_bytes(n: int) -> str:
    return f"{n / 1024 ** 3:.2f} GB" if n >= 1024 ** 3 else f"{n / 1024 ** 2:.1f} MB"


def bgr_to_pixmap(img: np.ndarray) -> QPixmap:
    img = np.ascontiguousarray(img[:, :, ::-1])  # BGR → RGB
    h, w = img.shape[:2]
    return QPixmap.fromImage(QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy())


class StageChip(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("stageChip")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(1)
        self.title = QLabel(title, objectName="stageTitle")
        self.state = QLabel(objectName="stageState")
        self.detail = QLabel(objectName="hint")
        self.detail.setWordWrap(True)
        for w in (self.title, self.state, self.detail):
            lay.addWidget(w)
        self.t0 = 0.0
        self.set("wait")

    def set(self, state: str, detail: str = ""):
        if state == "run":
            self.t0 = time.perf_counter()
        elif state in ("done", "fail") and self.t0:
            detail = f"{detail} · {fmt_dur(time.perf_counter() - self.t0)}".strip(" ·")
        self.setProperty("state", state)
        self.state.setText(STATE_TEXT.get(state, state))
        self.detail.setText(detail)
        self.style().unpolish(self)
        self.style().polish(self)


class PreviewPane(QWidget):
    def __init__(self, title: str):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.title = QLabel(title, objectName="hint")
        self.img = QLabel("변환을 시작하면 여기에 표시됩니다", objectName="preview", alignment=Qt.AlignCenter)
        self.img.setMinimumSize(320, 180)
        self.img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.title)
        lay.addWidget(self.img, 1)
        self._pm: QPixmap | None = None

    def set_image(self, pm: QPixmap):
        self._pm = pm
        self._fit()

    def _fit(self):
        if self._pm is not None:
            self.img.setPixmap(self._pm.scaled(self.img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def clear(self):
        self._pm = None
        self.img.setPixmap(QPixmap())
        self.img.setText("변환을 시작하면 여기에 표시됩니다")


class ProgressView(QWidget):
    def __init__(self):
        super().__init__()
        self.plan: Plan | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        row = QHBoxLayout()
        self.chips = {}
        for key, title in STAGES:
            chip = StageChip(title)
            self.chips[key] = chip
            row.addWidget(chip)
        lay.addLayout(row)

        pv = QHBoxLayout()
        self.src_pane = PreviewPane("원본 그림 (이 순간의 기준 그림)")
        self.out_pane = PreviewPane("지금 만든 출력 프레임 (효과 포함)")
        pv.addWidget(self.src_pane, 1)
        pv.addWidget(self.out_pane, 1)
        lay.addLayout(pv, 1)
        self.caption = QLabel("", objectName="caption", alignment=Qt.AlignCenter)
        self.caption.setWordWrap(True)
        lay.addWidget(self.caption)

        grid = QGridLayout()
        grid.setSpacing(8)
        self.c_elapsed = StatCard("경과 시간")
        self.c_eta = StatCard("남은 시간")
        self.c_finish = StatCard("예상 완료 시각")
        self.c_speed = StatCard("처리 속도")
        self.c_pos = StatCard("영상 위치")
        self.c_gen = StatCard("AI 생성 프레임")
        self.c_size = StatCard("출력 파일 크기")
        self.c_gpu = StatCard("GPU 메모리")
        cards = [self.c_elapsed, self.c_eta, self.c_finish, self.c_speed,
                 self.c_pos, self.c_gen, self.c_size, self.c_gpu]
        for i, c in enumerate(cards):
            grid.addWidget(c, i // 4, i % 4)
        lay.addLayout(grid)
        self.kinds = QLabel("", objectName="hint", alignment=Qt.AlignCenter)
        lay.addWidget(self.kinds)

    # ---------------------------------------------------------------- 갱신
    def reset(self):
        self.plan = None
        for chip in self.chips.values():
            chip.set("wait")
        self.src_pane.clear()
        self.out_pane.clear()
        self.caption.setText("")
        self.kinds.setText("")
        for c in (self.c_elapsed, self.c_eta, self.c_finish, self.c_speed,
                  self.c_pos, self.c_gen, self.c_size, self.c_gpu):
            c.set("-")

    def set_stage(self, key: str, state: str, detail: str):
        chip = self.chips.get(key)
        if chip:
            chip.set(state, detail)

    def set_plan(self, plan: Plan, est_sec: float | None):
        self.plan = plan
        if est_sec:
            self.c_eta.set(fmt_dur(est_sec), "이 PC의 이전 변환 속도로 계산한 예상값")

    def set_preview(self, k: int, out_thumb: np.ndarray, src_thumb: np.ndarray):
        self.out_pane.set_image(bgr_to_pixmap(out_thumb))
        self.src_pane.set_image(bgr_to_pixmap(src_thumb))
        p = self.plan
        if p is None or k >= p.n_out:
            return
        kind = int(p.kind[k])
        t = k / float(p.fps)
        a, b = int(p.key_a[k]) + 1, int(p.key_b[k]) + 1
        if kind == KIND_GEN:
            what = f"AI 생성: 그림 #{a:,} → #{b:,} 사이 {p.ratio[k] * 100:.0f}% 지점"
        else:
            what = f"{KIND_LABELS[kind]}: 그림 #{a:,}"
        self.caption.setText(f"출력 프레임 {k + 1:,} / {p.n_out:,}   ·   {fmt_time(t)}   ·   {what}")

    def set_status(self, st: dict):
        el, eta = st["elapsed"], st["eta"]
        self.c_elapsed.set(fmt_dur(el))
        if eta is not None and st["work_frac"] > 0.003:
            self.c_eta.set(fmt_dur(eta))
            self.c_finish.set(time.strftime("%H:%M", time.localtime(time.time() + eta)))
        self.c_speed.set(f"{st['fps']:.1f} fps", "초당 출력 프레임 수")
        self.c_pos.set(f"{fmt_time(st['video_time'])[:-4]} / {fmt_time(st['duration'])[:-4]}")
        self.c_gen.set(f"{st['gen_done']:,} / {st['gen_total']:,}")
        self.c_size.set(fmt_bytes(st.get("out_bytes", 0)))
        if "gpu_used" in st:
            self.c_gpu.set(f"{st['gpu_used'] / 1024 ** 3:.1f} / {st['gpu_total'] / 1024 ** 3:.0f} GB")
        self.chips["render"].detail.setText(f"{st['work_frac'] * 100:.1f}% · 남은 시간 {fmt_dur(eta)}")
        counts = st["kind_counts"]
        parts = [f"{KIND_LABELS[i]} {counts[i]:,}" for i in range(len(counts)) if counts[i]]
        self.kinds.setText("지금까지:  " + "   ·   ".join(parts))
