"""AnimeFlow 메인 창."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget, QButtonGroup,
)

from .. import __version__
from ..analyzer import Timeline
from ..config import CODECS, CONTAINER_LABELS, ROOT, Settings, ffmpeg_bin
from ..director import _scene_stats, list_models
from ..effects import preview_jpeg
from ..ffmpeg_io import probe
from ..interpolator import rife_available
from ..planner import build_plan, estimate_render_seconds, risky_keys, scene_interp_params
from .style import QSS
from .progress_view import ProgressView, fmt_dur
from .tasks import FuncTask, PipelineTask
from .widgets import StatCard, TimelineView, fmt_time

VIDEO_FILTER = "영상 파일 (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.ts *.m2ts *.flv *.wmv);;모든 파일 (*.*)"
FPS_PRESETS = [24, 30, 48, 60, 90, 120]
FAST_MODE_ITEMS = [("자동 - 잔상 날 순간만 원본 (권장)", "auto"),
                   ("끊김 없이 - 전부 AI 보간 (잔상 감수)", "ai"),
                   ("잔상 없이 - 위험하면 원본 (멈칫 감수)", "hold"),
                   ("선 안정 우선 - 작은 움직임만 보간 (24fps와 추천)", "lines")]
BACKENDS = [("RIFE 4.25 · 로컬 AI (권장, GPU)", "rife"),
            ("OpenCV 광학 흐름 · AI 없이", "opencv"),
            ("단순 블렌드 · 테스트용", "blend")]


def hint(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setObjectName("hint")
    lb.setWordWrap(True)
    return lb


def slider_row(lo: int, hi: int, suffix_fn) -> tuple[QWidget, QSlider, QLabel]:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    sl = QSlider(Qt.Horizontal)
    sl.setRange(lo, hi)
    val = QLabel()
    val.setMinimumWidth(92)
    val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    sl.valueChanged.connect(lambda v: val.setText(suffix_fn(v)))
    lay.addWidget(sl, 1)
    lay.addWidget(val)
    return w, sl, val


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"AnimeFlow {__version__} — 애니메이션 프레임 보간")
        self.resize(1360, 880)
        self.setAcceptDrops(True)

        self.qs = QSettings("AnimeFlow", "AnimeFlow")
        try:
            self.s = Settings.from_dict(json.loads(self.qs.value("settings", "{}")))
        except (TypeError, ValueError):
            self.s = Settings()

        self.tl: Timeline | None = None
        self.tl_sig = None
        self.ai_done_model: str | None = None
        self.task: PipelineTask | None = None
        self._bg: list = []  # 실행 중인 FuncTask 참조 보관
        self._out_manual = bool(self.s.output_path)
        self._stage = ""
        self._stage_t0 = 0.0
        self._loading = True
        self._save_timer = QTimer(self, singleShot=True, interval=400, timeout=self._save_settings)
        self._preview_timer = QTimer(self, singleShot=True, interval=250, timeout=self._update_preview)

        self._build_ui()
        self._apply_settings(self.s)
        self._loading = False

        self._check_environment()
        if self.s.input_path and Path(self.s.input_path).exists():
            self._probe_input()

    # ------------------------------------------------------------------ UI 구성
    def _build_ui(self):
        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(14)

        # ---- 왼쪽: 설정
        left_inner = QWidget()
        left = QVBoxLayout(left_inner)
        left.setContentsMargins(0, 0, 6, 0)
        left.setSpacing(4)
        title = QLabel("AnimeFlow", objectName="title")
        sub = QLabel("같은 그림을 묶고 · AI로 중간 프레임을 만들고 · 싱크는 그대로", objectName="subtitle")
        left.addWidget(title)
        left.addWidget(sub)

        left.addWidget(self._card_io())
        left.addWidget(self._card_frames())
        left.addWidget(self._card_effects())
        left.addWidget(self._card_ai())
        left.addWidget(self._card_dub())
        left.addWidget(self._card_advanced())
        left.addWidget(self._card_experimental())
        left.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(left_inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(480)
        # 긴 항목 이름 때문에 콤보박스가 패널보다 넓어지지 않게
        for combo in left_inner.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        outer.addWidget(scroll)

        # ---- 오른쪽: 결과
        right = QVBoxLayout()
        right.setSpacing(10)
        stats = QHBoxLayout()
        self.card_src = StatCard("원본 프레임")
        self.card_keys = StatCard("고유 그림 (중복 제거)")
        self.card_scenes = StatCard("장면(컷)")
        self.card_out = StatCard("출력 프레임")
        self.card_gen = StatCard("새로 생성할 프레임")
        self.card_est = StatCard("예상 변환 시간")
        for c in (self.card_src, self.card_keys, self.card_scenes, self.card_out, self.card_gen, self.card_est):
            stats.addWidget(c)
        right.addLayout(stats)
        self.sync_label = hint("영상을 넣고 [분석]을 누르면 프레임 구조를 보여줍니다. 영상 파일을 창에 끌어다 놓아도 됩니다.")
        right.addWidget(self.sync_label)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_analysis(), "분석 결과")
        self.progress_view = ProgressView()
        self.tabs.addTab(self.progress_view, "진행 상황")
        self.log_view = QPlainTextEdit(readOnly=True)
        self.log_view.setMaximumBlockCount(5000)
        self.tabs.addTab(self.log_view, "로그")
        right.addWidget(self.tabs, 1)

        right.addWidget(self._bottom_bar())
        outer.addLayout(right, 1)

    def _card_io(self) -> QGroupBox:
        g = QGroupBox("1   입력 · 출력")
        lay = QVBoxLayout(g)
        row = QHBoxLayout()
        self.in_edit = QLineEdit(placeholderText="영상 파일을 선택하거나 창에 끌어다 놓으세요")
        self.in_edit.setReadOnly(True)
        b = QPushButton("영상 열기")
        b.clicked.connect(self._browse_input)
        row.addWidget(self.in_edit, 1)
        row.addWidget(b)
        lay.addLayout(row)
        self.info_label = hint("")
        lay.addWidget(self.info_label)

        row = QHBoxLayout()
        self.out_edit = QLineEdit(placeholderText="저장할 파일 경로")
        self.out_edit.textEdited.connect(self._on_out_edited)
        b = QPushButton("저장 위치")
        b.clicked.connect(self._browse_output)
        row.addWidget(self.out_edit, 1)
        row.addWidget(b)
        lay.addLayout(row)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        self.container_combo = QComboBox()
        for k, label in CONTAINER_LABELS.items():
            self.container_combo.addItem(label, k)
        self.container_combo.currentIndexChanged.connect(self._on_container_changed)
        self.codec_combo = QComboBox()
        self.codec_combo.currentIndexChanged.connect(self._changed)
        w, self.quality_slider, _ = slider_row(
            0, 40, lambda v: f"{v} · " + ("최고화질" if v <= 14 else "고화질" if v <= 20 else "보통" if v <= 27 else "저용량"))
        self.quality_slider.valueChanged.connect(self._changed)
        form.addRow("출력 형식", self.container_combo)
        form.addRow("비디오 코덱", self.codec_combo)
        form.addRow("화질", w)
        lay.addLayout(form)
        lay.addWidget(hint("화질 숫자가 낮을수록 고화질·대용량입니다. 오디오·자막은 원본에서 그대로 가져옵니다."))
        return g

    def _card_frames(self) -> QGroupBox:
        g = QGroupBox("2   프레임")
        lay = QVBoxLayout(g)
        row = QHBoxLayout()
        row.addWidget(QLabel("Target FPS"))
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setRange(24, 120)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(24, 120)
        self.fps_spin.setSuffix(" fps")
        self.fps_slider.valueChanged.connect(self.fps_spin.setValue)
        self.fps_spin.valueChanged.connect(self.fps_slider.setValue)
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        row.addWidget(self.fps_slider, 1)
        row.addWidget(self.fps_spin)
        lay.addLayout(row)

        row = QHBoxLayout()
        self.preset_group = QButtonGroup(self)
        self.preset_group.setExclusive(False)
        for f in FPS_PRESETS:
            pb = QPushButton(str(f), objectName="preset", checkable=True)
            pb.clicked.connect(lambda _=False, f=f: self.fps_spin.setValue(f))
            self.preset_group.addButton(pb, f)
            row.addWidget(pb)
        lay.addLayout(row)

        form = QFormLayout()
        self.backend_combo = QComboBox()
        for label, key in BACKENDS:
            self.backend_combo.addItem(label, key)
        self.backend_combo.currentIndexChanged.connect(self._changed)
        self.scale_combo = QComboBox()
        for label, v in [("1.0 · 1080p 이하 (기본)", 1.0), ("0.5 · 4K 영상용", 0.5), ("0.25 · 8K / 저사양", 0.25)]:
            self.scale_combo.addItem(label, v)
        self.scale_combo.currentIndexChanged.connect(self._changed)
        self.fp16_check = QCheckBox("FP16 가속 (RTX 그래픽카드 권장)")
        self.fp16_check.toggled.connect(self._changed)
        form.addRow("보간 엔진", self.backend_combo)
        form.addRow("RIFE 배율", self.scale_combo)
        lay.addLayout(form)
        lay.addWidget(self.fp16_check)
        row = QHBoxLayout()
        row.addWidget(QLabel("빠른 동작"))
        self.fast_combo = QComboBox()
        for label, key in FAST_MODE_ITEMS:
            self.fast_combo.addItem(label, key)
        self.fast_combo.setToolTip(
            "보간 AI는 움직임을 '옮기는' 방식이라, 그림 한 장 사이에 포즈가 크게 바뀌면 새 포즈를 그리지 못하고\n"
            "두 그림이 겹친 잔상을 만듭니다. 그런 순간을 어떻게 처리할지 고릅니다.\n\n"
            "자동: 실측 기준으로 잔상이 날 순간만 원본 그림 유지\n"
            "끊김 없이: 모두 AI 보간 - 부드럽지만 빠른 동작에서 잔상\n"
            "잔상 없이: 조금이라도 위험하면 원본 유지 - 깔끔하지만 가끔 멈칫\n"
            "선 안정 우선: 눈·입·머리카락 같은 작은 움직임만 보간하고 나머지는 원래 그림 그대로.\n"
            "  선 하나가 다른 선을 애매하게 따라가며 잔상처럼 보이는 현상이 거의 없어집니다. 24fps와 함께 쓰면 가장 원작에 가깝습니다.")
        self.fast_combo.currentIndexChanged.connect(self._changed)
        row.addWidget(self.fast_combo, 1)
        lay.addLayout(row)
        self.rife_status = hint("RIFE 상태 확인 중…")
        lay.addWidget(self.rife_status)
        return g

    def _card_effects(self) -> QGroupBox:
        g = QGroupBox("3   화질 보정 · 후처리 효과")
        lay = QGridLayout(g)
        self.restore_check = QCheckBox("원본 보정 (선 정리·노이즈 제거, AI)")
        self.restore_check.setToolTip("애니메이션 전용 복원 AI로 번진 선과 압축 노이즈를 정리합니다.\n"
                                      "원본 그림에 먼저 적용하고 그 그림으로 중간 프레임을 만들어서, 새 프레임도 깨끗해집니다.")
        self.restore_res_combo = QComboBox()
        for label, key in [("해상도 원본 유지", "source"), ("1080p로 올리기", "1080"), ("1440p로 올리기", "1440")]:
            self.restore_res_combo.addItem(label, key)
        self.restore_check.toggled.connect(self.restore_res_combo.setEnabled)
        self.restore_check.toggled.connect(self._on_effects_changed)
        self.restore_res_combo.currentIndexChanged.connect(self._on_effects_changed)
        self.shading_check = QCheckBox("명암 보정")
        w1, self.shading_slider, _ = slider_row(0, 100, lambda v: f"강도 {v}%")
        self.lighting_check = QCheckBox("광원 효과 (빛 번짐)")
        w2, self.lighting_slider, _ = slider_row(0, 100, lambda v: f"강도 {v}%")
        self.vignette_check = QCheckBox("비네팅 (가장자리 어둡게)")
        for wdg in (self.shading_check, self.lighting_check, self.vignette_check):
            wdg.toggled.connect(self._on_effects_changed)
        for sl in (self.shading_slider, self.lighting_slider):
            sl.valueChanged.connect(self._on_effects_changed)
        self.shading_check.toggled.connect(w1.setEnabled)
        self.lighting_check.toggled.connect(w2.setEnabled)
        lay.addWidget(self.restore_check, 0, 0)
        lay.addWidget(self.restore_res_combo, 0, 1)
        lay.addWidget(self.shading_check, 1, 0)
        lay.addWidget(w1, 1, 1)
        lay.addWidget(self.lighting_check, 2, 0)
        lay.addWidget(w2, 2, 1)
        lay.addWidget(self.vignette_check, 3, 0, 1, 2)
        lay.addWidget(hint("오른쪽 '분석 결과' 탭에서 장면을 선택하면 효과 미리보기를 볼 수 있습니다."), 4, 0, 1, 2)
        return g

    def _card_experimental(self) -> QGroupBox:
        g = QGroupBox("실험 기능  (결과가 장면마다 들쭉날쭉할 수 있음)", objectName="experimental")
        lay = QGridLayout(g)
        self.relight_check = QCheckBox("입체 조명 (그림자·역광·발광)")
        self.relight_check.setToolTip(
            "AI가 장면의 앞뒤 깊이를 추정해서 광원 방향에 맞게 인물에 셀 그림자를 넣고,\n"
            "광원 쪽 윤곽에 림 라이트, 조명·창문처럼 작은 빛에 발광을 넣습니다.\n"
            "광원 방향 자동 추정이 틀리는 장면이 있어 실험 기능으로 분류했습니다.")
        w3, self.relight_slider, _ = slider_row(0, 200, lambda v: f"강도 {v}%")
        self.light_combo = QComboBox()
        for label, key in [("광원 자동 (배경의 밝은 곳)", "auto"), ("왼쪽 위에서", "left"),
                           ("오른쪽 위에서", "right"), ("위에서", "top"), ("뒤에서 (역광)", "back")]:
            self.light_combo.addItem(label, key)
        self.light_color_combo = QComboBox()
        for label, key in [("빛 색 자동 (장면에서)", "auto"), ("따뜻한 노을빛", "warm"), ("차가운 푸른빛", "cool"),
                           ("붉은 조명", "red"), ("흰 빛", "white")]:
            self.light_color_combo.addItem(label, key)
        for combo in (self.light_combo, self.light_color_combo):
            combo.currentIndexChanged.connect(self._on_effects_changed)
            self.relight_check.toggled.connect(combo.setEnabled)
        self.relight_check.toggled.connect(w3.setEnabled)
        self.relight_check.toggled.connect(self._on_effects_changed)
        self.relight_slider.valueChanged.connect(self._on_effects_changed)
        lay.addWidget(self.relight_check, 0, 0)
        lay.addWidget(w3, 0, 1)
        lay.addWidget(self.light_combo, 1, 0)
        lay.addWidget(self.light_color_combo, 1, 1)
        lay.addWidget(hint("그림체는 그대로 두고 조명만 극적으로 바꿉니다. 광원 자동 추정이 틀리면 방향을 직접 고르세요. "
                           "720p 기준 프레임당 약 10~15ms 더 걸립니다."), 2, 0, 1, 2)
        return g

    def _card_ai(self) -> QGroupBox:
        g = QGroupBox("4   로컬 AI 장면 감독  (Ollama)")
        lay = QVBoxLayout(g)
        self.ai_check = QCheckBox("AI가 장면을 보고 보간 구조를 정하게 하기")
        self.ai_check.toggled.connect(self._changed)
        lay.addWidget(self.ai_check)
        form = QFormLayout()
        self.ollama_url = QLineEdit()
        self.ollama_url.editingFinished.connect(self._refresh_models)
        self.ollama_url.textChanged.connect(self._changed)
        row = QHBoxLayout()
        self.model_combo = QComboBox(editable=True)
        self.model_combo.currentTextChanged.connect(self._changed)
        rb = QPushButton("새로고침")
        rb.clicked.connect(self._refresh_models)
        row.addWidget(self.model_combo, 1)
        row.addWidget(rb)
        self.ai_min_spin = QDoubleSpinBox(decimals=1, minimum=0.0, maximum=30.0, singleStep=0.5, suffix=" 초 이상")
        self.ai_min_spin.valueChanged.connect(self._changed)
        form.addRow("서버 주소", self.ollama_url)
        form.addRow("비전 모델", row)
        form.addRow("분석할 장면", self.ai_min_spin)
        lay.addLayout(form)
        self.ai_status = hint("")
        lay.addWidget(self.ai_status)
        return g

    def _card_dub(self) -> QGroupBox:
        g = QGroupBox("5   한국어 AI 더빙  (실험)")
        lay = QVBoxLayout(g)
        self.dub_check = QCheckBox("변환이 끝나면 한국어 더빙 파일도 만들기")
        self.dub_check.setToolTip("대사를 분리해 받아쓰고, 로컬 AI로 번역한 뒤 AI 목소리로 한국어 음성 트랙과 자막을 넣습니다.\n"
                                  "원본 음성도 선택할 수 있게 남겨둡니다. 결과는 '(한국어 더빙)' 파일로 따로 저장됩니다.")
        self.dub_vc_check = QCheckBox("캐릭터 원래 목소리로 음색 변환 (Seed-VC)")
        form = QFormLayout()
        self.dub_llm_combo = QComboBox()
        for label, key in [("자동 (설치된 것 중 가장 좋은 것)", "auto"), ("EXAONE 3.5 32B (정확, 느림)", "exaone3.5:32b"),
                           ("EXAONE 3.5 7.8B (빠름)", "exaone3.5:7.8b"),
                           ("DeepSeek-V3 · 클라우드 (가장 자연스러움, API 키 필요)", "deepseek-chat"),
                           ("DeepSeek-R1 · 클라우드 (추론형, 느림, API 키 필요)", "deepseek-reasoner")]:
            self.dub_llm_combo.addItem(label, key)
        form.addRow("번역 AI", self.dub_llm_combo)
        self.dub_ctx_edit = QLineEdit(placeholderText="예: 인빈시블. 옴니맨(아빠)과 마크(아들)의 대화. 서로 반말")
        self.dub_ctx_edit.setToolTip("누가 누구인지, 서로 반말/존댓말인지 적어주면 번역 AI가 추측 대신 이 정보를 따릅니다.\n"
                                     "대사만으로는 누가 말하는지 알기 어려운 장면에서 번역이 크게 좋아집니다.")
        self.dub_ctx_edit.textChanged.connect(self._changed)
        self.dub_check.toggled.connect(self.dub_ctx_edit.setEnabled)
        form.addRow("작품·인물 메모", self.dub_ctx_edit)
        self.dub_key_edit = QLineEdit(placeholderText="DeepSeek을 고른 경우에만: platform.deepseek.com 에서 발급한 키")
        self.dub_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.dub_key_edit.setToolTip("DeepSeek 클라우드 번역을 고르면 받아쓴 영어 대사 글자와 작품·인물 메모가 DeepSeek 서버로 전송됩니다.\n"
                                     "음성·영상은 보내지 않습니다. 요금은 본인 계정에 청구됩니다 (1분 영상에 보통 몇 원 수준).\n"
                                     "키는 이 PC의 앱 설정에만 저장됩니다.")
        self.dub_key_edit.textChanged.connect(self._changed)
        form.addRow("DeepSeek API 키", self.dub_key_edit)
        self.dub_llm_combo.currentIndexChanged.connect(self._update_dub_key)
        self.dub_check.toggled.connect(self._update_dub_key)
        for w in (self.dub_check, self.dub_vc_check):
            w.toggled.connect(self._changed)
        self.dub_llm_combo.currentIndexChanged.connect(self._changed)
        self.dub_check.toggled.connect(self.dub_vc_check.setEnabled)
        self.dub_check.toggled.connect(self.dub_llm_combo.setEnabled)
        lay.addWidget(self.dub_check)
        lay.addWidget(self.dub_vc_check)
        lay.addLayout(form)
        self.dub_status = hint("")
        lay.addWidget(self.dub_status)
        lay.addWidget(hint("⚠ AI 목소리(XTTS)와 캐릭터 목소리 복제는 개인 감상용으로만 쓰세요. 공유·판매 금지. "
                           "1분 영상에 약 10~20분 걸립니다."))
        return g

    def _update_dub_key(self, *_):
        cloud = str(self.dub_llm_combo.currentData() or "").startswith("deepseek-")
        self.dub_key_edit.setEnabled(self.dub_check.isChecked() and cloud)

    def _card_advanced(self) -> QGroupBox:
        g = QGroupBox("고급 · 분석 감도")
        form = QFormLayout(g)
        self.dup_spin = QDoubleSpinBox(decimals=3, minimum=0.005, maximum=2.0, singleStep=0.01, suffix=" %")
        self.dup_spin.setToolTip("두 프레임에서 바뀐 픽셀이 이 비율보다 적으면 '같은 그림'으로 묶습니다.\n"
                                 "입 모양 같은 작은 움직임을 놓치면 값을 낮추세요.")
        self.mean_spin = QDoubleSpinBox(decimals=1, minimum=0.2, maximum=8.0, singleStep=0.1)
        self.mean_spin.setToolTip("화면 전체가 조금씩 변하는 페이드를 '같은 그림'으로 오판하지 않게 막는 값입니다.")
        self.scene_spin = QDoubleSpinBox(decimals=2, minimum=0.05, maximum=0.95, singleStep=0.05)
        self.scene_spin.setToolTip("색 분포 유사도가 이 값보다 낮으면 장면 전환(컷)으로 보고 그 사이는 보간하지 않습니다.\n"
                                   "값을 올리면 컷을 더 많이 감지합니다.")
        self.maxgap_spin = QSpinBox(minimum=1, maximum=12, suffix=" 원본 프레임")
        self.maxgap_spin.setToolTip("그림이 이보다 오래 멈춰 있으면(홀드) 끝부분만 움직이게 보간합니다.\n"
                                    "그래야 멈춘 그림이 천천히 녹아내리는 듯한 어색한 모핑이 생기지 않습니다.")
        for w in (self.dup_spin, self.mean_spin, self.scene_spin, self.maxgap_spin):
            w.valueChanged.connect(self._changed)
        form.addRow("같은 그림 판정", self.dup_spin)
        form.addRow("페이드 보호", self.mean_spin)
        form.addRow("장면 전환 감도", self.scene_spin)
        form.addRow("최대 보간 길이", self.maxgap_spin)
        rst = QPushButton("기본값으로")
        rst.clicked.connect(self._reset_advanced)
        form.addRow("", rst)
        return g

    def _tab_analysis(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        legend = hint("■ 보라: 보간함   ■ 회색: 보간 안 함   ■ 주황: AI가 보간을 끈 장면   · 막대 높이 = 움직임 양 · 클릭하면 장면 선택")
        lay.addWidget(self.timeline_view_init())
        lay.addWidget(legend)

        split = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["#", "시작", "끝", "길이", "그림 수", "작화 리듬", "움직임", "AI 판단", "보간"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(7, QHeaderView.Stretch)
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        split.addWidget(self.table)

        pv = QWidget()
        pl = QVBoxLayout(pv)
        pl.setContentsMargins(6, 0, 0, 0)
        pl.addWidget(hint("장면 첫 그림 (원본)"))
        self.preview_src = QLabel("장면을 선택하세요", objectName="preview", alignment=Qt.AlignCenter)
        pl.addWidget(self.preview_src, 1)
        pl.addWidget(hint("효과 적용 미리보기"))
        self.preview_fx = QLabel("", objectName="preview", alignment=Qt.AlignCenter)
        pl.addWidget(self.preview_fx, 1)
        for lbl in (self.preview_src, self.preview_fx):
            lbl.setMinimumSize(280, 158)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        split.addWidget(pv)
        split.setSizes([640, 340])
        lay.addWidget(split, 1)
        return w

    def timeline_view_init(self) -> TimelineView:
        self.timeline = TimelineView()
        self.timeline.sceneClicked.connect(self._select_scene_row)
        return self.timeline

    def _bottom_bar(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.stage_label = QLabel("대기 중")
        self.eta_label = hint("")
        top.addWidget(self.stage_label, 1)
        top.addWidget(self.eta_label)
        lay.addLayout(top)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        lay.addWidget(self.progress)
        row = QHBoxLayout()
        self.open_btn = QPushButton("출력 폴더 열기", objectName="secondary")
        self.open_btn.clicked.connect(self._open_output_folder)
        self.analyze_btn = QPushButton("분석만 하기", objectName="secondary")
        self.analyze_btn.clicked.connect(lambda: self._start(render=False))
        self.cancel_btn = QPushButton("취소", objectName="danger")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        self.start_btn = QPushButton("▶  변환 시작", objectName="primary")
        self.start_btn.clicked.connect(lambda: self._start(render=True))
        row.addWidget(self.open_btn)
        row.addStretch(1)
        row.addWidget(self.analyze_btn)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.start_btn)
        lay.addLayout(row)
        return w

    # ------------------------------------------------------------------ 설정 <-> 위젯
    def _apply_settings(self, s: Settings):
        self.in_edit.setText(s.input_path)
        self.out_edit.setText(s.output_path)
        self._set_combo(self.container_combo, s.container)
        self._fill_codecs(s.container, s.video_codec)
        self.quality_slider.setValue(s.quality)
        self.fps_spin.setValue(int(s.target_fps))
        self.fps_slider.setValue(int(s.target_fps))
        self._sync_presets(int(s.target_fps))
        self._set_combo(self.backend_combo, s.backend)
        self._set_combo(self.scale_combo, s.rife_scale)
        self.fp16_check.setChecked(s.fp16)
        self.shading_check.setChecked(s.shading)
        self.shading_slider.setValue(int(s.shading_strength * 100))
        self.lighting_check.setChecked(s.lighting)
        self.lighting_slider.setValue(int(s.lighting_strength * 100))
        self.vignette_check.setChecked(s.vignette)
        self.dub_check.setChecked(s.dub)
        self.dub_vc_check.setChecked(s.dub_vc)
        self._set_combo(self.dub_llm_combo, s.dub_llm)
        self.dub_vc_check.setEnabled(s.dub)
        self.dub_llm_combo.setEnabled(s.dub)
        self.dub_ctx_edit.setText(s.dub_context)
        self.dub_ctx_edit.setEnabled(s.dub)
        self.dub_key_edit.setText(s.dub_api_key)
        self._update_dub_key()
        self.restore_check.setChecked(s.restore)
        self._set_combo(self.restore_res_combo, s.restore_res)
        self.restore_res_combo.setEnabled(s.restore)
        self.relight_check.setChecked(s.relight)
        self.relight_slider.setValue(int(s.relight_strength * 100))
        self._set_combo(self.light_combo, s.light_dir)
        self.relight_slider.parentWidget().setEnabled(s.relight)
        self.light_combo.setEnabled(s.relight)
        self._set_combo(self.light_color_combo, s.light_color)
        self.light_color_combo.setEnabled(s.relight)
        self.shading_slider.parentWidget().setEnabled(s.shading)
        self.lighting_slider.parentWidget().setEnabled(s.lighting)
        self.ai_check.setChecked(s.use_ai_director)
        self.ollama_url.setText(s.ollama_url)
        self.model_combo.setEditText(s.ollama_model)
        self.ai_min_spin.setValue(s.ai_min_scene_sec)
        self.dup_spin.setValue(s.dup_pixel_ratio * 100)
        self.mean_spin.setValue(s.dup_mean_diff)
        self.scene_spin.setValue(s.scene_hist_corr)
        self.maxgap_spin.setValue(s.max_interp_frames)
        self._set_combo(self.fast_combo, s.fast_motion_mode)

    def collect(self) -> Settings:
        s = Settings()
        s.input_path = self.in_edit.text().strip()
        s.output_path = self.out_edit.text().strip()
        s.container = self.container_combo.currentData()
        s.video_codec = self.codec_combo.currentData() or CODECS[s.container][0][1]
        s.quality = self.quality_slider.value()
        s.target_fps = self.fps_spin.value()
        s.backend = self.backend_combo.currentData()
        s.rife_scale = float(self.scale_combo.currentData())
        s.fp16 = self.fp16_check.isChecked()
        s.shading = self.shading_check.isChecked()
        s.shading_strength = self.shading_slider.value() / 100
        s.lighting = self.lighting_check.isChecked()
        s.lighting_strength = self.lighting_slider.value() / 100
        s.vignette = self.vignette_check.isChecked()
        s.dub = self.dub_check.isChecked()
        s.dub_vc = self.dub_vc_check.isChecked()
        s.dub_llm = self.dub_llm_combo.currentData() or "auto"
        s.dub_context = self.dub_ctx_edit.text().strip()
        s.dub_api_key = self.dub_key_edit.text().strip()
        s.restore = self.restore_check.isChecked()
        s.restore_res = self.restore_res_combo.currentData() or "source"
        s.relight = self.relight_check.isChecked()
        s.relight_strength = self.relight_slider.value() / 100
        s.light_dir = self.light_combo.currentData() or "auto"
        s.light_color = self.light_color_combo.currentData() or "auto"
        s.use_ai_director = self.ai_check.isChecked()
        s.ollama_url = self.ollama_url.text().strip() or Settings.ollama_url
        s.ollama_model = self.model_combo.currentText().strip() or Settings.ollama_model
        s.ai_min_scene_sec = self.ai_min_spin.value()
        s.dup_pixel_ratio = self.dup_spin.value() / 100
        s.dup_mean_diff = self.mean_spin.value()
        s.scene_hist_corr = self.scene_spin.value()
        s.max_interp_frames = self.maxgap_spin.value()
        s.fast_motion_mode = self.fast_combo.currentData() or "auto"
        return s

    @staticmethod
    def _set_combo(combo: QComboBox, data):
        i = combo.findData(data)
        if i >= 0:
            combo.setCurrentIndex(i)

    def _fill_codecs(self, container: str, prefer: str | None = None):
        prev = prefer or self.codec_combo.currentData()
        self.codec_combo.blockSignals(True)
        self.codec_combo.clear()
        for label, codec in CODECS[container]:
            self.codec_combo.addItem(label, codec)
        self._set_combo(self.codec_combo, prev)
        self.codec_combo.blockSignals(False)

    def _save_settings(self):
        self.s = self.collect()
        self.qs.setValue("settings", json.dumps(self.s.to_dict(), ensure_ascii=False))

    def _reset_advanced(self):
        d = Settings()
        self.dup_spin.setValue(d.dup_pixel_ratio * 100)
        self.mean_spin.setValue(d.dup_mean_diff)
        self.scene_spin.setValue(d.scene_hist_corr)
        self.maxgap_spin.setValue(d.max_interp_frames)

    # ------------------------------------------------------------------ 이벤트
    def _changed(self, *_):
        if self._loading:
            return
        self._save_timer.start()
        self._update_plan_stats()
        self.timeline.set_timeline(self.tl, self.collect())

    def _on_fps_changed(self, v: int):
        self._sync_presets(v)
        self._auto_output()
        self._changed()

    def _sync_presets(self, v: int):
        for b in self.preset_group.buttons():
            b.setChecked(self.preset_group.id(b) == v)

    def _on_container_changed(self, *_):
        c = self.container_combo.currentData()
        self._fill_codecs(c)
        out = self.out_edit.text().strip()
        if out:
            self.out_edit.setText(str(Path(out).with_suffix("." + c)))
        self._auto_output()
        self._changed()

    def _on_effects_changed(self, *_):
        self._auto_output()
        self._changed()
        self._preview_timer.start()

    def _on_out_edited(self, _):
        self._out_manual = True
        self._changed()

    def _auto_output(self):
        if self._loading or self._out_manual:
            return
        src = self.in_edit.text().strip()
        if not src:
            return
        p = Path(src)
        fx = "_fx" if (self.shading_check.isChecked() or self.lighting_check.isChecked()
                       or self.vignette_check.isChecked() or self.relight_check.isChecked()) else ""
        name = f"{p.stem}_{self.fps_spin.value()}fps{fx}.{self.container_combo.currentData()}"
        self.out_edit.setText(str(p.with_name(name)))

    def _browse_input(self):
        start = str(Path(self.in_edit.text()).parent) if self.in_edit.text() else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "영상 선택", start, VIDEO_FILTER)
        if path:
            self._set_input(path)

    def _browse_output(self):
        c = self.container_combo.currentData()
        path, _ = QFileDialog.getSaveFileName(self, "저장 위치", self.out_edit.text() or str(Path.home()),
                                              f"{c.upper()} (*.{c})")
        if path:
            if not path.lower().endswith("." + c):
                path += "." + c
            self.out_edit.setText(path)
            self._out_manual = True
            self._changed()

    def _set_input(self, path: str):
        path = os.path.normpath(path)
        if path != self.in_edit.text():
            self.tl = None
            self.tl_sig = None
            self.ai_done_model = None
            self._populate_results()
        self.in_edit.setText(path)
        self._out_manual = False
        self._auto_output()
        self._probe_input()
        self._changed()

    def _probe_input(self):
        path = self.in_edit.text()
        self.info_label.setText("영상 정보 읽는 중…")

        def ok(info):
            a = ", ".join(f"{x.codec}" for x in info.audio) or "없음"
            self.info_label.setText(
                f"{info.width}×{info.height} · {float(info.fps):.3f} fps · {fmt_time(info.duration)} · "
                f"약 {info.nb_frames_est:,} 프레임 · 영상 {info.codec} · 오디오 {a}")
        self._run_bg(probe, (path,), ok, lambda e: self.info_label.setText(f"⚠ 영상을 읽을 수 없습니다: {e[:200]}"))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if urls and not self._busy():
            self._set_input(urls[0])

    # ------------------------------------------------------------------ 환경 확인
    def _run_bg(self, fn, args, on_ok, on_err=None):
        t = FuncTask(fn, *args)
        t.result.connect(on_ok)
        if on_err:
            t.error.connect(on_err)
        t.finished.connect(lambda: self._bg.remove(t) if t in self._bg else None)
        self._bg.append(t)
        t.start()

    def _check_environment(self):
        ff = ffmpeg_bin("ffmpeg")
        if not Path(ff).exists() and ff == "ffmpeg":
            self._log("⚠ FFmpeg를 찾을 수 없습니다. tools/ffmpeg 폴더에 ffmpeg.exe를 넣어주세요.")

        def rife_ok(res):
            ok, msg = res
            self.rife_status.setText(("✔ RIFE " if ok else "✖ RIFE ") + msg)
        self._run_bg(rife_available, (), rife_ok, lambda e: self.rife_status.setText(f"✖ RIFE 확인 실패: {e}"))
        from ..dubbing import dub_available
        ok, msg = dub_available()
        self.dub_status.setText(("✔ 더빙 " if ok else "✖ 더빙 ") + msg)
        if not ok:
            self.dub_check.setChecked(False)
            self.dub_check.setEnabled(False)
        self._refresh_models()

    def _refresh_models(self):
        url = self.ollama_url.text().strip() or Settings.ollama_url
        self.ai_status.setText("Ollama 연결 확인 중…")

        def ok(models):
            cur = self.model_combo.currentText()
            self.model_combo.blockSignals(True)
            self.model_combo.clear()
            self.model_combo.addItems(models)
            self.model_combo.setEditText(cur)
            self.model_combo.blockSignals(False)
            if models:
                self.ai_status.setText(f"✔ Ollama 연결됨 · 모델 {len(models)}개. 이미지를 볼 수 있는 비전 모델"
                                       f"(qwen2.5vl, llava, gemma3 등)을 선택하세요.")
            else:
                self.ai_status.setText("Ollama는 켜져 있지만 모델이 없습니다. 터미널에서:  ollama pull qwen2.5vl:7b")

        def err(_):
            self.ai_status.setText("✖ Ollama 서버에 연결할 수 없습니다 (ollama.com 에서 설치 후 실행). "
                                   "AI 감독 없이도 변환은 가능합니다.")
        self._run_bg(list_models, (url,), ok, err)

    # ------------------------------------------------------------------ 결과 표시
    def _analysis_sig(self, s: Settings):
        return (s.input_path, s.analysis_width, s.dup_pixel_ratio, s.dup_mean_diff, s.scene_hist_corr)


    def _populate_results(self):
        tl = self.tl
        s = self.collect()
        self.timeline.set_timeline(tl, s)
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        if tl is None:
            for c in (self.card_src, self.card_keys, self.card_scenes, self.card_out, self.card_gen, self.card_est):
                c.set("-")
            self.table.blockSignals(False)
            self.preview_src.setText("장면을 선택하세요")
            self.preview_src.setPixmap(QPixmap())
            self.preview_fx.setPixmap(QPixmap())
            return
        enabled, _ = scene_interp_params(tl, s)
        risky = risky_keys(tl, s)
        self.table.setRowCount(len(tl.scenes))
        for r, sc in enumerate(tl.scenes):
            st = _scene_stats(tl, sc)
            hold = st["median_hold"]
            rhythm = ("1프레임 작화" if hold < 1.5 else "2프레임 작화" if hold < 2.5
                      else "3프레임 작화" if hold < 3.5 else f"정지 많음 ({hold:.0f}프레임)")
            ai = f"{sc.ai['scene_type']} · {sc.ai.get('reason', '')}" if sc.ai else "-"
            vals = [str(sc.id + 1), fmt_time(sc.start), fmt_time(sc.end), f"{sc.duration:.2f}s",
                    str(sc.n_keys), rhythm, self._motion_text(sc, risky), ai]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c < 7:
                    it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, it)
            chk = QTableWidgetItem("보간" if enabled[sc.id] else "안 함")
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            chk.setCheckState(Qt.Checked if enabled[sc.id] else Qt.Unchecked)
            chk.setToolTip("체크를 바꾸면 이 장면의 보간 여부를 직접 지정합니다 (AI 판단보다 우선).")
            self.table.setItem(r, 8, chk)
        self.table.blockSignals(False)

        self.card_src.set(f"{tl.n_frames:,}", f"{float(tl.info.fps):.3f} fps · {fmt_time(tl.duration)}")
        pct = tl.n_keys / max(1, tl.n_frames) * 100
        self.card_keys.set(f"{tl.n_keys:,}", f"원본의 {pct:.1f}% · 나머지는 같은 그림 반복")
        self.card_scenes.set(f"{len(tl.scenes):,}")
        self._update_plan_stats()
        if tl.scenes:
            self._select_scene_row(0)

    SPEED_DEFAULTS = {"rife": 0.02, "opencv": 0.08, "blend": 0.004}

    def _estimate(self, plan, s: Settings) -> float:
        v = self.qs.value(f"speed/{s.backend}")
        try:
            sec = float(v) if v is not None else self.SPEED_DEFAULTS.get(s.backend, 0.03)
        except (TypeError, ValueError):
            sec = self.SPEED_DEFAULTS.get(s.backend, 0.03)
        return estimate_render_seconds(plan, self.tl.info.width, self.tl.info.height, sec)

    @staticmethod
    def _motion_text(sc, risky) -> str:
        n = int(np.count_nonzero(risky[sc.first_key + 1: sc.last_key + 1]))
        return f"{sc.motion * 100:.2f}%" + (f" · 보호 {n}" if n else "")

    def _update_plan_stats(self):
        if self.tl is None:
            return
        s = self.collect()
        plan = build_plan(self.tl, s)
        out_dur = plan.n_out / float(plan.fps)
        self.card_out.set(f"{plan.n_out:,}", f"@ {plan.fps} fps")
        self.card_gen.set(f"{plan.n_generated:,}", f"출력의 {plan.n_generated / max(1, plan.n_out) * 100:.1f}%")
        est = self._estimate(plan, s)
        self.card_est.set(f"약 {fmt_dur(est)}", "이 PC에서 마지막으로 변환했을 때의 속도로 계산합니다 (처음엔 RTX 3060 기준값)")
        n_risky = int(np.count_nonzero(risky_keys(self.tl, s)))
        hist = self.tl.cadence_histogram()
        total = sum(hist.values()) or 1
        top = max(hist, key=hist.get)
        self.sync_label.setText(
            f"🔊 출력 길이 {out_dur:.3f}s = 원본 {self.tl.duration:.3f}s (오차 {abs(out_dur - self.tl.duration) * 1000:.1f}ms, "
            f"1프레임 미만) → 오디오 원본 그대로 싱크 유지   ·   주 작화 리듬: {top}프레임마다 새 그림 "
            f"({hist[top] / total * 100:.0f}%)"
            + (f"   ·   ⚡ 빠른 동작 {n_risky}곳은 원본 그림 유지" if n_risky else "")
            + (f"   ·   🎞 {self.tl.pulldown}" if self.tl.pulldown else ""))

    def _select_scene_row(self, sid: int):
        self.table.selectRow(sid)
        self.table.scrollToItem(self.table.item(sid, 0))

    def _on_table_selection(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows or self.tl is None:
            return
        sid = rows[0].row()
        self.timeline.select(sid)
        sc = self.tl.scenes[sid]
        if sc.thumb_first:
            self._set_pix(self.preview_src, sc.thumb_first)
        self._preview_timer.start()

    def _on_table_item_changed(self, item: QTableWidgetItem):
        if item.column() != 8 or self.tl is None:
            return
        sc = self.tl.scenes[item.row()]
        on = item.checkState() == Qt.Checked
        sc.manual_interp = on
        self.table.blockSignals(True)
        item.setText("보간" if on else "안 함")
        self.table.blockSignals(False)
        self._changed()

    def _set_pix(self, label: QLabel, jpeg: bytes):
        pm = QPixmap()
        pm.loadFromData(jpeg, "JPG")
        label.setPixmap(pm.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _update_preview(self):
        rows = self.table.selectionModel().selectedRows() if self.tl else []
        if not rows:
            return
        sc = self.tl.scenes[rows[0].row()]
        s = self.collect()
        if not (s.shading or s.lighting or s.vignette or s.relight):
            self.preview_fx.setPixmap(QPixmap())
            self.preview_fx.setText("효과가 꺼져 있습니다")
            return
        self._run_bg(preview_jpeg, (sc.thumb_first, s), lambda b: self._set_pix(self.preview_fx, b),
                     lambda e: self.preview_fx.setText(f"미리보기 실패: {e[:120]}"))

    # ------------------------------------------------------------------ 작업 실행
    def _busy(self) -> bool:
        return self.task is not None and self.task.isRunning()

    def _start(self, render: bool):
        if self._busy():
            return
        s = self.collect()
        if not s.input_path or not Path(s.input_path).exists():
            QMessageBox.warning(self, "입력 영상", "먼저 변환할 영상을 선택하세요.")
            return
        if render:
            if not s.output_path:
                QMessageBox.warning(self, "출력 파일", "저장할 파일 경로를 지정하세요.")
                return
            if os.path.normcase(os.path.abspath(s.output_path)) == os.path.normcase(os.path.abspath(s.input_path)):
                QMessageBox.warning(self, "출력 파일", "원본 파일과 다른 이름으로 저장해야 합니다.")
                return
            if Path(s.output_path).exists():
                if QMessageBox.question(self, "덮어쓰기", f"이미 파일이 있습니다. 덮어쓸까요?\n{s.output_path}") \
                        != QMessageBox.Yes:
                    return
            Path(s.output_path).parent.mkdir(parents=True, exist_ok=True)

        need_analysis = self.tl is None or self.tl_sig != self._analysis_sig(s)
        run_ai = s.use_ai_director and (need_analysis or self.ai_done_model != s.ollama_model)
        if need_analysis and self.tl is not None:
            self._log("분석 설정이 바뀌어 다시 분석합니다 (표에서 직접 지정한 보간 설정은 초기화됩니다).")
        self._save_settings()

        self.task = PipelineTask(s, None if need_analysis else self.tl, need_analysis, run_ai, render)
        self.task.log.connect(self._log)
        self.task.progress.connect(self._on_progress)
        self.task.analyzed.connect(lambda tl, s=s: self._on_analyzed(tl, s))
        self.task.done.connect(lambda st, s=s, run_ai=run_ai, render=render: self._on_done(st, s, run_ai, render))
        self.task.failed.connect(self._on_failed)
        pv = self.progress_view
        pv.reset()
        self.task.stage.connect(pv.set_stage)
        self.task.status.connect(self._on_status)
        self.task.preview.connect(pv.set_preview)
        self.task.plan_ready.connect(lambda plan, s=s: pv.set_plan(plan, self._estimate(plan, s)))
        self._set_busy(True)
        self.tabs.setCurrentIndex(1 if render else 0)
        self._log("─" * 60)
        self.task.start()

    def _cancel(self):
        if self._busy():
            self.task.stop()
            self.stage_label.setText("취소하는 중…")

    def _set_busy(self, busy: bool):
        self.start_btn.setEnabled(not busy)
        self.analyze_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        if not busy:
            self.progress.setRange(0, 1000)
            self.eta_label.setText("")

    def _on_status(self, st: dict):
        self.progress_view.set_status(st)
        if st["eta"] is not None and st["work_frac"] > 0.003:
            fin = time.strftime("%H:%M", time.localtime(time.time() + st["eta"]))
            self.eta_label.setText(f"프레임 {st['k']:,} / {st['n_out']:,}  ·  {st['fps']:.1f} fps  ·  "
                                   f"남은 시간 약 {fmt_dur(st['eta'])}  ·  완료 예상 {fin}")

    def _on_progress(self, stage: str, done: int, total: int):
        if stage != self._stage:
            self._stage = stage
            self._stage_t0 = time.perf_counter()
        self.stage_label.setText(stage)
        if total <= 0:
            self.progress.setRange(0, 0)
            return
        self.progress.setRange(0, 1000)
        self.progress.setValue(int(done / total * 1000))
        el = time.perf_counter() - self._stage_t0
        if stage.startswith("④"):
            return  # 렌더 단계의 남은 시간은 _on_status 에서 (AI 생성량 기준으로 더 정확)
        if done > 0 and el > 1.5:
            rate = done / el
            remain = (total - done) / rate if rate > 0 else 0
            self.eta_label.setText(f"{done:,} / {total:,}  ·  {rate:,.1f}/초  ·  남은 시간 약 {fmt_time(remain)[:-4]}")

    def _on_analyzed(self, tl: Timeline, s: Settings):
        self.tl = tl
        self.tl_sig = self._analysis_sig(s)
        self._populate_results()

    def _on_done(self, stats: dict, s: Settings, run_ai: bool, render: bool):
        self._set_busy(False)
        if run_ai:
            self.ai_done_model = s.ollama_model
        self.progress.setValue(1000)
        self.stage_label.setText("완료")
        if render and stats.get("sec_per_gen_mp"):
            self.qs.setValue(f"speed/{s.backend}", stats["sec_per_gen_mp"])
            self._update_plan_stats()
        if render and stats:
            mb = Path(s.output_path).stat().st_size / 1024 / 1024 if Path(s.output_path).exists() else 0
            box = QMessageBox(self)
            box.setWindowTitle("변환 완료")
            box.setText(f"변환이 끝났습니다.\n\n{s.output_path}\n\n{stats['frames']:,} 프레임 · {mb:,.1f} MB · "
                        f"{fmt_time(stats['seconds'])[:-4]} 소요"
                        + (f"\n\n한국어 더빙 파일:\n{stats['dub_path']}" if stats.get("dub_path") else ""))
            open_btn = box.addButton("폴더 열기", QMessageBox.AcceptRole)
            play_btn = box.addButton("영상 재생", QMessageBox.ActionRole)
            box.addButton("닫기", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is open_btn:
                self._open_output_folder()
            elif box.clickedButton() is play_btn:
                QDesktopServices.openUrl(QUrl.fromLocalFile(s.output_path))
        else:
            self.tabs.setCurrentIndex(0)

    def _on_failed(self, msg: str):
        self._set_busy(False)
        self.progress.setValue(0)
        self.stage_label.setText("중단됨")
        self._log("✖ " + msg)
        if "취소" not in msg:
            QMessageBox.critical(self, "오류", msg[:1500])

    def _open_output_folder(self):
        p = self.out_edit.text().strip()
        folder = Path(p).parent if p else ROOT
        if sys.platform == "win32" and p and Path(p).exists():
            import subprocess
            subprocess.Popen(["explorer", "/select,", os.path.normpath(p)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _log(self, text: str):
        self.log_view.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")

    def closeEvent(self, e):
        if self._busy():
            if QMessageBox.question(self, "종료", "작업이 진행 중입니다. 취소하고 종료할까요?") != QMessageBox.Yes:
                e.ignore()
                return
            self.task.stop()
            self.task.wait(15000)
        self._save_settings()
        super().closeEvent(e)


def _install_crash_handler():
    """콘솔 없이(pythonw) 실행되므로, 예상 못 한 오류는 파일에 남기고 창으로 알린다."""
    import traceback

    log_path = ROOT / "cache" / "error.log"

    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n{text}")
        except OSError:
            pass
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "AnimeFlow 오류",
                                 f"예상하지 못한 오류가 발생했습니다.\n자세한 내용: {log_path}\n\n{text[-1200:]}")

    sys.excepthook = hook


def main():
    _install_crash_handler()
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AnimeFlow.App")
        except (AttributeError, OSError):
            pass
    app = QApplication(sys.argv)
    app.setApplicationName("AnimeFlow")
    app.setStyleSheet(QSS)
    icon = ROOT / "assets" / "icon.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
