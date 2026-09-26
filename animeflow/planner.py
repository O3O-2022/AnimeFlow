"""출력 프레임 구조(플랜) 생성.

출력 프레임 k 는 정확히 t = k / 목표FPS 시점의 그림이다.
출력 전체 길이 = 원본 길이 이므로 오디오를 건드리지 않아도 싱크가 유지된다.

각 출력 프레임은 (키 A, 키 B, 비율 r) 로 표현된다.
  r == 0  → 키 A 그대로 (복사)
  0<r<1   → A와 B 사이 r 위치의 중간 프레임 생성
보간하지 않는 경우: 장면 전환(컷) 경계, 보간 꺼진 장면, 마지막 키,
그리고 '빠른 액션 보호'에 걸린 키 (보간 AI가 따라갈 수 없을 만큼 큰 움직임 → 원본의 박력을 유지).
긴 정지(홀드)는 끝부분 max_interp_frames 구간만 움직이도록 해서
'천천히 녹아내리는' 부자연스러운 모핑을 막는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .analyzer import Timeline
from .config import FAST_MOTION_MODES, INSTANT_CHANGE_AREA, INSTANT_CHANGE_RESIDUAL, Settings


SNAP = 0.02

# 출력 프레임 종류 (진행 화면 표시용)
KIND_COPY, KIND_GEN, KIND_CUT, KIND_PROTECT, KIND_OFF = range(5)
KIND_LABELS = {
    KIND_COPY: "원본 그림 유지",
    KIND_GEN: "AI 생성",
    KIND_CUT: "장면·순간 전환 (보간 안 함)",
    KIND_PROTECT: "빠른 동작 (원본 유지)",
    KIND_OFF: "보간 끈 장면",
}


@dataclass
class Plan:
    fps: Fraction
    key_a: np.ndarray  # int32
    key_b: np.ndarray  # int32
    ratio: np.ndarray  # float32
    kind: np.ndarray   # int8, KIND_*

    @property
    def n_out(self) -> int:
        return len(self.key_a)

    @property
    def n_generated(self) -> int:
        return int(np.count_nonzero(self.ratio > 0))


def scene_interp_params(tl: Timeline, s: Settings) -> tuple[np.ndarray, np.ndarray]:
    """장면별 (보간 여부, 최대 보간 길이[원본 프레임 수])."""
    n = len(tl.scenes)
    enabled = np.ones(n, dtype=bool)
    max_frames = np.full(n, s.max_interp_frames, dtype=np.float64)
    for sc in tl.scenes:
        if sc.ai:
            enabled[sc.id] = sc.ai.get("interpolate", True)
            max_frames[sc.id] = sc.ai.get("max_interp_frames", s.max_interp_frames)
        if sc.manual_interp is not None:
            enabled[sc.id] = sc.manual_interp
    return enabled, max_frames


def risky_keys(tl: Timeline, s: Settings) -> np.ndarray:
    """이전 그림에서 이 그림으로 넘어가는 움직임이 너무 커서 보간하면 잔상이 생길 키."""
    lim = FAST_MOTION_MODES.get(s.fast_motion_mode, FAST_MOTION_MODES["auto"])
    if lim is None:
        return np.zeros(tl.n_keys, dtype=bool)
    max_motion, max_residual = lim
    return (tl.key_motion > max_motion) | (tl.key_residual > max_residual)


def instant_change_keys(tl: Timeline) -> np.ndarray:
    """화면 대부분이 한 번에 바뀌는 '순간 전환' 키 (보간하면 두 장면이 뭉개진다)."""
    return (tl.key_change > INSTANT_CHANGE_AREA) & (tl.key_residual > INSTANT_CHANGE_RESIDUAL)


def build_plan(tl: Timeline, s: Settings) -> Plan:
    fps = Fraction(int(s.target_fps), 1)
    n_out = max(1, int(round(tl.duration * float(fps))))
    t = np.arange(n_out, dtype=np.float64) / float(fps) + 1e-6

    ks = tl.key_start
    a = np.clip(np.searchsorted(ks, t, side="right") - 1, 0, tl.n_keys - 1)
    b = np.minimum(a + 1, tl.n_keys - 1)

    enabled, max_frames = scene_interp_params(tl, s)
    scene_a = tl.key_scene[a]
    has_next = b > a
    same_scene = tl.key_scene[b] == scene_a
    risky = risky_keys(tl, s)[b]
    same_scene = same_scene & ~instant_change_keys(tl)[b]  # 순간 전환은 장면 전환처럼 취급
    ok = has_next & same_scene & enabled[scene_a] & ~risky

    gap = ks[b] - ks[a]
    window = np.minimum(gap, max_frames[scene_a] * tl.frame_dur)
    window = np.where(window > 0, window, 1.0)
    r = (t - (ks[b] - window)) / window
    r = np.where(ok, np.clip(r, 0.0, 1.0), 0.0)
    # 원본 그림 시점과 2% 이내(24fps 기준 약 1.7ms)면 눈으로 구분이 안 되므로 AI 생성 없이 원본 사용.
    # 23.976fps와 60fps처럼 격자가 조금씩 어긋날 때 생기는 '거의 원본' 프레임을 줄여 약 5% 빨라진다.
    r[r < SNAP] = 0.0

    # r≈1 이면 사실상 B → B를 그대로 쓰도록 정규화
    near_b = r > 1.0 - SNAP
    a = np.where(near_b, b, a)
    r[near_b] = 0.0

    kind = np.full(n_out, KIND_COPY, dtype=np.int8)
    kind[has_next & same_scene & ~enabled[scene_a]] = KIND_OFF
    kind[has_next & same_scene & enabled[scene_a] & risky] = KIND_PROTECT
    kind[has_next & ~same_scene] = KIND_CUT
    kind[r > 0] = KIND_GEN
    return Plan(fps=fps, key_a=a.astype(np.int32), key_b=b.astype(np.int32),
                ratio=r.astype(np.float32), kind=kind)


# RTX 3060 · RIFE 4.25 실측값 (전체 화면 0.028, 부분 영역 처리 포함 평균 추정) (메가픽셀당, 생성 프레임 1장 처리 시간). 변환을 한 번 하면 실제 값으로 갱신된다.
DEFAULT_SEC_PER_GEN_MP = 0.02
COPY_WEIGHT = 0.03


def estimate_render_seconds(plan: Plan, width: int, height: int, sec_per_gen_mp: float | None) -> float:
    n_gen = plan.n_generated
    work = n_gen + (plan.n_out - n_gen) * COPY_WEIGHT
    return work * (sec_per_gen_mp or DEFAULT_SEC_PER_GEN_MP) * width * height / 1e6
