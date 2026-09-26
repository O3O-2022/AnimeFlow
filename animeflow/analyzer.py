"""프로그램 기반(AI 아님) 프레임 분석.

1) 같은 그림이 반복되는 프레임(애니메이션의 2~3프레임 작화, 홀드)을 하나의 '키 프레임'으로 묶고
   시작 시간 ~ 끝 시간을 기록한다.
2) 장면 전환(컷)을 감지해 컷을 넘어서 보간하지 않도록 한다.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import Settings
from .ffmpeg_io import FrameReader, VideoInfo, frame_timestamps

ProgressFn = Callable[[int, int], None]


class Cancelled(Exception):
    pass


@dataclass
class Scene:
    id: int
    first_key: int
    last_key: int  # 포함
    start: float
    end: float
    thumb_first: bytes | None = None  # JPEG 소형 이미지 (메모리 절약)
    thumb_last: bytes | None = None
    motion: float = 0.0          # 키 사이 평균 변화량 (0~1)
    ai: dict | None = None       # AI 감독 판단 결과
    manual_interp: bool | None = None  # 사용자가 표에서 직접 지정한 값

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def n_keys(self) -> int:
        return self.last_key - self.first_key + 1


@dataclass
class Timeline:
    info: VideoInfo
    n_frames: int
    frame_dur: float                 # 원본 한 프레임 길이(초)
    duration: float                  # 전체 길이(초) - 출력도 정확히 이 길이
    key_src: np.ndarray              # 키 프레임의 원본 프레임 번호
    key_start: np.ndarray            # 키 프레임 시작 시간(초, 영상 시작 기준)
    key_end: np.ndarray              # 키 프레임 끝 시간(초)
    key_scene: np.ndarray            # 소속 장면 번호
    key_motion: np.ndarray           # 이전 그림에서 이동량 (화면 폭 대비 비율)
    key_residual: np.ndarray         # 움직임 추적 실패율 (0~1, 높으면 보간 AI도 실패할 가능성 큼)
    key_change: np.ndarray           # 이전 그림 대비 바뀐 면적 비율 (0~1)
    scenes: list[Scene] = field(default_factory=list)
    pulldown: str = ""               # 텔레시네 복원 설명 (예: "30fps → 24fps 작화 타이밍 복원")

    @property
    def n_keys(self) -> int:
        return len(self.key_src)

    def cadence_histogram(self) -> dict[int, int]:
        """키 하나가 몇 프레임 유지되는지 분포 (1=1프레임 작화, 2=2프레임 작화 ...)."""
        runs = np.diff(np.append(self.key_src, self.n_frames))
        runs = np.clip(runs, 1, 8)
        vals, counts = np.unique(runs, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    def save_json(self, path: Path):
        data = {
            "source": self.info.path,
            "n_frames": self.n_frames,
            "fps": str(self.info.fps),
            "duration": self.duration,
            "keys": [
                {"key": i, "src_frame": int(s), "start": round(float(a), 6),
                 "end": round(float(b), 6), "scene": int(sc),
                 "motion": round(float(m), 4), "track_fail": round(float(r), 3), "change": round(float(c), 4)}
                for i, (s, a, b, sc, m, r, c) in enumerate(zip(self.key_src, self.key_start, self.key_end,
                                                               self.key_scene, self.key_motion, self.key_residual,
                                                               self.key_change))
            ],
            "scenes": [
                {"scene": s.id, "start": s.start, "end": s.end, "keys": s.n_keys,
                 "motion": round(s.motion, 4), "ai": s.ai, "manual_interp": s.manual_interp}
                for s in self.scenes
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _analysis_size(info: VideoInfo, width: int) -> tuple[int, int]:
    w = min(width, info.width)
    h = int(round(info.height * w / info.width / 2)) * 2
    return w - (w % 2), max(2, h)


def _jpeg(bgr: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def _hist(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


class _MotionProbe:
    """저해상도 광학 흐름으로 그림 사이 이동량과 추적 실패율을 잰다 (키 프레임마다 1~3ms)."""

    def __init__(self, w: int, h: int):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.gx, self.gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.w = w

    def measure(self, prev: np.ndarray, cur: np.ndarray, changed: np.ndarray) -> tuple[float, float]:
        if not changed.any():
            return 0.0, 0.0
        f = self.dis.calc(prev, cur, None)
        mag = np.hypot(f[..., 0], f[..., 1])
        motion = float(np.percentile(mag[changed], 90)) / self.w
        warped = cv2.remap(cur, self.gx + f[..., 0], self.gy + f[..., 1], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        residual = float((cv2.absdiff(warped, prev)[changed] > 20).mean())
        return motion, residual


def _gap_regularity(gaps: np.ndarray) -> float:
    """연속된 두 간격이 같은 비율. 2프레임 작화면 2,2,2… 처럼 일정해야 자연스럽다."""
    if len(gaps) < 3:
        return 0.0
    return float(np.mean(gaps[1:] == gaps[:-1]))


def detect_pulldown(key_src: np.ndarray, key_scene: np.ndarray, fps: float):
    """24fps로 그린 애니를 30fps로 늘린 영상(1프레임씩 복제, 텔레시네)인지 확인하고
    그렇다면 각 그림의 원래 24fps 프레임 번호를 돌려준다.

    30fps에서는 그림 간격이 2,3,2,3… 처럼 들쭉날쭉해서 보간하면 속도가 빨라졌다 느려졌다 한다.
    """
    if not (29.0 < fps < 30.5) or len(key_src) < 40:
        return None
    k = key_src.astype(np.int64)
    same = key_scene[1:] == key_scene[:-1]
    g30 = np.diff(k)
    use = same & (g30 <= 4)
    if use.sum() < 30:
        return None
    reg30 = _gap_regularity(g30[use])
    best = None
    for phase in np.arange(0.0, 1.0, 0.05):
        j = np.floor(k * 0.8 + phase).astype(np.int64)
        g24 = np.diff(j)
        collisions = int(np.sum((g24 <= 0) & same))
        reg = _gap_regularity(g24[use])
        cand = (collisions, -reg, phase, j)
        if best is None or cand[:2] < best[:2]:
            best = cand
    collisions, neg_reg, phase, j = best
    reg24 = -neg_reg
    # 충돌(두 그림이 같은 24fps 칸에 들어감)이 거의 없고, 간격이 확실히 더 규칙적일 때만 적용
    if collisions <= max(1, int(0.01 * len(k))) and reg24 > reg30 + 0.1:
        return j, reg30, reg24
    return None


def analyze(info: VideoInfo, s: Settings, progress: ProgressFn | None = None,
            cancel: threading.Event | None = None, log: Callable[[str], None] = print) -> Timeline:
    aw, ah = _analysis_size(info, s.analysis_width)
    reader = FrameReader(info.path, info.width, info.height, "bgr24", scale_to=(aw, ah))
    n_pix = aw * ah
    min_changed = max(3, int(n_pix * s.dup_pixel_ratio))
    total_est = max(1, info.nb_frames_est)

    key_src: list[int] = []
    key_scene: list[int] = []
    key_motion: list[float] = []
    key_residual: list[float] = []
    key_change: list[float] = []
    probe = _MotionProbe(aw, ah)
    scenes: list[Scene] = []
    scene_motion: list[float] = []

    key_gray = None   # 현재 키 프레임 (흐림 처리된 흑백)
    key_hist = None
    key_img = None
    idx = -1
    try:
        for idx, frame in enumerate(reader):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)

            if key_gray is None:
                is_new_key, is_cut = True, True
            else:
                diff = cv2.absdiff(gray, key_gray)
                changed_mask = diff > 12
                changed = int(np.count_nonzero(changed_mask))
                mean = float(diff.mean())
                is_new_key = changed >= min_changed or mean >= s.dup_mean_diff
                is_cut = False
                if is_new_key:
                    hist = _hist(frame)
                    corr = float(cv2.compareHist(key_hist, hist, cv2.HISTCMP_CORREL))
                    is_cut = corr < s.scene_hist_corr or mean > 70
                    key_hist = hist
                    if not is_cut:
                        scene_motion.append(changed / n_pix)
                        mo, res = probe.measure(key_gray, gray, changed_mask)

            if is_new_key:
                if is_cut:
                    if scenes:
                        sc = scenes[-1]
                        sc.last_key = len(key_src) - 1
                        sc.thumb_last = _jpeg(key_img)
                        sc.motion = float(np.mean(scene_motion)) if scene_motion else 0.0
                    scenes.append(Scene(id=len(scenes), first_key=len(key_src), last_key=len(key_src),
                                        start=0.0, end=0.0, thumb_first=_jpeg(frame)))
                    scene_motion = []
                    if key_hist is None:
                        key_hist = _hist(frame)
                key_src.append(idx)
                key_scene.append(scenes[-1].id)
                key_motion.append(0.0 if is_cut else mo)
                key_residual.append(0.0 if is_cut else res)
                key_change.append(0.0 if is_cut else changed / n_pix)
                key_gray = gray
                key_img = frame.copy()

            if progress and idx % 50 == 0:
                progress(idx, max(total_est, idx + 1))
    finally:
        reader.close()

    n_frames = idx + 1
    if n_frames <= 0:
        raise RuntimeError("프레임을 읽지 못했습니다. 지원되지 않는 영상일 수 있습니다.")
    sc = scenes[-1]
    sc.last_key = len(key_src) - 1
    sc.thumb_last = _jpeg(key_img)
    sc.motion = float(np.mean(scene_motion)) if scene_motion else 0.0

    # 실제 표시 시간(PTS)을 사용해 가변 프레임레이트에서도 싱크를 유지
    frame_dur = 1.0 / info.fps_float
    times = None
    try:
        pts = frame_timestamps(info.path)
        if len(pts) == n_frames:
            times = pts - pts[0]
            if n_frames > 1:
                frame_dur = float(np.median(np.diff(times)))
        else:
            log(f"PTS 개수({len(pts)})와 프레임 수({n_frames})가 달라 고정 프레임레이트로 계산합니다.")
    except Exception as e:  # noqa: BLE001
        log(f"PTS 읽기 실패, 고정 프레임레이트로 계산: {e}")
    if times is None:
        times = np.arange(n_frames, dtype=np.float64) * frame_dur
    duration = float(times[-1] + frame_dur)

    ks = np.array(key_src, dtype=np.int64)
    kstart = times[ks]
    pulldown = ""
    pd = detect_pulldown(ks, np.array(key_scene), 1.0 / frame_dur)
    if pd is not None:
        j, reg30, reg24 = pd
        # 원래 24fps 타이밍으로 되돌림 (각 그림이 원본 프레임에서 1프레임 이내로만 움직임)
        f24 = 0.8 / frame_dur
        retimed = kstart[0] + (j - j[0]) / f24
        # 같은 장면 안에서 순서가 뒤바뀌지 않도록 보정
        retimed = np.maximum.accumulate(retimed)
        shift = np.abs(retimed - kstart).max()
        if shift <= 1.5 * frame_dur:
            kstart = retimed
            frame_dur = 1.0 / f24
            pulldown = (f"30fps 영상 속 24fps 작화 감지 → 원래 타이밍으로 복원 "
                        f"(그림 간격 규칙성 {reg30 * 100:.0f}% → {reg24 * 100:.0f}%)")
            log(pulldown)
    kend = np.append(kstart[1:], duration)
    for sc in scenes:
        sc.start = float(kstart[sc.first_key])
        sc.end = float(kend[sc.last_key])

    if progress:
        progress(n_frames, n_frames)
    return Timeline(info=info, n_frames=n_frames, frame_dur=frame_dur, duration=duration,
                    key_src=ks, key_start=kstart, key_end=kend,
                    key_scene=np.array(key_scene, dtype=np.int32),
                    key_motion=np.array(key_motion, dtype=np.float32),
                    key_residual=np.array(key_residual, dtype=np.float32),
                    key_change=np.array(key_change, dtype=np.float32), scenes=scenes,
                    pulldown=pulldown)
