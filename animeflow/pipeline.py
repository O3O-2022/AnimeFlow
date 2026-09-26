"""분석 → AI 감독 → 플랜 → 렌더(보간 + 효과 + 인코딩) 전체 흐름.

렌더는 3개 스레드가 동시에 일한다:
  디코더 스레드: 원본에서 '키 프레임'만 골라 큐에 넣음
  작업 스레드  : 플랜대로 복사/보간 (GPU)
  인코더 스레드: FFmpeg로 전송 (효과 필터 + 인코딩 + 원본 오디오 복사)
"""
from __future__ import annotations

import os
import queue
import threading
import time
from typing import Callable

import cv2
import numpy as np

from .analyzer import Cancelled, Timeline
from .config import Settings
from .ffmpeg_io import EncodeOptions, FrameReader, VideoWriter
from .effects import build_filter_chain, effects_enabled
from .gpu_output import GpuOutput, PendingFrame, cuda_available
from .interpolator import make_interpolator
from .planner import COPY_WEIGHT, KIND_GEN, Plan

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int], None]
StatusFn = Callable[[dict], None]
PreviewFn = Callable[[int, np.ndarray, np.ndarray], None]

PREVIEW_W = 640
MAX_BATCH = 8  # 한 번에 생성하는 최대 중간 프레임 수


def _thumb(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.resize(frame, (PREVIEW_W, max(2, int(h * PREVIEW_W / w))), interpolation=cv2.INTER_AREA)

_END = object()


def _video_offset(tl: Timeline) -> float:
    """원본에서 영상이 오디오보다 늦게 시작하면 그만큼 출력 영상도 늦춰 싱크 유지."""
    info = tl.info
    if not info.audio:
        return 0.0
    base = min([info.start] + [a.start for a in info.audio])
    return max(0.0, info.start - base)


def render(tl: Timeline, plan: Plan, s: Settings, progress: ProgressFn, log: LogFn,
           cancel: threading.Event, status: StatusFn | None = None,
           preview: PreviewFn | None = None) -> dict:
    """status: 0.5초마다 진행 정보(dict), preview: 0.4초마다 (출력 번호, 출력 썸네일, 원본 썸네일)."""
    info = tl.info
    interp = make_interpolator(s.backend, s.rife_scale, s.fp16, log)
    # 원본 보정: 고유 그림을 먼저 깨끗하게 만든 뒤 그것으로 중간 프레임을 생성
    restorer = None
    out_w, out_h = info.width, info.height
    if s.restore:
        from .restore import Restorer, output_size, restore_available
        if restore_available():
            out_w, out_h = output_size(info.width, info.height, s.restore_res)
            restorer = Restorer(out_w, out_h, fp16=s.fp16)
            log(f"원본 보정 사용 (애니 복원 AI) · 출력 {out_w}x{out_h}")
        else:
            log("원본 보정 모델이 없어 건너뜁니다 (third_party/restore).")
    fx_on = effects_enabled(s)
    # GPU 출력 단계: RIFE가 GPU에서 돌거나, 효과를 GPU로 처리할 수 있을 때
    gpu = None
    if cuda_available() and (fx_on or (interp.gpu_native and getattr(interp, "cuda", False))):
        gpu = GpuOutput(s, out_h, out_w, fp16=s.fp16)
    chain = build_filter_chain(s, out_h) if (fx_on and gpu is None) else ""
    if s.relight and gpu is None:
        log("입체 조명은 NVIDIA GPU가 있어야 사용할 수 있어 건너뜁니다.")
    elif s.relight:
        from .relight import depth_available
        if not depth_available():
            log("입체 조명 모델(Depth Anything V2)이 없어 건너뜁니다. README 참고.")
    opts = EncodeOptions(container=s.container, video_codec=s.video_codec, quality=s.quality,
                         filter_chain=chain, video_offset=_video_offset(tl),
                         width=out_w, height=out_h)
    fx_where = "없음" if not fx_on else ("GPU" if gpu else "CPU(FFmpeg)")
    log(f"보간 엔진: {interp.name} · 효과 처리: {fx_where}")
    use_key_tensor = gpu is not None and interp.gpu_native
    batched = use_key_tensor and hasattr(interp, "interpolate_batch")

    key_q: queue.Queue = queue.Queue(maxsize=6)
    out_q: queue.Queue = queue.Queue(maxsize=12)
    errors: list[BaseException] = []
    reader = FrameReader(info.path, info.width, info.height, "bgr24")
    writer = VideoWriter(info, s.output_path, plan.fps, opts)

    def decode_worker():
        try:
            key_src = tl.key_src
            ki = 0
            for idx, frame in enumerate(reader):
                if cancel.is_set() or errors:
                    break
                if ki < len(key_src) and idx == key_src[ki]:
                    key_q.put((ki, frame))
                    ki += 1
                    if ki >= len(key_src):
                        break
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        finally:
            key_q.put(_END)

    def encode_worker():
        # 오류가 나도 _END 가 올 때까지 큐를 계속 비워서 작업 스레드가 막히지 않게 한다
        last_preview = 0.0
        while True:
            item = out_q.get()
            if item is _END:
                break
            if errors:
                continue
            try:
                k, frame, src = item
                if isinstance(frame, PendingFrame):
                    frame = frame.get()
                writer.write(frame)
                now = time.perf_counter()
                if preview is not None and now - last_preview >= 0.4:
                    last_preview = now
                    preview(k, _thumb(frame), _thumb(src))
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

    t_dec = threading.Thread(target=decode_worker, daemon=True, name="decode")
    t_enc = threading.Thread(target=encode_worker, daemon=True, name="encode")
    t_dec.start()
    t_enc.start()

    cache: dict[int, np.ndarray] = {}
    decoded_end = False

    def get_key(i: int) -> np.ndarray:
        nonlocal decoded_end
        while i not in cache:
            if decoded_end:
                # 원본 끝까지 읽었는데 키가 없으면 가장 가까운 키로 대체
                return cache[max(k for k in cache if k <= i)] if cache else None
            item = key_q.get()
            if item is _END:
                decoded_end = True
                continue
            cache[item[0]] = restorer(item[1]) if restorer is not None else item[1]
        return cache[i]

    n_out = plan.n_out
    is_gen = plan.kind == KIND_GEN
    work_cum = np.cumsum(np.where(is_gen, 1.0, COPY_WEIGHT))
    work_total = float(work_cum[-1])
    gen_total = int(np.count_nonzero(is_gen))
    cuda_mem = None
    if gpu is not None:
        import torch
        cuda_mem = torch.cuda.mem_get_info

    def emit_status(k: int, elapsed: float):
        done = k + 1
        work_done = float(work_cum[k])
        eta = elapsed * (work_total - work_done) / work_done if work_done > 0 else None
        counts = np.bincount(plan.kind[:done], minlength=5)
        st = {
            "k": done, "n_out": n_out, "elapsed": elapsed, "eta": eta,
            "fps": done / elapsed if elapsed > 0 else 0.0,
            "gen_done": int(counts[KIND_GEN]), "gen_total": gen_total,
            "kind_counts": counts.tolist(),
            "video_time": done / float(plan.fps), "duration": tl.duration,
            "work_frac": work_done / work_total,
        }
        try:
            st["out_bytes"] = os.path.getsize(s.output_path)
        except OSError:
            st["out_bytes"] = 0
        if cuda_mem is not None:
            free, total = cuda_mem()
            st["gpu_used"], st["gpu_total"] = total - free, total
        status(st)

    t0 = time.perf_counter()
    last_status = last_prog = 0.0
    try:
        k = 0
        while k < n_out:
            if cancel.is_set():
                raise Cancelled()
            if errors:
                raise errors[0]
            a = int(plan.key_a[k])
            fa = get_key(a)
            if fa is None:
                raise RuntimeError("원본 프레임을 읽지 못했습니다.")
            if plan.ratio[k] > 0.0:
                # 같은 그림 쌍(a→b)의 연속된 생성 프레임을 묶어서 한 번에 처리
                b = int(plan.key_b[k])
                j = k + 1
                while (j < n_out and j - k < MAX_BATCH and plan.ratio[j] > 0.0
                       and plan.key_a[j] == a and plan.key_b[j] == b):
                    j += 1
                fb = get_key(b)
                ratios = [float(x) for x in plan.ratio[k:j]]
                if batched:
                    outs = interp.interpolate_batch(a, fa, b, fb, ratios)
                    # 효과(입체 조명)의 깊이 추정은 더 가까운 원본 그림 것을 재사용
                    frames = [gpu.finish(outs[i:i + 1], key=a if ratios[i] < 0.5 else b)
                              for i in range(len(ratios))]
                else:
                    frames = []
                    for r in ratios:
                        f = interp.interpolate(a, fa, b, fb, r)
                        if gpu is not None and gpu.has_effects:
                            f = gpu.finish(gpu.upload(f), key=a if r < 0.5 else b)
                        frames.append(f)
            else:
                j = k + 1
                if gpu is not None and gpu.has_effects:
                    frames = [gpu.finish(interp.key_tensor(a, fa) if use_key_tensor else gpu.upload(fa), key=a)]
                else:
                    frames = [fa]  # 원본 그림 그대로 (복사)
            for i, frame in enumerate(frames):
                out_q.put((k + i, frame, fa))
            last = j - 1
            k = j
            # 더 이상 필요 없는 키 제거 (메모리 절약)
            if len(cache) > 3:
                for old in [x for x in cache if x < a]:
                    del cache[old]
            now = time.perf_counter()
            if now - last_prog >= 0.2:
                last_prog = now
                progress(int(work_cum[last] / work_total * 1000), 1000)
                if status is not None and now - last_status >= 0.5:
                    last_status = now
                    emit_status(last, now - t0)
        out_q.put(_END)
        t_enc.join()
        if errors:
            raise errors[0]
        writer.finish()
    except BaseException:
        cancel.set()
        writer.abort()
        while t_enc.is_alive():
            try:
                out_q.put(_END, timeout=0.1)
                break
            except queue.Full:
                pass
        t_enc.join(timeout=5)
        try:
            os.remove(s.output_path)  # 미완성 파일 삭제
        except OSError:
            pass
        raise
    finally:
        reader.close()
        # 디코더가 put 에서 막혀 있으면 풀어준다
        while t_dec.is_alive():
            try:
                key_q.get(timeout=0.1)
            except queue.Empty:
                pass
        interp.close()
        if gpu is not None:
            gpu.close()

    elapsed = time.perf_counter() - t0
    if hasattr(interp, "stats_crop"):
        tot = interp.stats_crop + interp.stats_full
        if tot:
            log(f"그림 쌍 {tot:,}개 중 {interp.stats_crop:,}개({interp.stats_crop / tot * 100:.0f}%)는 "
                f"바뀐 영역만 처리해서 시간을 줄였습니다.")
    progress(1000, 1000)
    if status is not None:
        emit_status(n_out - 1, elapsed)
    return {"frames": n_out, "generated": plan.n_generated, "seconds": elapsed,
            "fps": n_out / max(elapsed, 1e-6),
            # 다음 작업의 예상 시간 계산용: 메가픽셀당 생성 프레임 1장 처리 시간(초)
            "sec_per_gen_mp": elapsed / max(1.0, work_total) / (info.width * info.height / 1e6)}


def describe_plan(tl: Timeline, plan: Plan) -> str:
    out_dur = plan.n_out / float(plan.fps)
    return (f"출력 {plan.n_out:,} 프레임 @ {plan.fps} fps · 생성 {plan.n_generated:,} · "
            f"복사 {plan.n_out - plan.n_generated:,} · 길이 {out_dur:.3f}s (원본 {tl.duration:.3f}s, "
            f"차이 {abs(out_dur - tl.duration) * 1000:.1f}ms)")

