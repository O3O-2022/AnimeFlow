"""백그라운드 작업 (GUI가 멈추지 않도록 QThread에서 실행)."""
from __future__ import annotations

import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ..analyzer import Cancelled, analyze
from ..config import CACHE_DIR, Settings
from ..director import direct_scenes
from ..ffmpeg_io import probe
from ..pipeline import describe_plan, render
from ..planner import build_plan


class PipelineTask(QThread):
    log = Signal(str)
    progress = Signal(str, int, int)      # 단계 이름, 완료, 전체
    analyzed = Signal(object)             # Timeline
    stage = Signal(str, str, str)         # 단계 키, 상태(run/done/skip/fail), 설명
    status = Signal(dict)                 # 렌더 진행 정보
    preview = Signal(int, object, object) # 출력 번호, 출력 썸네일, 원본 썸네일 (BGR)
    plan_ready = Signal(object)           # Plan
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, settings: Settings, timeline, need_analysis: bool, run_ai: bool, do_render: bool):
        super().__init__()
        self.s = settings
        self.tl = timeline
        self.need_analysis = need_analysis
        self.run_ai = run_ai
        self.do_render = do_render
        self.cancel = threading.Event()

    def stop(self):
        self.cancel.set()

    def _p(self, stage: str):
        return lambda d, t: self.progress.emit(stage, int(d), int(t))

    def run(self):
        s, log = self.s, self.log.emit
        self._cur = None
        try:
            tl = self.tl
            if self.need_analysis:
                self._begin("analyze", "영상 정보 확인 중")
                self.progress.emit("영상 정보 확인 중", 0, 0)
                info = probe(s.input_path)
                log(f"입력: {Path(s.input_path).name} · {info.width}x{info.height} · {float(info.fps):.3f}fps · "
                    f"{info.duration:.2f}s · 오디오 {len(info.audio)}개")
                tl = analyze(info, s, self._p("① 프레임 분석 (같은 그림 묶기 · 장면 전환 감지)"), self.cancel, log)
                uniq = tl.n_keys / tl.n_frames * 100
                log(f"분석 완료: 원본 {tl.n_frames:,} 프레임 → 고유 그림 {tl.n_keys:,}장 ({uniq:.1f}%), "
                    f"장면 {len(tl.scenes)}개")
                out = CACHE_DIR / f"{Path(s.input_path).stem}_timeline.json"
                tl.save_json(out)
                log(f"타임라인 저장: {out}")
                self.analyzed.emit(tl)
                self._end("analyze", f"그림 {tl.n_keys:,}장 · 장면 {len(tl.scenes)}개")
            else:
                self.stage.emit("analyze", "skip", "이전 분석 결과 사용")

            if self.run_ai:
                self._begin("ai", s.ollama_model)
                log(f"② AI 장면 감독 시작 (Ollama · {s.ollama_model})")
                n = direct_scenes(tl, s.ollama_url, s.ollama_model, s.ai_min_scene_sec,
                                  self._p("② AI 장면 감독 (장면별 보간 구조 결정)"), self.cancel, log)
                log(f"AI가 {n}개 장면의 보간 구조를 결정했습니다.")
                self.analyzed.emit(tl)
                self._end("ai", f"{n}개 장면 판단")
            else:
                self.stage.emit("ai", "skip", "사용 안 함" if not s.use_ai_director else "이전 결과 사용")

            if self.do_render:
                self._begin("plan", "")
                plan = build_plan(tl, s)
                log("③ " + describe_plan(tl, plan))
                self.plan_ready.emit(plan)
                self._end("plan", f"생성 {plan.n_generated:,} / 전체 {plan.n_out:,}")
                self._begin("render", "보간 AI 모델 준비 중…")
                self.progress.emit("④ 보간 AI 모델 준비 중 (RIFE)", 0, 0)
                stats = render(tl, plan, s, self._p("④ 프레임 생성 · 효과 · 인코딩"), log, self.cancel,
                               status=self.status.emit, preview=self.preview.emit)
                self._end("render", f"{stats['frames']:,} 프레임")
                log(f"완료! {stats['frames']:,} 프레임, {stats['seconds']:.1f}초 ({stats['fps']:.1f} fps 처리)")
                log(f"출력 파일: {s.output_path}")
                if s.dub:
                    from ..dubbing import run_dub
                    self._begin("dub", "대사 분리·받아쓰기·번역·음성 생성")
                    try:
                        dub_out = run_dub(s.output_path, s.ollama_url, s.dub_llm, s.dub_vc, s.dub_context,
                                          lambda st, d, t: self.progress.emit(st, d, t), log, self.cancel,
                                          api_key=s.dub_api_key)
                    except InterruptedError:
                        raise Cancelled()
                    stats["dub_path"] = dub_out
                    self._end("dub", "완료")
                    log(f"더빙 파일: {dub_out}")
                else:
                    self.stage.emit("dub", "skip", "사용 안 함")
                self.done.emit(stats)
            else:
                self.done.emit({})
        except Cancelled:
            self._fail("취소됨")
            self.failed.emit("사용자가 작업을 취소했습니다.")
        except Exception as e:  # noqa: BLE001
            self._fail("오류")
            self.log.emit(traceback.format_exc())
            self.failed.emit(str(e))

    def _begin(self, key: str, text: str):
        self._cur = key
        self.stage.emit(key, "run", text)

    def _end(self, key: str, text: str):
        self._cur = None
        self.stage.emit(key, "done", text)

    def _fail(self, text: str):
        if self._cur:
            self.stage.emit(self._cur, "fail", text)


class FuncTask(QThread):
    """간단한 함수 하나를 백그라운드에서 실행."""
    result = Signal(object)
    error = Signal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args

    def run(self):
        try:
            self.result.emit(self.fn(*self.args))
        except Exception as e:  # noqa: BLE001
            self.error.emit(str(e))
