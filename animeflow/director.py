"""로컬 AI(Ollama 비전 모델) '장면 감독'.

프로그램 분석 결과(시간, 작화 리듬, 움직임 양)와 장면의 첫/마지막 이미지를 AI에게 넘기면
AI가 장면 종류를 판단하고 보간 구조(보간할지, 얼마나 길게 보간할지)를 정한다.
실제 중간 프레임 생성은 RIFE(보간 전용 AI)가 담당한다.
"""
from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from typing import Callable

import numpy as np

from .analyzer import Cancelled, Scene, Timeline

PROMPT = """You are the frame-interpolation director for an anime video.
You get the FIRST and LAST drawing of one scene (cut) plus measured timing data.
Decide how the scene should be interpolated to a higher frame rate.

Timing data:
- scene time: {start:.2f}s ~ {end:.2f}s (length {dur:.2f}s)
- unique drawings in scene: {keys}
- average hold per drawing: {hold:.2f} source frames (1 = animated on ones, 2 = on twos, 3 = on threes)
- longest hold: {max_hold} source frames
- motion amount between drawings: {motion:.4f} (0 = still, >0.2 = big motion)
- drawings that move too far to interpolate safely (auto-protected, kept as original): {risky}

Rules of thumb:
- dialogue / slow character acting / camera pan / slow zoom -> interpolate, max_interp_frames 3-4
- fast action (superhero fights, punches, smear frames, impact frames, camera shake) -> interpolate
  but keep it snappy: max_interp_frames 2, so hits still feel strong
- western TV action (e.g. Invincible) is often animated on twos/threes with many held poses;
  held talking shots are fine to interpolate with max_interp_frames 3
- flashes, explosions, glitch effects, heavy particle effects, rapid cuts, text/credits -> do not interpolate
- still image with no motion -> interpolate false (nothing to do)

Reply ONLY with JSON:
{{"scene_type": "dialogue|action|pan|zoom|effect|still|other",
  "interpolate": true or false,
  "max_interp_frames": integer 1-6,
  "reason": "short reason in Korean"}}"""


def _http_json(url: str, payload: dict | None = None, timeout: float = 5.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def list_models(base_url: str) -> list[str]:
    """Ollama에 설치된 모델 목록. 서버가 꺼져 있으면 예외."""
    data = _http_json(base_url.rstrip("/") + "/api/tags", timeout=3)
    return [m["name"] for m in data.get("models", [])]


def _scene_stats(tl: Timeline, sc: Scene) -> dict:
    src = tl.key_src[sc.first_key: sc.last_key + 1]
    next_src = tl.key_src[sc.last_key + 1] if sc.last_key + 1 < tl.n_keys else tl.n_frames
    runs = np.diff(np.append(src, next_src))
    return {
        "start": sc.start, "end": sc.end, "dur": sc.duration, "keys": sc.n_keys,
        "hold": float(runs.mean()) if len(runs) else 1.0,
        "median_hold": float(np.median(runs)) if len(runs) else 1.0,
        "max_hold": int(runs.max()) if len(runs) else 1,
        "motion": sc.motion,
        "risky": int(np.count_nonzero((tl.key_motion[sc.first_key + 1: sc.last_key + 1] > 0.20)
                                      | (tl.key_residual[sc.first_key + 1: sc.last_key + 1] > 0.45))),
    }


def _sanitize(ans: dict) -> dict:
    out = {
        "scene_type": str(ans.get("scene_type", "other"))[:20],
        "interpolate": bool(ans.get("interpolate", True)),
        "reason": str(ans.get("reason", ""))[:200],
    }
    try:
        out["max_interp_frames"] = int(max(1, min(6, int(ans.get("max_interp_frames", 4)))))
    except (TypeError, ValueError):
        out["max_interp_frames"] = 4
    return out


def direct_scenes(tl: Timeline, base_url: str, model: str, min_scene_sec: float,
                  progress: Callable[[int, int], None] | None = None,
                  cancel: threading.Event | None = None,
                  log: Callable[[str], None] = print) -> int:
    """각 장면에 AI 판단(scene.ai)을 채운다. 반환: 판단한 장면 수."""
    url = base_url.rstrip("/") + "/api/chat"
    targets = [sc for sc in tl.scenes if sc.duration >= min_scene_sec and sc.n_keys > 1]
    done = 0
    failures = 0
    if not targets:
        return 0
    # 첫 호출 때 모델(약 6GB)을 GPU로 올리는 데 수십 초~몇 분 걸리므로 미리 불러둔다
    log("AI 모델을 GPU에 불러오는 중… (처음 한 번은 1~2분 걸릴 수 있음)")
    if progress:
        progress(0, 0)
    try:
        _http_json(base_url.rstrip("/") + "/api/generate", {"model": model, "keep_alive": "10m"}, timeout=600)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log(f"AI 모델을 불러오지 못했습니다 ({e}) → 기본 규칙으로 진행합니다.")
        return 0
    for i, sc in enumerate(targets):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        images = [base64.b64encode(b).decode() for b in (sc.thumb_first, sc.thumb_last) if b]
        prompt = PROMPT.format(**_scene_stats(tl, sc))
        try:
            resp = _http_json(url, {
                "model": model,
                "stream": False,
                "format": "json",
                "keep_alive": "10m",
                "options": {"temperature": 0.1, "num_predict": 200},
                "messages": [{"role": "user", "content": prompt, "images": images}],
            }, timeout=90)
            sc.ai = _sanitize(json.loads(resp["message"]["content"]))
            done += 1
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, OSError) as e:
            failures += 1
            log(f"  장면 {sc.id}: AI 판단 실패 ({e}) → 기본 규칙 사용")
            if failures >= 3 and done == 0:
                log("AI 서버 응답이 없어 AI 장면 감독을 건너뜁니다.")
                break
        if progress:
            progress(i + 1, len(targets))
    unload(base_url, model)
    return done


def unload(base_url: str, model: str):
    """프레임 생성(RIFE)에 GPU 메모리를 양보하도록 AI 모델을 즉시 내린다."""
    try:
        _http_json(base_url.rstrip("/") + "/api/generate", {"model": model, "keep_alive": 0}, timeout=10)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        pass
