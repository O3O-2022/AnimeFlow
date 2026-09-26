"""앱에서 한국어 AI 더빙 실행 (더빙은 라이브러리 충돌을 피하려고 별도 가상환경에서 돈다).

더빙 환경: %USERPROFILE%\\.animeflow\\dub-venv  (Demucs · Whisper · XTTS v2)
음색 변환 환경: %USERPROFILE%\\.animeflow\\vc-venv (Seed-VC, 없으면 음색 변환 없이 진행)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Callable

from .config import ROOT, ffmpeg_bin

HOME = Path(os.path.expanduser("~")) / ".animeflow"
DUB_PY = HOME / "dub-venv" / "Scripts" / "python.exe"
VC_PY = HOME / "vc-venv" / "Scripts" / "python.exe"
WORKER = ROOT / "animeflow" / "dub" / "worker.py"
PREFERRED_LLMS = ["exaone3.5:32b", "exaone3.5:7.8b"]
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def dub_available() -> tuple[bool, str]:
    if not DUB_PY.exists():
        return False, "더빙 환경이 없습니다 (dub-venv)"
    vc = "캐릭터 음색 변환 가능" if VC_PY.exists() else "음색 변환 환경 없음"
    return True, f"사용 가능 · {vc}"


def pick_llm(ollama_url: str, wanted: str) -> str:
    """'auto' 이면 설치된 모델 중 한국어 번역에 가장 좋은 것을 고른다."""
    if wanted and wanted != "auto":
        return wanted   # deepseek-chat / deepseek-reasoner 는 클라우드 (작업기가 직접 호출)
    try:
        with urllib.request.urlopen(ollama_url.rstrip("/") + "/api/tags", timeout=5) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except OSError:
        names = []
    for n in PREFERRED_LLMS:
        if n in names:
            return n
    return names[0] if names else PREFERRED_LLMS[-1]


def dubbed_path(video: str) -> str:
    p = Path(video)
    return str(p.with_name(f"{p.stem} (한국어 더빙){p.suffix}"))


def run_dub(video: str, ollama_url: str, llm: str, use_vc: bool, context: str,
            progress: Callable[[str, int, int], None], log: Callable[[str], None],
            cancel: threading.Event, api_key: str = "") -> str:
    out = dubbed_path(video)
    model = pick_llm(ollama_url, llm)
    cloud = model.startswith("deepseek-")
    if cloud and not api_key.strip():
        raise RuntimeError("DeepSeek 번역을 쓰려면 더빙 카드에 API 키를 넣어 주세요.")
    args = [str(DUB_PY), str(WORKER), "--video", video, "--out", out, "--ffmpeg", ffmpeg_bin("ffmpeg"),
            "--ollama", ollama_url, "--llm", model, "--vc", str(VC_PY) if (use_vc and VC_PY.exists()) else "", "--context", context or ""]
    if cloud:
        log("  번역에 DeepSeek 클라우드를 씁니다: 받아쓴 영어 대사 글자와 작품·인물 메모만 전송 (음성·영상은 보내지 않음)")
    log(f"한국어 더빙 시작 · 번역 AI {model} · 음색 변환 {'켬' if use_vc and VC_PY.exists() else '끔'}")
    # XTTS v2 라이선스(CPML, 비상업·개인 감상용)는 사용자가 동의함
    env = {**os.environ, "COQUI_TOS_AGREED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    if cloud:
        env["DEEPSEEK_API_KEY"] = api_key.strip()   # 명령줄에 남지 않도록 환경 변수로만 전달
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, creationflags=NO_WINDOW)
    tail: list[str] = []
    for raw in p.stdout:
        if cancel.is_set():
            p.kill()
            break
        line = raw.decode("utf-8", "replace").rstrip()
        if line.startswith("PROGRESS "):
            try:
                stage, done, total = line[len("PROGRESS "):].split("\t")
                progress(f"⑤ 더빙: {stage}", int(done), int(total))
            except ValueError:
                pass
        elif line and not any(k in line for k in ("Warning", "warn", "deprecat")):
            tail = (tail + [line])[-30:]
            if any(k in line for k in ("대사", "인물", "발음", "음색", "완료", "화자")):
                log("  " + line[:300])
    p.wait()
    if cancel.is_set():
        raise InterruptedError("더빙 취소")
    if p.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("더빙 실패:\n" + "\n".join(tail[-12:]))
    return out
