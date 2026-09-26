"""AI 더빙 작업기 (별도 가상환경 dub-venv 에서 실행).

영상 → 대사/배경 분리(Demucs) → 받아쓰기(Whisper) → 인물 구분 → 번역(Ollama)
     → 한국어 AI 음성(XTTS v2) → 시간 맞추기 → 배경과 합치기 → 영상에 음성 트랙 + 자막 추가

진행 상황은 표준출력에  PROGRESS <단계> <완료> <전체>  형식으로 알린다.
사용: python worker.py --video 입력.mp4 --out 출력.mp4 --ffmpeg ffmpeg.exe [--voice stock|clone]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SR = 44100
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

# XTTS v2 기본 제공 목소리 중 남/녀 (인물마다 다르게 배정)
MALE_VOICES = ["Damien Black", "Craig Gutsy", "Viktor Eka", "Andrew Chipper", "Badr Odhiambo",
               "Dionisio Schuyler", "Royston Min", "Abrahan Mack", "Adde Michal", "Baldur Sanjin"]
FEMALE_VOICES = ["Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema", "Alison Dietlinde",
                 "Ana Florence", "Annmarie Nele", "Asya Anara", "Brenda Stern", "Gitta Nikolina"]


def say(msg: str):
    print(msg, flush=True)


def progress(stage: str, done: int, total: int):
    print(f"PROGRESS {stage}\t{done}\t{total}", flush=True)


def run(args: list[str]):
    p = subprocess.run(args, capture_output=True, creationflags=NO_WINDOW)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[-1500:])


# ------------------------------------------------------------------ 1~2. 소리 추출·분리
def extract_and_separate(ffmpeg: str, video: str, work: Path) -> tuple[np.ndarray, np.ndarray]:
    wav = work / "mix.wav"
    run([ffmpeg, "-v", "error", "-y", "-i", video, "-map", "0:a:0", "-ac", "2", "-ar", str(SR), str(wav)])
    progress("대사와 배경음 분리 (Demucs)", 0, 0)
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs").cuda().eval()
    audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    x = torch.from_numpy(audio.T).cuda()
    ref = x.mean(0)
    x = (x - ref.mean()) / ref.std().clamp(min=1e-8)
    with torch.no_grad():
        out = apply_model(model, x[None], split=True, overlap=0.25, progress=False)[0]
    out = out * ref.std() + ref.mean()
    names = model.sources
    vocals = out[names.index("vocals")].cpu().numpy().T
    rest = sum(out[i] for i in range(len(names)) if names[i] != "vocals").cpu().numpy().T
    del model
    torch.cuda.empty_cache()
    return vocals, rest


# ------------------------------------------------------------------ 3. 받아쓰기
# 말이 아닌 소리(기합·비명·신음)만 있는 줄: 번역하지 않고 원래 목소리를 살린다
INTERJECTION = re.compile(
    r"^(?:[\s,.!?~-]*(?:a+h+|o+h+|h+a+h*|ha(?:ha)+|u+g+h+|a+r+g+h+|gr+|hm+|huh|hng+|nn+g*|e+h+|w+o+a+h+|whoa|"
    r"noo+|a+a+|y+a+h+|hi+ya+|ow+|ouch|oof|u+h+|u+m+|mm+|ho+)[\s,.!?~-]*)+$", re.I)

def transcribe(vocals: np.ndarray, work: Path) -> list[dict]:
    import whisper

    progress("받아쓰기 (Whisper)", 0, 0)
    mono = vocals.mean(1)
    path = work / "vocals16k.wav"
    import torchaudio.functional as AF
    v16 = AF.resample(torch.from_numpy(mono), SR, 16000).numpy()
    sf.write(str(path), v16, 16000)
    model = whisper.load_model("turbo", device="cuda")
    # 파일 경로를 넘기면 whisper가 PATH의 ffmpeg를 찾으므로, 이미 16kHz로 만든 데이터를 바로 넘긴다
    res = model.transcribe(v16.astype(np.float32), language="en", condition_on_previous_text=False, fp16=True,
                           word_timestamps=True, no_speech_threshold=0.6)
    del model
    torch.cuda.empty_cache()
    segs = []
    for s in res["segments"]:
        text = s["text"].strip()
        if not re.search(r"[A-Za-z]", text):          # "..." 같은 지어낸 대사 제거
            continue
        if INTERJECTION.match(text):                   # "Hah!", "Argh!" 같은 기합·비명은 원래 목소리를 그대로 쓴다
            continue
        if s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -0.8:
            continue
        words = [w for w in s.get("words", []) if w.get("word", "").strip()]
        if words:                                      # 단어 단위 시간으로 대사 시작·끝을 정확하게
            s["start"], s["end"] = words[0]["start"], words[-1]["end"]
        if s["end"] - s["start"] < 0.25:
            continue
        # 너무 조용한 구간(환각)은 제외
        a, b = int(s["start"] * SR), int(s["end"] * SR)
        if np.sqrt(np.mean(mono[a:b] ** 2) + 1e-12) < 0.004:
            continue
        segs.append({"start": float(s["start"]), "end": float(s["end"]), "en": text})
    return segs


# ------------------------------------------------------------------ 4. 인물 구분
def assign_speakers(tts, segs: list[dict], vocals: np.ndarray, work: Path):
    """목소리 특징(XTTS 화자 임베딩)으로 같은 인물끼리 묶고, 목소리 높낮이로 남/녀를 추정."""
    import torchaudio.functional as AF

    model = tts.synthesizer.tts_model
    mono = vocals.mean(1)
    embs, pitches = [], []
    for i, s in enumerate(segs):
        a, b = int(s["start"] * SR), int(s["end"] * SR)
        if b - a < int(0.5 * SR):  # 너무 짧으면 앞뒤를 조금 넓혀서 목소리 특징을 잡는다
            a, b = max(0, a - int(0.25 * SR)), b + int(0.25 * SR)
        clip = mono[a:min(b, len(mono))]
        p = work / f"seg_{i}.wav"
        if len(clip) < int(0.2 * SR):
            # 소리가 거의 없는 조각(받아쓰기 시간 오류) → 나중에 앞 대사의 인물로 넣는다
            embs.append(None)
            pitches.append(None)
            s["ref"] = segs[i - 1]["ref"] if i > 0 else ""
            continue
        sf.write(str(p), clip, SR)
        s["ref"] = str(p)
        _, emb = model.get_conditioning_latents(audio_path=[str(p)])
        e = emb.flatten().float().cpu().numpy()
        embs.append(e / (np.linalg.norm(e) + 1e-8))
        s["_emb"] = embs[-1]   # 나중에 번역 AI의 인물 판단을 목소리로 확인할 때 쓴다 (대본 파일에는 안 남김)
        progress("인물 구분", i + 1, len(segs))
    # 평균 연결 군집: 두 무리의 평균 유사도가 기준 이상이면 같은 인물로 합친다
    idx = [i for i, e in enumerate(embs) if e is not None]
    groups = [[i] for i in idx]
    E = {i: embs[i] for i in idx}

    def link(g1, g2):
        return float(np.mean([E[x] @ E[y] for x in g1 for y in g2]))

    while len(groups) > 1:
        best, pair = -1.0, None
        for x in range(len(groups)):
            for y in range(x + 1, len(groups)):
                v = link(groups[x], groups[y])
                if v > best:
                    best, pair = v, (x, y)
        if best < 0.55:
            break
        x, y = pair
        groups[x] += groups.pop(y)
    # 대사 1~2줄짜리 무리는 짧은 소리라 특징이 흔들린다 (소리치는 대사 등) → 조금 낮은 기준으로 가장 비슷한 무리에 합친다
    # (실측: 같은 인물 0.41~0.75, 다른 인물 0.19~0.34)
    while len(groups) > 1:
        small = sorted((g for g in groups if len(g) <= 2), key=len)
        merged = False
        for g in small:
            others = [h for h in groups if h is not g]
            h = max(others, key=lambda o: link(g, o))
            if link(g, h) >= 0.38:
                h += g
                groups.remove(g)
                merged = True
                break
        if not merged:
            break
    # XTTS 기본 목소리들의 특징과 비교해서, 원래 목소리와 가장 비슷한 같은 성별의 기본 목소리를 고른다 (복제 아님)
    bank = {}
    sm = model.speaker_manager
    for name in MALE_VOICES + FEMALE_VOICES:
        if name in sm.speakers:
            e = sm.speakers[name]["speaker_embedding"].flatten().float().cpu().numpy()
            bank[name] = e / (np.linalg.norm(e) + 1e-8)
    used = set()
    speakers = []
    for j, mem in enumerate(sorted(groups, key=len, reverse=True)):
        c = np.mean([E[k] for k in mem], 0)
        c /= np.linalg.norm(c) + 1e-8
        sim_m = np.mean(sorted([c @ bank[n] for n in MALE_VOICES if n in bank])[-3:])
        sim_f = np.mean(sorted([c @ bank[n] for n in FEMALE_VOICES if n in bank])[-3:])
        male = sim_m >= sim_f
        pool = [n for n in (MALE_VOICES if male else FEMALE_VOICES) if n in bank]
        cand = sorted(pool, key=lambda n: -(c @ bank[n]))
        voice = next((n for n in cand if n not in used), cand[0])
        used.add(voice)
        speakers.append({"id": j, "voice": voice, "male": bool(male), "lines": len(mem)})
        for k in mem:
            segs[k]["speaker"] = j
    if not speakers:
        speakers.append({"id": 0, "voice": MALE_VOICES[0], "male": True, "lines": len(segs)})
    # 목소리 특징을 못 잡은 조각은 바로 앞 대사의 인물로
    for i, s in enumerate(segs):
        if "speaker" not in s:
            s["speaker"] = segs[i - 1].get("speaker", 0) if i > 0 else 0
    return speakers


# ------------------------------------------------------------------ 5. 번역
HAN = re.compile(r"[぀-ヿ㐀-鿿]")   # 한자·일본어 (섞이면 안 됨)
HANGUL = re.compile(r"[가-힣]")


DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def is_cloud(model: str) -> bool:
    return model.startswith("deepseek-")


def _json_text(text: str) -> str:
    """추론 모델은 JSON 앞뒤에 설명을 붙이기도 하므로 { ... } 부분만 꺼낸다."""
    a, b = text.find("{"), text.rfind("}")
    return text[a:b + 1] if a >= 0 and b > a else text


def _chat_deepseek(model: str, prompt: str, as_json: bool, temperature: float, max_tokens: int | None) -> str:
    """DeepSeek 클라우드 API (사용자가 직접 넣은 키 사용, 대사 글자만 보낸다)."""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise OSError("DeepSeek API 키가 없습니다")
    body = {"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}]}
    if model != "deepseek-reasoner":        # R1은 온도·JSON 모드 설정을 받지 않는다
        body["temperature"] = temperature
        if as_json:
            body["response_format"] = {"type": "json_object"}
        if max_tokens:
            body["max_tokens"] = max_tokens
    req = urllib.request.Request(DEEPSEEK_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                text = json.loads(r.read())["choices"][0]["message"]["content"] or ""
            return _json_text(text) if as_json else text.strip()
        except urllib.error.HTTPError as e:
            if e.code in (401, 402, 403):
                raise SystemExit({401: "DeepSeek API 키가 잘못되었습니다", 402: "DeepSeek 계정 잔액이 부족합니다"}
                                 .get(e.code, "DeepSeek 접근이 거부되었습니다") + f" (HTTP {e.code})")
            last = e
        except OSError as e:
            last = e
    raise last


def _chat(url: str, model: str, prompt: str, as_json: bool, temperature: float = 0.3,
          max_tokens: int | None = None) -> str:
    if is_cloud(model):
        return _chat_deepseek(model, prompt, as_json, temperature, max_tokens)
    opts = {"temperature": temperature}
    if max_tokens:
        opts["num_predict"] = max_tokens
    body = {"model": model, "stream": False, "keep_alive": "10m", "options": opts,
            "messages": [{"role": "user", "content": prompt}]}
    if as_json:
        body["format"] = "json"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())["message"]["content"]


SYL_PER_SEC = 6.0   # 한국어 더빙 대사 속도 (초당 글자 수, 약간 빠른 편)


def hangul_len(text: str) -> int:
    return len(HANGUL.findall(text))


def line_slot(segs: list[dict], i: int) -> float:
    """이 대사가 쓸 수 있는 시간: 원래 대사 길이 + 다음 대사 전까지 여유(최대 0.4초)."""
    s = segs[i]
    nxt = segs[i + 1]["start"] if i + 1 < len(segs) else s["end"] + 1.0
    return (s["end"] - s["start"]) + max(0.0, min(0.4, nxt - s["end"]))


def char_budget(segs: list[dict], i: int) -> int:
    prev_end = segs[i - 1]["end"] if i > 0 else 0.0
    pre = max(0.0, min(0.3, segs[i]["start"] - prev_end - 0.08))
    return max(3, int((line_slot(segs, i) + pre) * SYL_PER_SEC))


def _first(text: str) -> str:
    """AI 답에서 대사 한 줄만 (따옴표·빈 줄 제거)."""
    for ln in text.strip().splitlines():
        ln = ln.strip().strip('"').strip()
        if ln:
            return ln
    return ""


def _ok_korean(text: str) -> bool:
    return bool(text) and bool(HANGUL.search(text)) and not HAN.search(text)


NAME_MAP = {"Dad": "아빠", "Daddy": "아빠", "Mom": "엄마", "Mommy": "엄마", "Father": "아버지", "Mother": "어머니"}
PARENT_WORDS = {"아빠": ("dad", "daddy", "father", "pop", "papa"), "아버지": ("dad", "father"),
                "엄마": ("mom", "mommy", "mother", "mum", "mama"), "어머니": ("mom", "mother")}


def drop_added_vocatives(en: str, ko: str) -> str:
    """부르는 말(', 아빠?' / '아빠, ...')만 지운다. '아빠가 있잖아' 같은 문장 속 단어는 그대로."""
    low = en.lower()
    for k, ens in PARENT_WORDS.items():
        if k in ko and not any(re.search(rf"\b{e}\b", low) for e in ens):
            new = re.sub(rf"\s*,\s*{k}(?=\s*[?!.~]*\s*$)", "", ko)
            new = re.sub(rf"^\s*{k}\s*,\s*", "", new)
            if _ok_korean(new):
                ko = new
    return ko


def identify_voices(segs: list[dict], url: str, model: str, user_ctx: str) -> dict[int, str]:
    """목소리 무리마다 누구인지 번역 AI가 정한다 (대화 순서 + 목소리별 대사를 함께 보여준다)."""
    voices = sorted({s.get("speaker", 0) for s in segs})
    by_voice = "\n".join(f"[목소리{v + 1}]\n" + "\n".join(f"  - {s['en']}" for s in segs if s.get("speaker", 0) == v)
                         for v in voices)
    in_order = "\n".join(f"{i + 1}. [목소리{s.get('speaker', 0) + 1}] {s['en']}" for i, s in enumerate(segs))
    names: dict[int, str] = {}
    try:
        ans = json.loads(_chat(url, model, (
            user_ctx +
            "다음은 애니메이션 한 장면의 영어 대사다. 같은 목소리 번호는 같은 사람이다.\n"
            "각 목소리가 누구인지 정해라. 근거(reason)를 먼저 쓰고 이름(name)을 써라.\n"
            "[판단 단서]\n"
            "- 어떤 목소리가 누군가를 이름으로 부르면(예: 'Think, Mark!') 그 목소리는 그 이름의 인물이 아니다\n"
            "- 'Dad', 'Mom'이라고 부르는 목소리는 그 사람의 자식이다\n"
            "- 질문 바로 다음 줄이 그 질문에 대한 대답이면, 대답한 목소리는 질문한 사람과 다른 사람이다\n"
            "- 서로 다른 목소리 번호는 서로 다른 사람일 가능성이 크다. 같은 이름은 확실할 때만\n"
            "- 위 정보에 나온 이름을 한글로 그대로 써라. 알 수 없으면 '인물N'\n\n"
            '반드시 JSON으로만: {"voices": [{"voice": 1, "reason": "...", "name": "..."}, ...]}\n\n'
            f"[대화 순서]\n{in_order}\n\n[목소리별로 모은 대사]\n{by_voice}"),
            as_json=True, temperature=0.1, max_tokens=250 + 150 * len(voices)))
        for x in ans.get("voices", []):
            names[int(x["voice"]) - 1] = str(x.get("name", "")).strip()
    except (OSError, ValueError, KeyError, TypeError) as e:
        say(f"인물 파악 건너뜀: {e}")
    for v in voices:
        # '옴니맨(Dad)' 처럼 괄호 호칭이 붙으면 번역 AI가 그 호칭을 대사에 끼워 넣으므로 이름만 남긴다
        names[v] = re.sub(r"\(.*?\)|\[.*?\]", "", names.get(v) or "").strip()
    return check_names_by_voice(segs, voices, names)


def _voice_link(segs: list[dict], a: int, b: int) -> float | None:
    ea = [s["_emb"] for s in segs if s.get("speaker") == a and s.get("_emb") is not None]
    eb = [s["_emb"] for s in segs if s.get("speaker") == b and s.get("_emb") is not None]
    if not ea or not eb:
        return None
    return float(np.mean([x @ y for x in ea for y in eb]))


def _voice_min_inner(segs: list[dict], a: int) -> float | None:
    """한 목소리 무리 안에서 가장 덜 비슷한 대사 쌍의 유사도 (같은 사람이 낼 수 있는 최저치)."""
    e = [s["_emb"] for s in segs if s.get("speaker") == a and s.get("_emb") is not None]
    if len(e) < 2:
        return None
    return float(min(e[i] @ e[j] for i in range(len(e)) for j in range(i + 1, len(e))))


def merge_named_voices(segs: list[dict], speakers: list[dict]) -> list[dict]:
    """번역 AI가 같은 인물이라고 했고 목소리도 어느 정도 비슷한 무리(0.35 이상)는 하나로 합친다.
    (같은 인물 대사가 다른 참고 음성으로 만들어져 목소리가 바뀌는 것 방지. 이름만으로는 합치지 않는다)"""
    by_id = {sp["id"]: sp for sp in speakers}
    changed = True
    while changed:
        changed = False
        ids = sorted({s["speaker"] for s in segs})
        for a in ids:
            for b in ids:
                if a >= b:
                    continue
                na = next(s["role"]["speaker"] for s in segs if s["speaker"] == a)
                nb = next(s["role"]["speaker"] for s in segs if s["speaker"] == b)
                link = _voice_link(segs, a, b)
                if na == nb and not na.startswith("인물") and link is not None and link >= 0.35:
                    keep, drop = (a, b) if by_id[a]["lines"] >= by_id[b]["lines"] else (b, a)
                    for s in segs:
                        if s["speaker"] == drop:
                            s["speaker"] = keep
                    by_id[keep]["lines"] += by_id[drop]["lines"]
                    say(f"같은 인물({na})로 합침: 목소리{drop + 1} → 목소리{keep + 1} (유사도 {link:.2f})")
                    changed = True
                    break
            if changed:
                break
    used = sorted({s["speaker"] for s in segs})
    remap = {old: new for new, old in enumerate(used)}
    for s in segs:
        s["speaker"] = remap[s["speaker"]]
    out = []
    for old in used:
        sp = dict(by_id[old])
        sp["id"] = remap[old]
        sp["name"] = next(s["role"]["speaker"] for s in segs if s["speaker"] == sp["id"])
        out.append(sp)
    return out


def check_names_by_voice(segs: list[dict], voices: list[int], names: dict[int, str]) -> dict[int, str]:
    """대사 1~2줄짜리 목소리 무리는 번역 AI도 누구인지 헷갈린다 (실측: 같은 대사인데 실행마다 옴니맨/모름으로 바뀜).
    이름이 '모름'이거나 다른 무리와 이름이 겹치는데 목소리는 다른 이름의 무리와 더 비슷하면, 그 무리의 이름을 따른다."""
    def unknown(n: str) -> bool:
        return not n or bool(re.fullmatch(r"(인물\s*(N|\d+)|unknown|모름|알 수 없음)", n, re.I))

    count = {v: sum(1 for s in segs if s.get("speaker") == v) for v in voices}
    for v in sorted(voices, key=lambda x: count[x]):
        if count[v] > 2:
            continue
        links = {u: _voice_link(segs, v, u) for u in voices if u != v}
        links = {u: x for u, x in links.items() if x is not None and not unknown(names[u])}
        if not links:
            continue
        nearest = max(links, key=links.get)
        same = [links[u] for u in links if names[u] == names[v]]
        # 같은 이름이라는 큰 무리(3줄 이상)와 비교: 그 무리 안의 대사끼리보다 확실히 덜 비슷하면 같은 사람이 아니다
        # (실측: 옴니맨 대사끼리 최소 0.43, 마크 대사와 옴니맨은 0.30)
        def not_same(u):   # 목소리로 봐서 u 무리와는 확실히 다른 사람
            own = _voice_min_inner(segs, u) if count[u] >= 3 else None
            return own is not None and links[u] < own - 0.05

        doubtful = any(names[u] == names[v] and not_same(u) for u in links)
        if unknown(names[v]):
            ok = [u for u in links if not not_same(u)]
            if ok:
                names[v] = names[max(ok, key=links.get)]
        elif doubtful:
            alt = [u for u in links if names[u] != names[v]]
            names[v] = names[max(alt, key=links.get)] if alt else ""
        elif same and names[nearest] != names[v] and links[nearest] > max(same) + 0.05:
            names[v] = names[nearest]
    for v in voices:
        if unknown(names[v]):
            names[v] = f"인물{v + 1}"
    return names


def translate(segs: list[dict], ollama: str, model: str, context: str = ""):
    url = ollama.rstrip("/") + "/api/chat"

    if not is_cloud(model):
        progress("번역 AI 불러오는 중 (큰 모델은 1~3분)", 0, 0)
        try:
            urllib.request.urlopen(urllib.request.Request(
                url.replace("/api/chat", "/api/generate"),
                data=json.dumps({"model": model, "keep_alive": "10m"}).encode(),
                headers={"Content-Type": "application/json"}), timeout=1800).read()
        except OSError as e:
            say(f"번역 AI 불러오기 실패: {e}")
    user_ctx = f"[사용자가 알려준 작품·인물 정보 — 추측보다 이 정보를 우선할 것]\n{context}\n\n" if context else ""
    # 1) 목소리 무리마다 누구인지 정한다. 같은 목소리 = 같은 사람이므로 무리의 대사 전체를 근거로 한 번에 판단
    #    (대사 줄마다 따로 추측시키면 실행할 때마다 결과가 흔들리고 자주 틀렸다)
    progress("번역 준비: 목소리마다 누구인지 파악", 0, 0)
    voices = sorted({s.get("speaker", 0) for s in segs})
    names = identify_voices(segs, url, model, user_ctx)
    say("인물 파악: " + ", ".join(f"목소리{v + 1}={names[v]}" for v in voices))
    script = "\n".join(f"[{names[s.get('speaker', 0)]}] {s['en']}" for s in segs)

    # 2) 관계·말투·분위기 정리
    progress("번역 준비: 인물 관계 파악", 0, 0)
    notes = _chat(url, model, (
        user_ctx +
        "다음은 애니메이션 한 장면의 영어 대사다. [ ] 안이 말하는 사람이다.\n"
        "한국어로 짧게 정리해라:\n"
        "1) 인물 사이 관계와 한국어 말투 (예: 아빠→아들 반말, 아들→아빠 반말)\n"
        "2) 장면 분위기 한 줄\n\n" + script), as_json=False, temperature=0.2, max_tokens=300)
    say("장면 분석: " + notes.replace("\n", " ")[:300])

    # 3) 대사마다 듣는 사람·말투 (말하는 사람은 목소리로 이미 정해짐)
    progress("번역 준비: 대사마다 누구에게 하는 말인지 파악", 0, 0)
    main_names = list(dict.fromkeys(names[v] for v in voices))   # 같은 이름이 여러 목소리에 있어도 한 번만
    numbered = "\n".join(f"{i}. [{names[s.get('speaker', 0)]}] {s['en']}" for i, s in enumerate(segs))
    roles = {}
    try:
        ans = json.loads(_chat(url, model, (
            "다음은 애니메이션 한 장면의 영어 대사다. [ ] 안이 말하는 사람이다 (확실함).\n"
            "각 줄마다 듣는 사람(listener)과 한국어 말투(tone: 반말/존댓말)를 정해라.\n"
            "- 가족·친구·적 사이는 반말, 처음 보는 어른·상사에게는 존댓말. 확실하지 않으면 반말\n"
            + user_ctx + f"[장면 정보]\n{notes}\n\n"
            '반드시 JSON으로만: {"lines": [{"id": 0, "listener": "누구", "tone": "반말"}, ...]}\n\n'
            + numbered), as_json=True, temperature=0.1, max_tokens=200 + 40 * len(segs)))
        for x in ans.get("lines", []):
            if "id" in x:
                roles[int(x["id"])] = x
    except (OSError, ValueError, KeyError, TypeError) as e:
        say(f"듣는 사람 파악 건너뜀: {e}")
    for i, s in enumerate(segs):
        r = roles.get(i, {})
        spk = names[s.get("speaker", 0)]
        others = [n for n in main_names if n != spk]
        listener = others[0] if len(main_names) == 2 and others else str(r.get("listener", ""))
        tone = str(r.get("tone", "반말"))
        s["role"] = {"speaker": spk, "listener": listener, "tone": tone if tone in ("반말", "존댓말") else "반말"}
    say("화자: " + " / ".join(f"{s['role']['speaker']}→{s['role']['listener']}" for s in segs)[:300])

    # 2) 번역
    B = 30
    for start in range(0, len(segs), B):
        chunk = segs[start:start + B]
        lines = [{"id": i, "화자": s.get("role", {}).get("speaker", ""), "청자": s.get("role", {}).get("listener", ""),
                  "말투": s.get("role", {}).get("tone", "반말"), "en": s["en"],
                  "max_chars": char_budget(segs, start + i)} for i, s in enumerate(chunk)]
        prompt = (
            "너는 한국 애니메이션 더빙 전문 번역가다. 영어 대사를 한국 성우가 그대로 읽을 더빙 대본으로 옮겨라.\n"
            + user_ctx +
            f"[장면 정보]\n{notes}\n\n"
            "[규칙]\n"
            "- 반드시 한국어(한글)로만. 한자·중국어·일본어 절대 금지\n"
            "- 직역하지 말고 뜻을 살려 자연스러운 구어체로 (예: 'I've got your back.' → '내가 있잖아.')\n"
            "- 번역투 금지: '당신', '~하는 것이다', '그것은' 같은 표현 쓰지 말 것\n"
            "- 각 줄의 '화자'가 '청자'에게 하는 말이다. 1인칭·2인칭과 호칭을 그 관계에 맞게 (예: 딸이 엄마에게 'I miss you' → '엄마 보고 싶어')\n"
            "- '말투'(반말/존댓말)를 반드시 지킬 것\n"
            "- 원문에 있는 이름·호칭(Mark, Dad 등)은 빼지 말 것. 글자 수가 부족하면 다른 말을 줄여라\n"
            "- 'max_chars'는 입 움직임 시간 안에 말할 수 있는 최대 한글 글자 수다. 반드시 그 이하로\n"
            "- 먼저 'meaning'에 원문의 진짜 뜻(속뜻·뉘앙스)을 한국어로 풀어 쓰고, 그 뜻을 살린 더빙 대사를 'ko'에 써라\n"
            "[좋은 예] (직역 → 더빙)\n"
            "- 'What's left for you when it's all over?' → meaning: 다 끝나고 나면 너한테 뭐가 남아 있겠냐는 뜻 → ko: '다 끝나면 너한테 뭐가 남는데?'\n"
            "- 'You don't get it, do you?' → meaning: 아직도 이해 못 했냐며 답답해하는 말 → ko: '아직도 모르겠어?'\n"
            "- 'I'm not going anywhere.' → meaning: 떠나지 않고 곁에 있겠다는 뜻 → ko: '난 어디 안 가.'\n"
            "- 'Look at me!' → ko: '날 봐!'\n"
            "- 'You did this to yourself.' → meaning: 네가 자초한 일이라는 뜻 → ko: '다 네가 자초한 거야.'\n"
            '반드시 JSON으로만: {"lines": [{"id": 0, "meaning": "...", "ko": "..."}, ...]}\n\n'
            + json.dumps(lines, ensure_ascii=False))
        try:
            ans = json.loads(_chat(url, model, prompt, as_json=True, max_tokens=300 + 150 * len(chunk)))
            got = {int(x["id"]): str(x.get("ko", "")).strip() for x in ans.get("lines", []) if "id" in x}
        except (OSError, ValueError, KeyError) as e:
            say(f"번역 오류, 한 줄씩 다시 시도: {e}")
            got = {}
        for i, s in enumerate(chunk):
            s["ko"] = got.get(i, "")
        progress(f"번역 ({model})", min(start + B, len(segs)), len(segs))

    # 3) 검증: 한글이 아니거나 한자·일본어가 섞인 줄은 그 줄만 다시 번역
    for i, s in enumerate(segs):
        tries = 0
        while not _ok_korean(s["ko"]) and tries < 3:
            prev = segs[i - 1]["en"] if i > 0 else ""
            nxt = segs[i + 1]["en"] if i + 1 < len(segs) else ""
            s["ko"] = _first(_chat(url, model, (
                f"[장면 정보]\n{notes}\n\n앞 대사: {prev}\n뒤 대사: {nxt}\n\n"
                f"다음 영어 대사 한 줄을 자연스러운 한국어 더빙 대사로 옮겨라. 한글로만, 설명 없이 대사만 출력:\n"
                f"[{s['role']['speaker']}→{s['role']['listener']}, {s['role']['tone']}] {s['en']}"), as_json=False, temperature=0.4 + 0.2 * tries
            ))
            tries += 1
        if not _ok_korean(s["ko"]):
            say(f"  번역 실패, 원문 유지: {s['en']}")
            s["ko"] = ""
            continue
        budget = char_budget(segs, i)
        for _ in range(2):
            if hangul_len(s["ko"]) <= budget * 1.15:
                break
            short = _first(_chat(url, model, (
                f"다음 한국어 더빙 대사를 뜻과 말투는 그대로 두고 한글 {budget}글자 이내로 줄여라. "
                f"한글로만, 설명 없이 대사만 출력:\n원문: {s['en']}\n대사: {s['ko']}"), as_json=False, temperature=0.3
            ))
            if _ok_korean(short) and hangul_len(short) < hangul_len(s["ko"]):
                s["ko"] = short
    # 원문에 없는 부모 호칭을 번역이 덧붙였으면 뺀다 (예: 아빠가 한 말 끝에 ', 아빠?')
    for s in segs:
        if s.get("ko"):
            s["ko"] = drop_added_vocatives(s["en"], s["ko"])
    # 이름·호칭 확인: 영어에 이름(Mark)·호칭(Dad)이 있는데 번역에서 빠졌으면 넣어서 다시
    ko_names: dict[str, str] = {}
    for i, s in enumerate(segs):
        if not s.get("ko"):
            continue
        words = re.findall(r"\b[A-Z][a-z]+\b", s["en"])
        names = [w for k, w in enumerate(words) if w not in ("I", "Oh", "Hey", "No", "Yes", "Okay", "Why", "What",
                                                             "You", "Think", "Look", "Come", "Stop", "Please", "The")
                 and (k > 0 or w in NAME_MAP or s["en"].strip().startswith(w + ",") or s["en"].strip().rstrip("?!.") == w)]
        for n in names:
            if n not in ko_names:
                ko_names[n] = NAME_MAP.get(n) or _chat(url, model, (
                    f"영어 이름 '{n}'을 한국 더빙에서 부르는 한글 표기로만 답해라 (예: Mark → 마크). 한글 단어 하나만:"),
                    as_json=False, temperature=0.0, max_tokens=10).strip().split()[0].strip(".,\"'")
            kn = ko_names[n]
            if kn and HANGUL.search(kn) and kn not in s["ko"]:
                fixed = _first(_chat(url, model, (
                    user_ctx + f"[장면 정보]\n{notes}\n\n"
                    f"영어 대사: {s['en']}\n지금 번역: {s['ko']}\n"
                    f"번역에서 '{kn}'(원문의 {n})이 빠졌다. '{kn}'를 넣어서 뜻과 말투를 살린 자연스러운 더빙 대사로 다시 써라. "
                    f"한글 {char_budget(segs, i)}글자 이내, 한글로만, 설명 없이 대사만 출력:"),
                    as_json=False, temperature=0.3))
                if _ok_korean(fixed) and kn in fixed:
                    s["ko"] = fixed
    # 번역 모델을 GPU에서 내림 (음성 생성에 메모리 양보)
    if is_cloud(model):
        return
    try:
        body = {"model": model, "keep_alive": 0}
        urllib.request.urlopen(urllib.request.Request(ollama.rstrip("/") + "/api/generate", data=json.dumps(body).encode(),
                                                      headers={"Content-Type": "application/json"}), timeout=30)
    except OSError:
        pass


# ------------------------------------------------------------------ 6. 음성 생성
def time_stretch(ffmpeg: str, wav: np.ndarray, sr: int, factor: float, work: Path) -> np.ndarray:
    """factor>1 이면 빨라짐(짧아짐). 음높이는 유지."""
    if abs(factor - 1) < 0.02:
        return wav
    i, o = work / "ts_in.wav", work / "ts_out.wav"
    sf.write(str(i), wav, sr)
    filt = []
    f = factor
    while f > 2.0:
        filt.append("atempo=2.0")
        f /= 2.0
    filt.append(f"atempo={f:.4f}")
    run([ffmpeg, "-v", "error", "-y", "-i", str(i), "-filter:a", ",".join(filt), str(o)])
    out, _ = sf.read(str(o), dtype="float32")
    return out


def build_voice_refs(segs, speakers, work: Path) -> dict[int, str]:
    """인물마다 그 인물의 원래 대사들을 이어 붙여 목소리 참고 파일을 만든다 (최대 20초)."""
    refs = {}
    gap = np.zeros(int(0.2 * SR), np.float32)
    for sp in speakers:
        parts, total = [], 0.0
        for s in segs:
            if s.get("speaker") == sp["id"] and s.get("ref") and os.path.exists(s["ref"]):
                x, _ = sf.read(s["ref"], dtype="float32")
                parts += [x, gap]
                total += len(x) / SR
                if total > 20:
                    break
        if parts:
            p = work / f"voice_ref_{sp['id']}.wav"
            sf.write(str(p), np.concatenate(parts), SR)
            refs[sp["id"]] = str(p)
    return refs


DIGITS = "영일이삼사오육칠팔구"
UNITS = [(10 ** 8, "억"), (10 ** 4, "만"), (1000, "천"), (100, "백"), (10, "십")]


def num_to_korean(n: int) -> str:
    if n == 0:
        return "영"
    out = ""
    for unit, name in UNITS:
        q, n = divmod(n, unit)
        if q:
            out += ("" if (q == 1 and unit <= 10 ** 4) else num_to_korean(q)) + name
    if n:
        out += DIGITS[n]
    return out


def normalize_ko(text: str) -> str:
    """AI 목소리가 숫자를 못 읽으므로 한글로 바꾼다 (500년 → 오백 년)."""
    text = re.sub(r"\d+", lambda m: num_to_korean(int(m.group())) + " ", text)
    text = re.sub(r"[\"'“”‘’]", "", text).strip()
    # 실제 말할 때는 '니가'가 자연스럽다 (자막은 맞춤법대로 '네가' 유지)
    text = re.sub(r"(^|[\s,.!?~])네가(?=[\s,.!?~]|$)", r"\1니가", text)
    text = re.sub(r"(^|[\s,.!?~])네게(?=[\s,.!?~]|$)", r"\1너한테", text)   # '네게'는 '내게'로 들린다
    # 말줄임표·물결은 AI 목소리가 '아아..' 하고 끄는 소리를 내게 만든다 → 짧은 쉼표·마침표로
    text = re.sub(r"(?:\.{2,}|…+|~+)$", ".", text)
    text = re.sub(r"\s*(?:\.{2,}|…+|~+)\s*", ", ", text)
    text = re.sub(r",\s*([.!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def speech_ref(script: str) -> str:
    """발음 채점용 대본: normalize_ko 가 소리 내어 읽을 때 바꾸는 말을 똑같이 바꾼다 (숫자는 받아쓰기가 숫자로 쓰므로 그대로)."""
    return re.sub(r"(^|[^가-힣])네게(?=[^가-힣]|$)", lambda m: m.group(1) + "너한테", script)


def _cer(ref: str, hyp: str) -> float:
    ref = re.sub(r"[^가-힣]", "", ref)
    hyp = re.sub(r"[^가-힣]", "", hyp)
    if not ref:
        return 0.0
    dp = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        prev, dp[0] = dp[0], i
        for j, b in enumerate(hyp, 1):
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + (a != b))
            prev, dp[j] = dp[j], cur
    return dp[-1] / len(ref)


def trim_silence(wav: np.ndarray, sr: int, thr: float = 0.015, pad: float = 0.04) -> np.ndarray:
    """AI 음성 앞뒤의 빈 소리를 잘라낸다 (짧은 대사가 시간 안에 들어가도록)."""
    if len(wav) == 0:
        return wav
    hop = int(0.01 * sr)
    rms = np.sqrt(np.convolve(wav ** 2, np.ones(hop) / hop, mode="same"))
    idx = np.where(rms > thr * max(1e-6, np.abs(wav).max()) + 1e-4)[0]
    if len(idx) == 0:
        return wav
    a = max(0, idx[0] - int(pad * sr))
    b = min(len(wav), idx[-1] + int(pad * sr))
    return wav[a:b]


def synthesize(tts, segs, speakers, ffmpeg: str, work: Path, voice_mode: str, llm=None) -> np.ndarray:
    """대사마다: 만들기 → 받아써서 확인 → 틀리면 다시 → 길면 더 빠르게 말하도록 다시 → 다음 대사와 절대 안 겹치게."""
    import torchaudio.functional as AF
    import whisper

    model = tts.synthesizer.tts_model
    sm = model.speaker_manager
    asr = whisper.load_model("turbo", device="cuda")
    tts_sr = tts.synthesizer.output_sample_rate
    total = int((max(s["end"] for s in segs) + 5) * SR)
    track = np.zeros(total, np.float32)
    refs = build_voice_refs(segs, speakers, work) if voice_mode == "clone" else {}
    report, cut_count, kept_orig = [], 0, 0
    prev_end = 0.0
    group_lat: dict = {}

    def make(text, script, gpt, spk_emb, speed):
        script = speech_ref(script)   # 채점 기준도 실제로 읽힌 말로 ('네게' → '너한테')
        best = None
        # 짧은 대사(아빠? 등)는 AI 목소리가 잘 틀리므로 더 여러 번 시도
        temps = (0.6, 0.5, 0.7, 0.45) + ((0.65, 0.55, 0.75, 0.4) if hangul_len(script) <= 4 else ())
        for t in temps:
            out = model.inference(text, "ko", gpt, spk_emb, temperature=t, repetition_penalty=5.0, top_p=0.85,
                                  speed=speed, enable_text_splitting=False)
            wav = np.asarray(out["wav"], dtype=np.float32)
            w16 = AF.resample(torch.from_numpy(wav), tts_sr, 16000).numpy()
            res = asr.transcribe(w16, language="ko", fp16=True, word_timestamps=True)
            words = [w for seg in res["segments"] for w in seg.get("words", [])]
            err = _cer(script, res["text"])
            # 대본과 가장 잘 맞는 시작·끝 단어를 찾아 그 앞(웅얼거림)과 뒤(덧붙인 말)를
            # '실제로 말이 더 있을 때만' 잘라낸다 (첫·마지막 글자 보호)
            best_j, best_k = 0, len(words) - 1
            for j in range(min(len(words), 3)):
                acc = ""
                for k in range(j, len(words)):
                    acc += words[k]["word"]
                    e = _cer(script, acc)
                    if e < err - 0.05:
                        err, best_j, best_k = e, j, k
            # 받아쓰기가 대본과 거의 맞을 때만 자른다 (못 알아들은 부분을 잘못 잘라내지 않도록)
            trusted = err <= 0.25
            if trusted and words and best_k < len(words) - 1:
                cut = int(min(words[best_k]["end"] + 0.3, words[best_k + 1]["start"]) * tts_sr)
                wav = wav[:max(cut, int(0.3 * tts_sr))]
            elif trusted and words:   # 마지막 단어 뒤에 받아쓰기에 안 잡힌 소리가 길게 붙어 있으면 잘라낸다
                wav = wav[:max(int((words[-1]["end"] + 0.45) * tts_sr), int(0.3 * tts_sr))]
            if trusted and words and best_j > 0:
                a0 = int(max(words[best_j - 1]["end"], words[best_j]["start"] - 0.12) * tts_sr)
                wav = wav[a0:] if len(wav) - a0 > int(0.3 * tts_sr) else wav
            # 받아쓰기에 안 잡히는 '아아..' 같은 소리: 대본 길이에 비해 음성이 너무 길면 감점
            dur = len(trim_silence(wav, tts_sr)) / tts_sr
            expect = hangul_len(script) / (5.5 * speed) + 0.3
            if dur > expect * 1.6:
                err += min(0.5, (dur / expect - 1.6) * 0.5)
            if best is None or err < best[0]:
                best = (err, wav)
            if err <= 0.12:
                break
        return best

    for i, s in enumerate(segs):
        script = s.get("ko", "").strip()
        if not script:
            continue
        text = normalize_ko(script)
        if not re.search(r"[.!?~]$", text):
            text += "."
        spk = speakers[s["speaker"]]
        stock = sm.speakers[spk["voice"]]
        if voice_mode == "clone" and spk["id"] in refs:
            if spk["id"] not in group_lat:
                group_lat[spk["id"]] = model.get_conditioning_latents(audio_path=[refs[spk["id"]]])
            gpt_g, spk_emb = group_lat[spk["id"]]   # 음색은 인물 전체 기준 하나로 고정 (줄마다 바뀌지 않게)
            if s.get("ref") and os.path.exists(s["ref"]):
                gpt_l, _ = model.get_conditioning_latents(audio_path=[refs[spk["id"]], s["ref"]])
                gpt_ch = 0.5 * gpt_g + 0.5 * gpt_l      # 이 대사의 감정은 조금만 반영
            else:
                gpt_ch = gpt_g
            # 말투·발음: 캐릭터 원래 음성(감정) 반 + 한국어가 자연스러운 기본 목소리 반 → 영어 억양을 줄임
            gpt = 0.5 * gpt_ch + 0.5 * stock["gpt_cond_latent"].to(gpt_ch.device)
        else:
            gpt, spk_emb = stock["gpt_cond_latent"], stock["speaker_embedding"]

        # 이 대사가 쓸 수 있는 구간: 앞 대사가 실제로 끝난 뒤(또는 원래 시작 0.3초 전)부터 다음 대사 직전까지
        earliest = max(prev_end + 0.08, s["start"] - 0.3)
        latest_end = (segs[i + 1]["start"] - 0.05) if i + 1 < len(segs) else s["end"] + 1.0
        avail = latest_end - earliest
        slot = min(line_slot(segs, i), latest_end - s["start"])
        speed = float(np.clip(hangul_len(script) / 5.0 / max(slot, 0.3), 1.0, 1.3))
        err, wav = make(text, script, gpt, spk_emb, speed)
        wav = trim_silence(wav, tts_sr)
        dur = len(wav) / tts_sr
        if dur > avail and speed < 1.5:
            # 더 빠르게 말하도록 다시 만든다 (음을 늘리는 것보다 자연스럽다)
            speed2 = float(min(1.5, speed * dur / avail * 1.03))
            err2, wav2 = make(text, script, gpt, spk_emb, speed2)
            wav2 = trim_silence(wav2, tts_sr)
            if err2 <= err + 0.1 and len(wav2) < len(wav):
                err, wav = err2, wav2
        dur = len(wav) / tts_sr
        if dur > avail:
            wav = time_stretch(ffmpeg, wav, tts_sr, min(1.25, dur / avail), work)
        if len(wav) / tts_sr > avail and llm is not None:
            # 그래도 길면 번역을 더 짧게 받아서 다시 만든다
            budget = max(2, int(avail * 5.5))
            short = _first(_chat(llm[0], llm[1], (
                f"다음 한국어 더빙 대사를 뜻과 말투는 그대로 두고 한글 {budget}글자 이내로 아주 짧게 줄여라. "
                f"한글로만, 설명 없이 대사만 출력:\n원문: {s['en']}\n대사: {script}"), as_json=False, temperature=0.3
            ))
            if _ok_korean(short) and hangul_len(short) < hangul_len(script):
                s["ko"] = short
                e3, w3 = make(normalize_ko(short) + ("" if re.search(r"[.!?~]$", short) else "."), short,
                              gpt, spk_emb, 1.3)
                w3 = trim_silence(w3, tts_sr)
                if len(w3) / tts_sr > avail:
                    w3 = time_stretch(ffmpeg, w3, tts_sr, min(1.25, len(w3) / tts_sr / avail), work)
                err, wav = e3, w3
        if len(wav) / tts_sr > avail:
            # 정말 마지막 수단: 다음 대사와 겹치지 않도록 끝을 부드럽게 줄인다
            wav = wav[:int(avail * tts_sr)]
            cut_count += 1
        report.append(err)
        if err > 0.5:
            # 여러 번 만들어도 알아들을 수 없는 소리(웅얼거림)면 넣지 않고 원래 성우 목소리를 살린다 (자막은 한국어 유지)
            s["keep_original"] = True
            kept_orig += 1
            progress("한국어 AI 음성 생성 (확인·속도 맞추기)", i + 1, len(segs))
            continue
        fade = min(len(wav) // 4, int(0.03 * tts_sr))
        if fade > 0:
            wav[-fade:] *= np.linspace(1, 0, fade)
        dur = len(wav) / tts_sr
        # 원래 시작 시간에 두되, 넘치면 필요한 만큼만 일찍 시작
        start_t = s["start"] if s["start"] + dur <= latest_end else max(earliest, latest_end - dur)
        prev_end = start_t + dur
        (work / "lines").mkdir(exist_ok=True)
        s["dub_file"] = str(work / "lines" / f"{i:03d}.wav")
        s["dub_start"], s["dub_len"] = float(start_t), float(dur)
        sf.write(s["dub_file"], wav, tts_sr)
        wav = AF.resample(torch.from_numpy(np.ascontiguousarray(wav)), tts_sr, SR).numpy()
        a = int(start_t * SR)
        b = min(total, a + len(wav))
        track[a:b] += wav[:b - a]
        progress("한국어 AI 음성 생성 (확인·속도 맞추기)", i + 1, len(segs))
    if report:
        say(f"발음 확인: 평균 오류율 {np.mean(report) * 100:.1f}% · 시간 넘쳐서 줄인 대사 {cut_count}개 (겹침 없음)"
            + (f" · 알아듣기 힘들어 원래 목소리로 둔 대사 {kept_orig}개" if kept_orig else ""))
    del asr
    torch.cuda.empty_cache()
    return track


# ------------------------------------------------------------------ 6-2. 음색 변환 (Seed-VC, 별도 환경)
def convert_voices(segs, speakers, work: Path, vc_python: str, total_len: int) -> np.ndarray | None:
    """한국어 음성의 음색만 원래 캐릭터 목소리로 바꾸고 같은 위치에 다시 배치한다."""
    import torchaudio.functional as AF

    script = str(Path(__file__).with_name("vc_worker.py"))

    def run_vc(targets, steps, cfg, tag) -> bool:
        jobs = []
        for s in targets:
            ref = work / f"voice_ref_{s.get('speaker', 0)}.wav"
            out = s["dub_file"].replace(".wav", f"_{tag}.wav")
            jobs.append({"src": s["dub_file"], "target": str(ref), "out": out})
            s["vc_try"] = out
        jf = work / f"vc_jobs_{tag}.json"
        jf.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
        p = subprocess.Popen([vc_python, script, "--jobs", str(jf), "--steps", str(steps), "--cfg", str(cfg)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             creationflags=NO_WINDOW, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        for raw in p.stdout:
            line = raw.decode("utf-8", "replace").rstrip()
            if line.startswith("PROGRESS"):
                print(line, flush=True)
        p.wait()
        return p.returncode == 0

    # 참고할 원래 목소리가 너무 짧은 인물(5초 미만)은 변환하면 발음이 뭉개진다 → 그 인물은 통째로 변환하지 않는다
    #  (줄마다 변환/미변환이 섞이면 목소리가 갑자기 바뀌므로 인물 단위로 정한다)
    def ref_ok(spk):
        p = work / f"voice_ref_{spk}.wav"
        return p.exists() and sf.info(str(p)).duration >= 5.0

    skipped = sorted({s.get("speaker", 0) for s in segs if s.get("dub_file") and not ref_ok(s.get("speaker", 0))})
    if skipped:
        say("원래 목소리 자료가 5초 미만이라 음색 변환을 건너뛴 인물: " + ", ".join(f"#{k + 1}" for k in skipped))
    todo = [s for s in segs if s.get("dub_file") and ref_ok(s.get("speaker", 0))]
    if not todo:
        return None
    import whisper
    asr = None

    def cer_of(path, script):
        x, sr0 = sf.read(path, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        x16 = AF.resample(torch.from_numpy(x), sr0, 16000).numpy()
        return _cer(speech_ref(script), asr.transcribe(x16, language="ko", fp16=True)["text"])

    # 변환이 가끔 알아들을 수 없는 소리로 망가진다 → 망가진 줄만 설정을 바꿔 다시 변환.
    # 변환 전 음성으로 돌아가면 그 줄만 목소리가 확 달라지므로 되돌리는 건 마지막 수단
    for n, (steps, cfg) in enumerate([(40, 0.7), (25, 0.5), (30, 0.3)]):
        if not run_vc(todo, steps, cfg, f"vc{n}"):
            if n == 0:
                say("음색 변환 실패 → 변환 없이 진행")
                return None
            break
        if asr is None:
            asr = whisper.load_model("turbo", device="cuda")
        failed = []
        for s in todo:
            if not os.path.exists(s["vc_try"]):
                failed.append(s)
                continue
            c_vc = cer_of(s["vc_try"], s["ko"])
            if "c_pre" not in s:
                s["c_pre"] = cer_of(s["dub_file"], s["ko"])
            if c_vc > max(0.35, s["c_pre"] + 0.2):
                failed.append(s)
            else:
                s["vc_file"] = s["vc_try"]
        if n > 0:
            say(f"음색 변환 다시 시도 {n}: {len(todo) - len(failed)}줄 복구")
        todo = failed
        if not todo:
            break
    for s in todo:
        s["vc_file"] = None
    for s in segs:
        s.pop("vc_try", None)
        s.pop("c_pre", None)
    del asr
    torch.cuda.empty_cache()
    kept = sum(1 for s in segs if s.get("vc_file"))
    say(f"음색 변환 확인: {kept}줄 적용, 여러 번 해도 뭉개진 {len(todo)}줄은 변환 전 음성 사용")
    track = np.zeros(total_len, np.float32)
    for s in segs:
        f = s.get("vc_file")
        if not f or not os.path.exists(f):
            f = s.get("dub_file")
        if not f:
            continue
        wav, sr = sf.read(f, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        wav = AF.resample(torch.from_numpy(wav), sr, SR).numpy()
        n = int(s["dub_len"] * SR)
        wav = wav[:n]
        fade = min(len(wav) // 4, int(0.03 * SR))
        if fade > 0:
            wav[-fade:] *= np.linspace(1, 0, fade)
        a = int(s["dub_start"] * SR)
        b = min(total_len, a + len(wav))
        track[a:b] += wav[:b - a]
    return track


# ------------------------------------------------------------------ 7. 합치기
def original_voice_mask(segs: list[dict], n: int, pad: float = 0.12, ramp: float = 0.06) -> np.ndarray:
    """원래 목소리를 살릴 곳 = 1. 번역한 영어 대사 구간과 한국어 음성이 나오는 구간만 0 (앞뒤 여유·부드러운 전환).
    받아쓰기에 안 잡히거나 일부러 뺀 기합·비명·신음은 원래 성우 목소리 그대로 남는다."""
    mask = np.ones(n, np.float32)
    for s in segs:
        if not s.get("ko") or s.get("keep_original"):
            continue   # 번역 못 한 줄·AI 음성이 망가진 줄은 원래 목소리 유지
        spans = [(s["start"], s["end"])]
        if s.get("dub_len"):
            spans.append((s["dub_start"], s["dub_start"] + s["dub_len"]))
        for t0, t1 in spans:
            a, b = max(0, int((t0 - pad) * SR)), min(n, int((t1 + pad) * SR))
            mask[a:b] = 0.0
    r = int(ramp * SR)
    if r > 1:   # 이동 평균(누적합)으로 경계를 부드럽게
        c = np.concatenate([[0.0], np.cumsum(np.pad(mask, (r // 2, r - r // 2), mode="edge"), dtype=np.float64)])
        mask = ((c[r:r + n] - c[:n]) / r).astype(np.float32)
    return mask


def srt_time(t: float) -> str:
    h, rem = divmod(int(t * 1000), 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--llm", default="exaone3.5:7.8b")
    ap.add_argument("--voice", default="clone", choices=["stock", "clone"])
    ap.add_argument("--keep", default="")
    ap.add_argument("--context", default="", help="작품·인물 메모 (예: 옴니맨(아빠)과 마크(아들). 서로 반말)")
    default_vc = os.path.join(os.path.expanduser("~"), ".animeflow", "vc-venv", "Scripts", "python.exe")
    ap.add_argument("--vc", default=default_vc if os.path.exists(default_vc) else "",
                    help="Seed-VC 환경의 python 경로 (음색을 원래 캐릭터 목소리로 변환). 빈 값이면 끔")
    ap.add_argument("--reuse", default="", help="이전 작업의 script.json (받아쓰기·인물·번역을 재사용하고 음성부터 다시 만든다)")
    a = ap.parse_args()

    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="animeflow_dub_"))
    work.mkdir(parents=True, exist_ok=True)
    vocals, rest = extract_and_separate(a.ffmpeg, a.video, work)
    if a.reuse:
        d = json.loads(Path(a.reuse).read_text(encoding="utf-8"))
        segs, speakers = d["lines"], d["speakers"]
        for x in segs:
            for k in ("dub_file", "dub_start", "dub_len", "vc_file", "keep_original"):
                x.pop(k, None)
        from TTS.api import TTS
        progress("AI 음성 모델 불러오는 중", 0, 0)
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        say(f"이전 대본 재사용: 대사 {len(segs)}개, 인물 {len(speakers)}명")
    else:
        segs, speakers, tts = prepare_script(a, vocals, work)
    dub = synthesize(tts, segs, speakers, a.ffmpeg, work, a.voice,
                     llm=(a.ollama.rstrip("/") + "/api/chat", a.llm))
    finish(a, segs, speakers, tts, dub, vocals, rest, work)


def prepare_script(a, vocals, work: Path):
    """받아쓰기 → 인물 구분 → 번역."""
    segs = transcribe(vocals, work)
    say(f"대사 {len(segs)}개 받아씀")
    if not segs:
        raise SystemExit("대사를 찾지 못했습니다.")

    from TTS.api import TTS
    progress("AI 음성 모델 불러오는 중", 0, 0)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    speakers = assign_speakers(tts, segs, vocals, work)
    say("인물 " + ", ".join(f"#{s['id'] + 1}({'남' if s['male'] else '여'}, 대사 {s['lines']}개 → {s['voice']})"
                          for s in speakers))
    # 번역 전에 음성 모델을 잠시 CPU로 (번역 AI가 GPU를 씀)
    tts.to("cpu")
    torch.cuda.empty_cache()
    translate(segs, a.ollama, a.llm, a.context)
    speakers = merge_named_voices(segs, speakers)
    tts.to("cuda")
    return segs, speakers, tts


def finish(a, segs, speakers, tts, dub, vocals, rest, work: Path):
    """음색 변환 → 배경음·원래 목소리와 합치기 → 자막·음성 트랙을 영상에 넣기."""
    if a.vc:
        n0 = min(len(rest), len(dub), len(vocals))
        pre = rest[:n0] + vocals[:n0] * original_voice_mask(segs, n0)[:, None] + dub[:n0, None] * 0.95
        sf.write(str(work / "dub_ko_before_vc.wav"), pre / max(1.0, np.abs(pre).max() / 0.98), SR)
        tts.to("cpu")
        torch.cuda.empty_cache()
        converted = convert_voices(segs, speakers, work, a.vc, len(dub))
        if converted is not None:
            dub = converted
            say("캐릭터 목소리로 음색 변환 완료")
    # 배경음(대사 뺀 소리) + 한국어 대사 + 더빙하지 않는 구간의 원래 목소리(기합·비명·숨소리)
    n = min(len(rest), len(dub), len(vocals))
    keep = original_voice_mask(segs, n)
    mix = rest[:n] + vocals[:n] * keep[:, None] + dub[:n, None] * 0.95
    peak = np.abs(mix).max()
    if peak > 0.98:
        mix *= 0.98 / peak
    dub_wav = work / "dub_ko.wav"
    sf.write(str(dub_wav), mix, SR)

    srt = work / "dub_ko.srt"
    with open(srt, "w", encoding="utf-8") as f:
        for i, s in enumerate([x for x in segs if x.get("ko")], 1):
            f.write(f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n{s['ko']}\n\n")
    lines = [{k: v for k, v in s.items() if not k.startswith("_")} for s in segs]
    (work / "script.json").write_text(json.dumps({"speakers": speakers, "lines": lines}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")

    progress("영상에 음성 트랙·자막 넣기", 0, 0)
    ext = Path(a.out).suffix.lower()
    sub_codec = "mov_text" if ext in (".mp4", ".mov", ".m4v") else "srt"
    run([a.ffmpeg, "-v", "error", "-y", "-i", a.video, "-i", str(dub_wav), "-i", str(srt),
         "-map", "0:v", "-map", "1:a", "-map", "0:a?", "-map", "2:s",
         "-c:v", "copy", "-c:a:0", "aac", "-b:a:0", "256k", "-c:a:1", "copy", "-c:s", sub_codec,
         "-metadata:s:a:0", "language=kor", "-metadata:s:a:0", "title=한국어 (AI 더빙)",
         "-metadata:s:a:1", "title=원본",
         "-metadata:s:s:0", "language=kor", "-metadata:s:s:0", "title=한국어",
         "-disposition:a:0", "default", "-disposition:a:1", "0",
         "-shortest", a.out])
    say(f"완료: {a.out}")
    say(f"작업 폴더(대본·자막): {work}")


if __name__ == "__main__":
    main()
