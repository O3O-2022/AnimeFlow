"""목소리 변환 작업기 (별도 가상환경 vc-venv 에서 실행, Seed-VC).

더빙 작업기가 만든 한국어 대사 음성의 '음색'만 원래 캐릭터 목소리로 바꾼다.
발음·연기·길이는 그대로 둔다.
사용: python vc_worker.py --jobs jobs.json
  jobs.json = [{"src": 한국어음성.wav, "target": 캐릭터원래목소리.wav, "out": 결과.wav}, ...]
"""
import argparse
import json
import os
import sys

SEED_VC = os.path.join(os.path.expanduser("~"), ".animeflow", "seed-vc")
sys.path.insert(0, SEED_VC)
os.chdir(SEED_VC)  # Seed-VC가 상대 경로로 설정 파일을 읽는다
# 일부 라이브러리가 PATH에서 ffmpeg를 찾으므로 프로그램 폴더의 ffmpeg를 알려준다
_FF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "ffmpeg")
os.environ["PATH"] = os.path.abspath(_FF_DIR) + os.pathsep + os.environ.get("PATH", "")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=0.7)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs, encoding="utf-8"))

    from seed_vc_wrapper import SeedVCWrapper
    print("PROGRESS 목소리 변환 모델 불러오는 중\t0\t0", flush=True)
    vc = SeedVCWrapper()
    for i, j in enumerate(jobs):
        # 조각 전송(mp3 변환) 없이 전체 결과를 한 번에 받는다. convert_voice 는 생성기라
        # stream_output=False 일 때 결과가 StopIteration 값으로 돌아온다.
        gen = vc.convert_voice(source=j["src"], target=j["target"], diffusion_steps=a.steps,
                               length_adjust=1.0, inference_cfg_rate=a.cfg, f0_condition=False,
                               auto_f0_adjust=False, pitch_shift=0, stream_output=False)
        full = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            full = e.value
        if isinstance(full, tuple):
            sr, wav = full
        else:
            sr, wav = 22050, full  # f0 조건 없을 때 Seed-VC 출력은 22.05kHz
        wav = np.asarray(wav)
        if wav.dtype.kind == "i":
            wav = wav.astype(np.float32) / 32768.0
        wav = wav.astype(np.float32).reshape(-1)
        sf.write(j["out"], wav, sr)
        print(f"PROGRESS 캐릭터 목소리로 음색 변환 (Seed-VC)\t{i + 1}\t{len(jobs)}", flush=True)
    print("vc done", flush=True)


if __name__ == "__main__":
    main()
