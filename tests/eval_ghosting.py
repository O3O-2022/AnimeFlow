"""실제 영상으로 보간 품질(잔상) 측정.

같은 간격으로 연속된 그림 A, M, B 를 찾아 A와 B로 가운데를 만들고 진짜 M과 비교한다.
사용: python tests/eval_ghosting.py cache/xxx_timeline.json [샘플수]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animeflow.ffmpeg_io import FrameReader, probe  # noqa: E402
from animeflow.interpolator import RifeInterpolator  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "cache" / "eval"


def load_triplets(tl_path, n):
    d = json.load(open(tl_path, encoding="utf-8"))
    k = d["keys"]
    trip = []
    for i in range(len(k) - 2):
        a, m, b = k[i], k[i + 1], k[i + 2]
        if not (a["scene"] == m["scene"] == b["scene"]):
            continue
        s1, s2 = m["src_frame"] - a["src_frame"], b["src_frame"] - m["src_frame"]
        if s1 == s2 and s1 <= 2:
            trip.append((a["src_frame"], m["src_frame"], b["src_frame"], max(m["track_fail"], b["track_fail"]),
                         max(m["motion"], b["motion"])))
    rng = np.random.default_rng(0)
    trip = sorted(trip, key=lambda x: x[3])
    # 추적 실패율 전 구간에서 고르게 뽑기
    idx = np.unique(np.linspace(0, len(trip) - 1, n).astype(int))
    return d["source"], [trip[i] for i in idx]


def main():
    tl_path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    src, trips = load_triplets(tl_path, n)
    need = sorted({f for t in trips for f in t[:3]})
    info = probe(src)
    frames = {}
    rd = FrameReader(src, info.width, info.height, "bgr24")
    needset = set(need)
    for i, fr in enumerate(rd):
        if i in needset:
            frames[i] = fr
        if i >= need[-1]:
            break
    rd.close()
    print(f"triplets {len(trips)}  video {info.width}x{info.height}")
    it = RifeInterpolator(1.0, True, log=lambda *a: None)
    OUT.mkdir(parents=True, exist_ok=True)
    import pickle
    pickle.dump((trips, {k: frames[k] for k in need}), open(OUT / "triplets.pkl", "wb"))
    print("saved", OUT / "triplets.pkl")


if __name__ == "__main__":
    main()
