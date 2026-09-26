"""GUI 없이 전체 파이프라인 실행 (테스트/배치용).

사용: python tests/run_headless.py 입력.mp4 출력.mp4 [fps] [backend] [fx]
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animeflow.analyzer import analyze  # noqa: E402
from animeflow.config import Settings  # noqa: E402
from animeflow.ffmpeg_io import probe  # noqa: E402
from animeflow.pipeline import describe_plan, render  # noqa: E402
from animeflow.planner import build_plan  # noqa: E402


def main():
    src, dst = sys.argv[1], sys.argv[2]
    s = Settings(input_path=src, output_path=dst)
    s.target_fps = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    s.backend = sys.argv[4] if len(sys.argv) > 4 else "rife"
    if len(sys.argv) > 5 and sys.argv[5] == "fx":
        s.shading = s.lighting = s.vignette = True
    if len(sys.argv) > 5 and sys.argv[5] == "relight":
        s.relight = True
    if len(sys.argv) > 5 and sys.argv[5].startswith("restore"):
        s.restore = True
        s.restore_res = sys.argv[5][len("restore"):] or "source"
    s.video_codec = "h264_nvenc"
    cancel = threading.Event()

    info = probe(src)
    t = time.perf_counter()
    tl = analyze(info, s, cancel=cancel)
    print(f"analyze {time.perf_counter() - t:.2f}s: frames={tl.n_frames} keys={tl.n_keys} "
          f"scenes={len(tl.scenes)} dur={tl.duration:.4f} cadence={tl.cadence_histogram()}")
    for sc in tl.scenes:
        print(f"  scene {sc.id}: {sc.start:.3f}-{sc.end:.3f} keys={sc.n_keys} motion={sc.motion:.4f}")
    plan = build_plan(tl, s)
    print(describe_plan(tl, plan))
    last = [-1]

    def prog(d, total):
        if d * 10 // total != last[0]:
            last[0] = d * 10 // total
            print(f"  render {d}/{total}")
    stats = render(tl, plan, s, prog, print, cancel)
    print(stats)


if __name__ == "__main__":
    main()
