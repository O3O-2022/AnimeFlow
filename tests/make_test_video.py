"""테스트용 '애니메이션 같은' 영상 생성.

- 23.976fps, 2프레임마다 새 그림(on twos) 으로 공이 움직임
- 중간에 1초 정지(홀드), 장면 전환(컷), 다시 3프레임 작화 구간
- 오디오: 공이 벽에 닿는 순간마다 '삑' 소리 → 싱크 검사용
"""
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animeflow.config import ffmpeg_bin  # noqa: E402

W, H = 1280, 720
FPS = Fraction(24000, 1001)


def draw(scene: int, x: float) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    if scene == 0:
        img[:] = (70, 50, 40)
        cv2.rectangle(img, (0, 520), (W, H), (60, 110, 80), -1)
        cv2.circle(img, (int(x), 400), 60, (60, 170, 250), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), 400), 60, (20, 20, 20), 4, cv2.LINE_AA)
    else:
        img[:] = (200, 180, 150)
        cv2.rectangle(img, (int(x) - 70, 250), (int(x) + 70, 470), (80, 60, 200), -1, cv2.LINE_AA)
        cv2.rectangle(img, (int(x) - 70, 250), (int(x) + 70, 470), (20, 20, 20), 4, cv2.LINE_AA)
    return img


def main(out: str):
    frames = []
    # 장면 0: on twos 로 좌→우 (48 프레임 = 24 그림)
    for i in range(48):
        k = i // 2
        frames.append(draw(0, 100 + k * 45))
    # 홀드 24 프레임 (1초)
    frames += [frames[-1]] * 24
    # 되돌아오기 on twos
    for i in range(48):
        k = i // 2
        frames.append(draw(0, 1135 - k * 45))
    # 컷 → 장면 1: on threes
    for i in range(72):
        k = i // 3
        frames.append(draw(1, 150 + k * 40))

    n = len(frames)
    dur = n / float(FPS)
    tmp_v = Path(out).with_suffix(".video.mp4")
    p = subprocess.Popen([ffmpeg_bin(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{W}x{H}", "-framerate", str(FPS), "-i", "pipe:0",
                          "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(tmp_v)],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    # 벽에 닿는 순간(프레임 47, 119)과 컷(144)에 삑 소리
    beeps = [47 / float(FPS), 119 / float(FPS), 144 / float(FPS)]
    sr = 48000
    t = np.arange(int(dur * sr)) / sr
    gate = np.zeros_like(t)
    for b in beeps:
        gate[(t >= b) & (t < b + 0.08)] = 1.0
    pcm = (gate * 0.8 * np.sin(2 * np.pi * 1000 * t) * 32767).astype("<i2")
    tmp_a = Path(out).with_suffix(".audio.wav")
    import wave
    with wave.open(str(tmp_a), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    subprocess.run([ffmpeg_bin(), "-y", "-nostdin", "-loglevel", "error", "-i", str(tmp_v), "-i", str(tmp_a),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out],
                   check=True, timeout=60)
    tmp_a.unlink()
    tmp_v.unlink()
    print(f"{out}: {n} frames, {dur:.3f}s, beeps at {[round(b, 3) for b in beeps]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_anime.mp4")
