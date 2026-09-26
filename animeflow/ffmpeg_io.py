"""FFmpeg 기반 영상 정보 조회 / 프레임 읽기 / 인코딩."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterator

import numpy as np

from .config import ffmpeg_bin

# 콘솔 창이 뜨지 않도록 (Windows)
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class FFmpegError(RuntimeError):
    pass


@dataclass
class AudioStream:
    index: int
    codec: str
    start: float


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: Fraction
    duration: float
    nb_frames_est: int
    start: float
    codec: str
    audio: list[AudioStream] = field(default_factory=list)
    has_subtitles: bool = False

    @property
    def fps_float(self) -> float:
        return float(self.fps)


def _run(args: list[str]) -> str:
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", creationflags=_NO_WINDOW)
    if p.returncode != 0:
        raise FFmpegError(p.stderr.strip()[-2000:])
    return p.stdout


def _parse_rate(s: str | None) -> Fraction | None:
    if not s or s in ("0/0", "N/A"):
        return None
    try:
        r = Fraction(s)
        return r if r > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def probe(path: str) -> VideoInfo:
    out = _run([ffmpeg_bin("ffprobe"), "-v", "error", "-print_format", "json",
                "-show_streams", "-show_format", path])
    data = json.loads(out)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")), None)
    if v is None:
        raise FFmpegError("영상 스트림을 찾을 수 없습니다.")

    fps = _parse_rate(v.get("avg_frame_rate")) or _parse_rate(v.get("r_frame_rate")) or Fraction(24000, 1001)
    duration = float(v.get("duration") or data.get("format", {}).get("duration") or 0)
    nb = int(v.get("nb_frames") or 0) or int(round(duration * float(fps)))
    audio = [AudioStream(int(s["index"]), s.get("codec_name", ""), float(s.get("start_time") or 0))
             for s in streams if s.get("codec_type") == "audio"]
    return VideoInfo(
        path=path,
        width=int(v["width"]),
        height=int(v["height"]),
        fps=fps,
        duration=duration,
        nb_frames_est=nb,
        start=float(v.get("start_time") or 0),
        codec=v.get("codec_name", ""),
        audio=audio,
        has_subtitles=any(s.get("codec_type") == "subtitle" for s in streams),
    )


def frame_timestamps(path: str) -> np.ndarray:
    """패킷 PTS(표시 시간)를 디코딩 없이 빠르게 읽어 정렬해서 반환."""
    out = _run([ffmpeg_bin("ffprobe"), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "packet=pts_time", "-of", "csv=p=0", path])
    vals = []
    for line in out.splitlines():
        line = line.strip().rstrip(",")
        if line and line != "N/A":
            try:
                vals.append(float(line))
            except ValueError:
                pass
    arr = np.array(sorted(vals), dtype=np.float64)
    return arr


class _StderrTail:
    """ffmpeg stderr를 계속 비워서 파이프가 막히지 않게 하고, 마지막 줄만 보관."""

    def __init__(self, stream):
        self.lines: deque[str] = deque(maxlen=40)
        self._t = threading.Thread(target=self._pump, args=(stream,), daemon=True)
        self._t.start()

    def _pump(self, stream):
        for raw in iter(stream.readline, b""):
            self.lines.append(raw.decode("utf-8", "replace").rstrip())

    def text(self) -> str:
        self._t.join(timeout=1)
        return "\n".join(self.lines)


class FrameReader:
    """원본 프레임을 하나도 빠짐없이(passthrough) 순서대로 읽는다."""

    def __init__(self, path: str, width: int, height: int, pix_fmt: str = "bgr24",
                 scale_to: tuple[int, int] | None = None):
        self.w, self.h = scale_to or (width, height)
        self.channels = 1 if pix_fmt == "gray" else 3
        args = [ffmpeg_bin("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin",
                "-threads", "0", "-i", path, "-map", "0:v:0", "-an", "-sn", "-dn",
                "-fps_mode", "passthrough"]
        if scale_to:
            args += ["-vf", f"scale={self.w}:{self.h}:flags=area"]
        args += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"]
        self.frame_bytes = self.w * self.h * self.channels
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=self.frame_bytes * 4, creationflags=_NO_WINDOW)
        self._err = _StderrTail(self.proc.stderr)

    def __iter__(self) -> Iterator[np.ndarray]:
        shape = (self.h, self.w) if self.channels == 1 else (self.h, self.w, 3)
        stdout = self.proc.stdout
        while True:
            buf = bytearray(self.frame_bytes)  # 쓰기 가능한 버퍼 (torch.from_numpy 경고 방지)
            if stdout.readinto(buf) < self.frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(shape)
        self.proc.wait()
        if self.proc.returncode not in (0, None) and self.proc.returncode != -15:
            err = self._err.text()
            if err:
                raise FFmpegError(err)

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


@dataclass
class EncodeOptions:
    container: str
    video_codec: str
    quality: int
    filter_chain: str  # "" 이면 효과 없음
    video_offset: float = 0.0
    width: int = 0     # 0 = 원본 크기
    height: int = 0


def _video_codec_args(codec: str, q: int) -> list[str]:
    if codec in ("libx264", "libx265"):
        a = ["-c:v", codec, "-preset", "medium", "-crf", str(q)]
        if codec == "libx265":
            a += ["-x265-params", "log-level=error"]
        return a
    if codec in ("h264_nvenc", "hevc_nvenc", "av1_nvenc"):
        return ["-c:v", codec, "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                "-cq", str(q), "-b:v", "0", "-spatial-aq", "1"]
    if codec == "libsvtav1":
        return ["-c:v", codec, "-preset", "6", "-crf", str(min(63, q + 10))]
    if codec == "libvpx-vp9":
        return ["-c:v", codec, "-crf", str(min(63, q + 12)), "-b:v", "0",
                "-row-mt", "1", "-deadline", "good", "-cpu-used", "2"]
    if codec == "prores_ks":
        return ["-c:v", codec, "-profile:v", "3", "-vendor", "apl0"]
    return ["-c:v", codec]


def _audio_args(container: str, audio: list[AudioStream]) -> list[str]:
    if not audio:
        return []
    codecs = {a.codec for a in audio}
    if container == "mkv":
        return ["-c:a", "copy"]
    if container in ("mp4", "mov"):
        if codecs <= {"aac", "mp3", "alac", "ac3", "eac3"}:
            return ["-c:a", "copy"]
        return ["-c:a", "aac", "-b:a", "256k"]
    if container == "webm":
        if codecs <= {"opus", "vorbis"}:
            return ["-c:a", "copy"]
        return ["-c:a", "libopus", "-b:a", "192k"]
    return ["-c:a", "copy"]


class VideoWriter:
    """보간된 프레임을 stdin으로 받아 원본 오디오와 함께 인코딩한다.

    출력 영상 길이 = 원본 길이 이므로 오디오를 그대로 복사해도 싱크가 맞는다.
    """

    def __init__(self, info: VideoInfo, out_path: str, fps: Fraction, opts: EncodeOptions):
        ow, oh = (opts.width or info.width), (opts.height or info.height)
        self.frame_bytes = ow * oh * 3
        pix = "yuv422p10le" if opts.video_codec == "prores_ks" else "yuv420p"
        chain = f"{opts.filter_chain},format={pix}" if opts.filter_chain else f"format={pix}"

        args = [ffmpeg_bin("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{ow}x{oh}", "-framerate", str(fps)]
        if opts.video_offset > 1e-4:
            args += ["-itsoffset", f"{opts.video_offset:.6f}"]
        args += ["-i", "pipe:0", "-i", info.path,
                 "-filter_complex", f"[0:v]{chain}[vout]", "-map", "[vout]"]
        if info.audio:
            args += ["-map", "1:a?"]
        if opts.container == "mkv" and info.has_subtitles:
            args += ["-map", "1:s?", "-c:s", "copy"]
        args += ["-map_metadata", "1", "-map_chapters", "1"]
        args += _video_codec_args(opts.video_codec, opts.quality)
        if opts.video_codec in ("hevc_nvenc", "libx265") and opts.container in ("mp4", "mov"):
            args += ["-tag:v", "hvc1"]
        args += _audio_args(opts.container, info.audio)
        if opts.container in ("mp4", "mov"):
            args += ["-movflags", "+faststart"]
        args += [out_path]
        self.args = args
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=self.frame_bytes * 2, creationflags=_NO_WINDOW)
        self._err = _StderrTail(self.proc.stderr)

    def write(self, frame: np.ndarray):
        try:
            self.proc.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
        except (BrokenPipeError, OSError):
            self.proc.wait()
            raise FFmpegError("인코더가 비정상 종료되었습니다:\n" + self._err.text())

    def finish(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.wait()
        if self.proc.returncode != 0:
            raise FFmpegError("인코딩 실패:\n" + self._err.text())

    def abort(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
