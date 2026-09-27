# AnimeFlow

**English** · [한국어](README.ko.md) · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

Make low-frame-rate anime look smoother and cleaner on your own PC.
Anime is usually drawn "on twos or threes" (a new drawing every 2–3 frames). AnimeFlow detects the
real drawings, generates in-between frames with a local AI (RIFE), and can clean up the original
drawings — **without changing the video length**, so audio and subtitles stay perfectly in sync.

> The app's interface is currently in Korean. This guide explains every button you need.

## Features

- **Frame interpolation to 24–120 fps** with RIFE 4.25 (runs locally on your GPU)
- **Keeps the original length** – each output frame is placed at exactly k / FPS seconds, audio is copied as-is
- **Smart analysis (no AI needed)** – groups duplicate frames, detects scene cuts, measures motion
- **Fast-action protection** – avoids ghosting on big pose changes (4 modes, see below)
- **24p restore** – fixes uneven 2-3 cadence when 24 fps anime was uploaded as 30 fps
- **Source restoration** – Real-ESRGAN AnimeVideo v3 cleans blurry lines and compression noise, optional 1080p / 1440p upscale
- **Progress view** – ETA, current task, and a live preview of the frame being drawn
- **Output** – MP4 / MKV / MOV / WebM (NVIDIA hardware encoding)
- Optional: local AI scene director (Ollama), experimental relighting, experimental Korean AI dubbing

## Requirements

| | |
|---|---|
| OS | Windows 10 / 11 (64-bit) |
| GPU | NVIDIA GeForce (RTX recommended, 6 GB+ VRAM). Tested on RTX 3060 12 GB |
| Python | **3.12** – https://www.python.org/downloads/ |
| Disk | about 8 GB (PyTorch ~5 GB + models/FFmpeg ~350 MB) |

## Install (one time)

1. Install **Python 3.12**. In the installer, tick **"Add python.exe to PATH"**.
2. Download AnimeFlow: green **Code** button → **Download ZIP**, then unzip it anywhere
   (or `git clone https://github.com/O3O-2022/AnimeFlow.git`).
3. Double-click **`setup.bat`**. It installs everything automatically (10–30 minutes):
   Python libraries, PyTorch (CUDA), the AI models and FFmpeg.
   If a download fails, just run `setup.bat` again – finished steps are skipped.

## Use

1. Double-click **`run.bat`**.
2. **1 입력 · 출력 (Input / Output)** – click **영상 열기** (Open video) or drag a video onto the window.
   **저장 위치** (Save to) chooses the output file.
3. **2 프레임 (Frames)** – pick the target FPS (24–120) and the **빠른 동작** (fast motion) mode.
4. **3 화질 보정 (Quality)** – tick **원본 보정** (Restore source) to clean the drawings; optionally choose 1080p / 1440p.
5. Click **▶ 변환 시작** (Start). The progress tab shows the ETA and a live preview.

### Fast-motion modes

| Mode (Korean label) | What it does | Good for |
|---|---|---|
| **자동** – Auto (default) | Keeps the original drawing only where interpolation would ghost | Most videos |
| **끊김 없이** – Always interpolate | Interpolates everything; smoothest, some ghosting in fast action | Slow scenes |
| **잔상 없이** – No ghosting | Keeps the original drawing whenever motion is risky | Clean look, a few holds |
| **선 안정 우선** – Line stability | Interpolates only small motion (eyes, mouth, hair) | Closest to the original art; use with **24 fps** |

**Recommended:** smooth action → *60 fps + Auto*. Faithful to the original → *24 fps + Line stability + Restore source*.

## Optional extras

- **AI scene director** (card 4): install [Ollama](https://ollama.com/download), run `ollama pull qwen2.5vl:7b`,
  then tick the checkbox. The AI looks at each scene and decides how to interpolate it.
- **Relighting** (experimental, bottom of the window): adds dramatic cel shading and rim light using depth estimation.
- **Korean AI dubbing** (experimental, card 5): needs two extra Python environments – see
  [technical notes (Korean)](docs/technical-notes.ko.md#한국어-ai-더빙용-환경-선택). Personal use only: the voice model
  (XTTS v2) is under a non-commercial license, and cloning a character's voice must not be shared or sold.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Python 3.12 was not found` | Install Python 3.12 and re-run `setup.bat` |
| A model download failed | Run `setup.bat` again, or download it manually (below) |
| "RIFE 모델 파일 없음" (RIFE model missing) | `third_party/Practical-RIFE/train_log/flownet.pkl` is missing – re-run `setup.bat` |
| Very slow / CUDA not available | Update the NVIDIA driver; make sure you have an NVIDIA GPU |

<details>
<summary>Manual install (if automatic download fails)</summary>

| What | Where to put it |
|---|---|
| [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) code | `third_party/Practical-RIFE/` |
| RIFE 4.25 model ([Google Drive](https://drive.google.com/file/d/1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg/view)) – `flownet.pkl, IFNet_HDv3.py, RIFE_HDv3.py, refine.py` | `third_party/Practical-RIFE/train_log/` |
| [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) code | `third_party/Depth-Anything-V2/` |
| [depth_anything_v2_vits.pth](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth) | `third_party/Depth-Anything-V2/checkpoints/` |
| [realesr-animevideov3.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth) | `third_party/restore/` |
| [FFmpeg 8.1](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-8.1.zip) `bin/*.exe` | `tools/ffmpeg/` |

</details>

## How it works

1. **Analyze** (no AI) – group identical frames, record when each drawing starts/ends, detect cuts, measure motion
2. **Plan** – output frame *k* is at exactly *k / FPS* seconds; decide which frames to copy and which to generate
3. **Generate** – RIFE creates the in-between frames (only the changed region is processed, 5–11× faster on dialogue scenes)
4. **Effects** – optional shading / lighting, applied to the final frames
5. **Encode** – FFmpeg, original audio and subtitles copied

More details (Korean): [docs/technical-notes.ko.md](docs/technical-notes.ko.md)

## Credits

AnimeFlow uses these projects (downloaded by `setup.bat`, not included in this repository):
[Practical-RIFE](https://github.com/hzwer/Practical-RIFE) (MIT) ·
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (Apache-2.0) ·
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD-3-Clause) ·
[FFmpeg](https://ffmpeg.org/) (GPL build by [BtbN](https://github.com/BtbN/FFmpeg-Builds)) ·
[PySide6](https://doc.qt.io/qtforpython-6/) · [PyTorch](https://pytorch.org/) · [spandrel](https://github.com/chaiNNer-org/spandrel).
Optional dubbing: [Coqui XTTS v2](https://huggingface.co/coqui/XTTS-v2) (CPML, non-commercial) ·
[Whisper](https://github.com/openai/whisper) · [Demucs](https://github.com/facebookresearch/demucs) · [Seed-VC](https://github.com/Plachtaa/seed-vc).

Please only process videos you have the right to use.
