# AnimeFlow

[English](README.md) · **한국어** · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

프레임이 적은 애니메이션을 내 PC에서 더 부드럽고 깨끗하게 만들어 주는 프로그램입니다.
애니메이션은 보통 2~3프레임마다 그림이 바뀝니다. AnimeFlow는 실제 그림을 찾아내 그 사이 프레임을
로컬 AI(RIFE)로 만들고, 원본 그림도 깨끗하게 보정합니다. **영상 길이가 그대로라서 소리·자막 싱크가 어긋나지 않습니다.**

## 기능

- **24~120fps 프레임 보간** — RIFE 4.25, 내 그래픽카드에서 실행
- **원본 길이 유지** — 출력 프레임 k를 정확히 k/FPS초에 배치, 오디오는 그대로 복사
- **프로그램 분석 (AI 불필요)** — 같은 그림 묶기, 장면 전환 감지, 움직임 측정
- **빠른 동작 보호** — 포즈가 크게 바뀌는 순간 잔상 방지 (4가지 모드, 아래 참고)
- **24fps 타이밍 복원** — 24fps 애니를 30fps로 올린 영상의 들쭉날쭉한 간격을 바로잡음
- **원본 보정** — Real-ESRGAN AnimeVideo v3로 번진 선·압축 노이즈 정리, 1080p/1440p 업스케일 선택
- **진행 화면** — 남은 시간, 현재 작업, 그려지는 프레임 미리보기
- **출력** — MP4 / MKV / MOV / WebM (NVIDIA 하드웨어 인코딩)
- 선택: 로컬 AI 장면 감독(Ollama), 입체 조명(실험), 한국어 AI 더빙(실험)

## 필요한 것

| | |
|---|---|
| 운영체제 | Windows 10 / 11 (64비트) |
| 그래픽카드 | NVIDIA GeForce (RTX 권장, VRAM 6GB 이상). RTX 3060 12GB에서 테스트 |
| 파이썬 | **3.12** — https://www.python.org/downloads/ |
| 저장 공간 | 약 8GB (PyTorch 약 5GB + 모델·FFmpeg 약 350MB) |

## 설치 (처음 한 번)

1. **Python 3.12** 설치. 설치 화면에서 **"Add python.exe to PATH"** 를 꼭 체크하세요.
2. AnimeFlow 받기: 초록색 **Code** 버튼 → **Download ZIP** → 원하는 곳에 압축 풀기
   (또는 `git clone https://github.com/O3O-2022/AnimeFlow.git`)
3. **`setup.bat`** 더블클릭. 라이브러리, PyTorch(CUDA), AI 모델, FFmpeg를 전부 자동으로 설치합니다 (10~30분).
   중간에 다운로드가 실패하면 `setup.bat` 을 다시 실행하세요. 끝난 단계는 건너뜁니다.

## 사용법

1. **`run.bat`** 더블클릭
2. **1 입력 · 출력** — **영상 열기** 를 누르거나 영상을 창에 끌어다 놓기. **저장 위치** 로 결과 파일 지정
3. **2 프레임** — 목표 FPS(24~120)와 **빠른 동작** 모드 선택
4. **3 화질 보정** — 그림을 깨끗하게 하려면 **원본 보정** 체크 (1080p/1440p 선택 가능)
5. **▶ 변환 시작** — 진행 탭에서 남은 시간과 미리보기 확인

### 빠른 동작 모드

| 모드 | 동작 | 추천 |
|---|---|---|
| **자동** (기본) | 잔상이 생길 순간만 원래 그림 유지 | 대부분의 영상 |
| **끊김 없이** | 전부 보간. 가장 부드럽지만 빠른 액션에서 잔상 | 느린 장면 |
| **잔상 없이** | 조금이라도 위험하면 원래 그림 유지 | 깔끔함 우선 (가끔 멈칫) |
| **선 안정 우선** | 눈·입·머리카락 같은 작은 움직임만 보간 | 원작 느낌 그대로. **24fps** 와 함께 |

**추천 설정:** 부드러운 액션 → *60fps + 자동*. 원작에 가깝게 → *24fps + 선 안정 우선 + 원본 보정*.

## 추가 기능 (선택)

- **AI 장면 감독** (4번 칸): [Ollama](https://ollama.com/download) 설치 후 `ollama pull qwen2.5vl:7b` 실행, 체크박스 켜기.
  AI가 장면마다 보간 방법을 정합니다.
- **입체 조명** (실험, 창 맨 아래): 깊이 추정으로 셀 그림자와 림 라이트를 넣습니다.
- **한국어 AI 더빙** (실험, 5번 칸): 별도 파이썬 환경 두 개가 필요합니다 —
  [기술 노트](docs/technical-notes.ko.md#한국어-ai-더빙용-환경-선택) 참고.
  번역은 로컬 EXAONE(Ollama) 또는 DeepSeek API(본인 키)를 씁니다.
  **개인 감상용으로만** 쓰세요: 음성 모델(XTTS v2)은 비상업 라이선스이고, 캐릭터 목소리 복제 결과는 공유·판매하면 안 됩니다.

## 문제 해결

| 증상 | 해결 |
|---|---|
| `Python 3.12 was not found` | Python 3.12 설치 후 `setup.bat` 다시 실행 |
| 모델 다운로드 실패 | `setup.bat` 다시 실행, 또는 아래 수동 설치 |
| "RIFE 모델 파일 없음" | `third_party/Practical-RIFE/train_log/flownet.pkl` 이 없음 → `setup.bat` 다시 실행 |
| 너무 느림 / CUDA 사용 불가 | NVIDIA 드라이버 업데이트, NVIDIA 그래픽카드인지 확인 |

<details>
<summary>수동 설치 (자동 다운로드가 안 될 때)</summary>

| 항목 | 넣을 위치 |
|---|---|
| [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) 코드 | `third_party/Practical-RIFE/` |
| RIFE 4.25 모델 ([Google Drive](https://drive.google.com/file/d/1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg/view)) — `flownet.pkl, IFNet_HDv3.py, RIFE_HDv3.py, refine.py` | `third_party/Practical-RIFE/train_log/` |
| [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) 코드 | `third_party/Depth-Anything-V2/` |
| [depth_anything_v2_vits.pth](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth) | `third_party/Depth-Anything-V2/checkpoints/` |
| [realesr-animevideov3.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth) | `third_party/restore/` |
| [FFmpeg 8.1](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-8.1.zip) 의 `bin/*.exe` | `tools/ffmpeg/` |

</details>

## 작동 방식

1. **분석** (AI 아님) — 같은 그림 묶기, 그림마다 시작·끝 시간 기록, 컷 감지, 움직임 측정
2. **계획** — 출력 프레임 *k* 는 정확히 *k/FPS* 초. 어떤 프레임을 복사하고 어떤 프레임을 만들지 결정
3. **생성** — RIFE가 중간 프레임 생성 (바뀐 영역만 처리해서 대사 장면은 5~11배 빠름)
4. **효과** — 명암·광원 효과(선택)를 최종 프레임에 적용
5. **인코딩** — FFmpeg, 원본 오디오·자막 그대로 복사

자세한 설계와 측정 결과: [docs/technical-notes.ko.md](docs/technical-notes.ko.md)

## 사용한 프로젝트

`setup.bat` 이 받는 외부 프로젝트 (이 저장소에는 포함되지 않음):
[Practical-RIFE](https://github.com/hzwer/Practical-RIFE) (MIT) ·
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (Apache-2.0) ·
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD-3-Clause) ·
[FFmpeg](https://ffmpeg.org/) ([BtbN](https://github.com/BtbN/FFmpeg-Builds) GPL 빌드) ·
[PySide6](https://doc.qt.io/qtforpython-6/) · [PyTorch](https://pytorch.org/) · [spandrel](https://github.com/chaiNNer-org/spandrel).
더빙(선택): [Coqui XTTS v2](https://huggingface.co/coqui/XTTS-v2) (CPML, 비상업) ·
[Whisper](https://github.com/openai/whisper) · [Demucs](https://github.com/facebookresearch/demucs) · [Seed-VC](https://github.com/Plachtaa/seed-vc).

사용할 권리가 있는 영상만 처리하세요.

## 라이선스

AnimeFlow는 [MIT 라이선스](LICENSE)로 배포됩니다.
위의 외부 프로젝트와 모델은 각자의 라이선스를 따릅니다 (예: XTTS v2는 비상업 전용).
