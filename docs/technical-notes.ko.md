# AnimeFlow — 애니메이션 프레임 보간

2~3프레임마다 그림이 바뀌는 애니메이션(인빈시블 같은 액션물 포함)을 60/120fps로 부드럽게 만드는 도구.
**원본 길이를 그대로 유지**하므로 오디오 싱크가 어긋나지 않는다.

## 실행

```
run.bat
```

처음이라면 `setup.bat` → `run.bat`. 사용자용 설명은 [README](../README.ko.md) 참고.

## 작업 흐름

| 단계 | 담당 | 내용 |
|---|---|---|
| ① 분석 | 프로그램 (AI 아님) | 같은 그림을 하나로 묶고 시작~끝 시간 기록, 장면 전환(컷) 감지, 그림 사이 움직임 측정 → `cache/<영상>_timeline.json` |
| ② 장면 감독 | 로컬 AI (Ollama 비전 모델, 선택) | 장면의 첫/끝 그림 + 시간 정보를 보고 장면마다 보간 여부·보간 길이 결정 |
| ③ 프레임 구조 | 프로그램 | 출력 프레임 k = 정확히 k/FPS 초 시점. 복사할지, 어느 두 그림 사이 몇 % 지점을 만들지 결정 |
| ④ 프레임 생성 | RIFE 4.25 (로컬 AI, GPU) | 중간 프레임 생성 |
| ⑤ 효과 | GPU | 명암 보정 / 빛 번짐 / 비네팅 / 입체 조명(실험 기능) — 프레임 작업이 모두 끝난 최종 프레임에만 적용 |
| ⑥ 출력 | FFmpeg | MP4 / MKV / MOV / WebM, 원본 오디오·자막 그대로 복사 |

### 원본 보정 (화질 보정 칸의 '원본 보정')
애니메이션 전용 복원 AI(Real-ESRGAN AnimeVideo v3, `third_party/restore/`)로 번진 선과 압축 노이즈를 정리하고,
선택하면 1080p/1440p로 올린다. 중복 제거된 고유 그림에만 적용한 뒤 그 그림으로 중간 프레임을 만든다.
- 속도: 고유 그림 한 장당 약 0.1초 (RTX 3060, 720p 입력)

### 30fps로 늘어난 24fps 애니 자동 복원
유튜브 등에서 24fps 작화를 30fps로 바꾸면 그림 간격이 2,3,2,3프레임으로 들쭉날쭉해져 보간 결과가 빨라졌다 느려졌다 한다.
분석 단계에서 이를 감지해 각 그림을 원래 24fps 타이밍으로 되돌린다 (길이·소리 싱크 그대로).

### 싱크가 안 어긋나는 이유
프레임을 "끼워 넣어서" 영상이 길어지는 방식이 아니라, 원본과 **같은 길이**의 시간축 위에서
1/FPS 간격마다 그 시점의 그림을 만든다. 출력 길이 오차는 항상 1프레임 미만이고 오디오는 원본을 그대로 복사한다.

### 액션물(인빈시블 등)을 위한 처리
- **빠른 동작 처리 (자동 / 끊김 없이 / 잔상 없이)**: 보간 AI는 픽셀을 '옮기는' 방식이라 그림 한 장 사이에
  포즈가 크게 바뀌면 새 포즈를 그리지 못하고 잔상을 만든다. 인빈시블 실제 영상 80곳에서
  진짜 중간 그림과 비교 측정한 기준으로, 자동 모드는 잔상이 날 순간만 원본 그림을 유지한다.
- **선 안정 우선 모드**: 눈·입·머리카락 같은 작은 움직임만 보간하고 나머지는 원래 그림을 둔다.
  보간 AI가 선마다 따로 움직임을 추정해서 한 선이 다른 선을 애매하게 따라가는(잔상처럼 보이는) 현상을 피한다.
  24fps 출력과 함께 쓰면 원작에 가장 가깝다.
- **순간 전환**: 화면 대부분이 한 번에 바뀌면(예: 군중 → 거리) 장면 전환처럼 처리해 두 장면이 뭉개지지 않게 한다.
- **임팩트 프레임 / 플래시**: 색 분포가 급변하면 장면 전환으로 처리해 섞이지 않게 한다.

### 입체 조명 — 실험 기능 (그림체는 그대로, 조명만 극적으로)
GUI 맨 아래 '실험 기능' 칸에 있다. 광원 방향 자동 추정이 틀리는 장면이 있어 실험 기능으로 분류.

1. Depth Anything V2 (Small)로 장면의 앞뒤 깊이 추정 (원본 그림마다 한 번)
2. 광원 방향·색: 배경에서 가장 밝은 곳에서 자동 추정하거나 직접 지정 (왼쪽/오른쪽/위/역광, 노을·푸른빛·붉은빛·흰빛)
3. 장면 전체를 어둡게(배경은 더), 광원 반대편에 셀 그림자, 광원 쪽 윤곽에 광원 색 림 라이트
4. 조명·창문·표시등처럼 색 있는 빛과 아주 작은 흰 점만 발광 (흰 옷은 빛나지 않음)
- 비용: 720p 기준 프레임당 약 10~15ms 추가
- **긴 정지(홀드)**: 대사 장면 등 오래 멈춘 그림은 끝부분만 움직이게 해서 "녹아내리는" 모핑을 막는다.

### 한국어 AI 더빙 — 실험 기능 (개인 감상용)
대사 분리(Demucs) → 받아쓰기(Whisper) → 번역 → AI 음성(XTTS v2) → 캐릭터 음색 변환(Seed-VC) → 원래 배경음과 합치기.
- **번역 AI**: 로컬 EXAONE 3.5 (32B/7.8B), 또는 DeepSeek-V3/R1 클라우드 (본인 API 키 필요,
  영어 대사 글자와 작품·인물 메모만 전송). DeepSeek-V3/R1 원본(약 670B)이나 Llama 70B/405B는
  12GB 그래픽카드로는 로컬 실행이 불가능하거나 대사 한 줄에 몇 분씩 걸려 클라우드로만 지원한다.
- **목소리 일관성**: 목소리 특징으로 나눈 인물을 번역 AI가 대사 내용으로 알아낸 이름(예: 옴니맨, 마크)으로 다시 묶고,
  인물마다 음색을 하나로 고정한다 (소리칠 때 다른 사람 목소리로 바뀌는 문제 방지).
- **기합·비명·숨소리**: "하!", "으아악" 같은 말 아닌 소리는 번역하지 않고 원래 성우 목소리를 그대로 살린다.
- **웅얼거림 방지**: 만든 음성을 다시 받아써 대본과 비교, 앞뒤에 붙은 "아아.." 같은 소리를 잘라내고 다시 만든다.

## 폴더 구조

```
AnimeFlow/
├─ run.bat / setup.bat
├─ animeflow/            소스 코드
│  ├─ analyzer.py        ① 중복 그림 묶기, 컷 감지, 움직임 측정
│  ├─ director.py        ② 로컬 AI 장면 감독 (Ollama)
│  ├─ planner.py         ③ 출력 프레임 구조
│  ├─ interpolator.py    ④ RIFE / OpenCV / 블렌드
│  ├─ effects.py         ⑤ 명암·광원 효과 (GPU, CPU 대체)
│  ├─ gpu_output.py      GPU 비동기 출력
│  ├─ ffmpeg_io.py       ⑥ 디코딩·인코딩
│  └─ gui/               화면
├─ tools/ffmpeg/         FFmpeg 8.1 (BtbN 빌드)
├─ third_party/Practical-RIFE/train_log/   RIFE 4.25 모델
├─ third_party/Depth-Anything-V2/checkpoints/   깊이 추정 모델 (입체 조명용)
├─ cache/                분석 결과 JSON
└─ tests/                테스트 영상 생성, GUI 없이 실행
```

가상환경(PyTorch 등 약 5GB)은 OneDrive 동기화를 피하려고 `%USERPROFILE%\.animeflow\venv` 에 있다.

## 사용한 프레임워크 / 링크

- Python 3.12 + PySide6 (GUI) — https://doc.qt.io/qtforpython-6/
- PyTorch (CUDA 12.8) — https://pytorch.org/
- Practical-RIFE (프레임 보간 AI) — https://github.com/hzwer/Practical-RIFE
  - 모델 4.25: README의 Google Drive 링크 → 압축 풀어서 `train_log/` 안의 `flownet.pkl, IFNet_HDv3.py, RIFE_HDv3.py, refine.py` 를 `third_party/Practical-RIFE/train_log/` 에
- Depth Anything V2 Small (입체 조명, Apache-2.0) — https://github.com/DepthAnything/Depth-Anything-V2
  - `depth_anything_v2_vits.pth` (https://huggingface.co/depth-anything/Depth-Anything-V2-Small) → `third_party/Depth-Anything-V2/checkpoints/`
  - 코드도 필요하다: 저장소를 `third_party/Depth-Anything-V2/` 에 clone
- Real-ESRGAN AnimeVideo v3 (원본 보정, BSD-3) — https://github.com/xinntao/Real-ESRGAN
  - `realesr-animevideov3.pth` (https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth) → `third_party/restore/`
- FFmpeg (BtbN 빌드, 안정 버전 8.1) — https://github.com/BtbN/FFmpeg-Builds/releases
  - `ffmpeg-n8.1-latest-win64-gpl-8.1.zip` 의 `bin/*.exe` → `tools/ffmpeg/`
  - 주의: master(nightly) 빌드는 AAC 인코더가 멈추는 버그가 있었음
- Ollama (로컬 AI, 선택) — https://ollama.com/download
  - 설치 후 `ollama pull qwen2.5vl:7b` (약 6GB, RTX 3060 12GB에서 동작, 장면당 약 1.5초)
  - 판단이 끝나면 모델을 GPU에서 내려서 프레임 생성에 메모리를 양보한다

### 한국어 AI 더빙용 환경 (선택)
라이브러리 충돌을 피하려고 앱과 별도의 가상환경 두 개를 쓴다 (`animeflow/dubbing.py` 가 자동으로 찾음).
- `%USERPROFILE%\.animeflow\dub-venv` (Python 3.12): `torch==2.7.1` (cu128), `coqui-tts==0.27.5`, `"transformers<5"`,
  `openai-whisper`, `demucs`, `resemblyzer`, `webrtcvad-wheels`, `soundfile`
  - XTTS v2 는 CPML(비상업) 라이선스 — 개인 감상용으로만
- `%USERPROFILE%\.animeflow\vc-venv` + `%USERPROFILE%\.animeflow\seed-vc` (음색 변환, 선택):
  https://github.com/Plachtaa/seed-vc 를 clone 하고 그 requirements 설치 (`transformers==4.46.3`)
- 번역: Ollama `exaone3.5:32b` 또는 `exaone3.5:7.8b`, 또는 DeepSeek API 키 (GUI에서 입력)

## 속도 (RTX 3060 12GB 측정)

| 장면 종류 | 생성 프레임당 (1080p) | 이유 |
|---|---|---|
| 화면 전체가 움직임 (카메라 이동, 큰 액션) | 약 55ms | RIFE 전체 화면 처리 |
| 일부만 움직임 (대사, 캐릭터 동작) | 약 5~15ms | **바뀐 영역만 잘라서** 처리 |
| 원본 그림 유지 (정지, 보호, 컷) | ~0ms | AI를 거치지 않음 |

최적화 내용
- **바뀐 영역만 처리**: 입만 움직이는 대사 장면이면 입 주변만 AI에 넣고 나머지는 원본 그대로 (품질 차이 PSNR 50dB 이상 = 육안 구분 불가)
- **묶음 처리**: 같은 그림 쌍 사이의 중간 프레임 4~8장을 한 번에 생성 (GPU 호출 오버헤드 감소)
- **거의 원본인 프레임 건너뛰기**: 원본 그림 시점과 2% 이내 차이인 프레임은 원본 사용 (약 5%)
- FP16 + channels_last, GPU→CPU 비동기 복사, 효과도 GPU에서 처리
- 테스트 영상 기준 **3.1배 빨라짐** (81초 → 26초)
- 변환 후 이 PC의 실제 속도를 기억해서 다음 '예상 변환 시간'을 더 정확하게 계산

진행 상황 탭: 단계별 상태, 원본/생성 프레임 실시간 미리보기, 남은 시간·완료 예상 시각·속도·영상 위치·파일 크기·GPU 메모리

## GUI 없이 실행 (배치/테스트)

```
%USERPROFILE%\.animeflow\venv\Scripts\python.exe tests\run_headless.py 입력.mp4 출력.mp4 60 rife fx
```
