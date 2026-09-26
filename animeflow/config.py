"""경로 및 작업 설정."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FFMPEG_DIR = ROOT / "tools" / "ffmpeg"
RIFE_ROOT = ROOT / "third_party" / "Practical-RIFE"
RIFE_MODEL_DIR = RIFE_ROOT / "train_log"
CACHE_DIR = ROOT / "cache"
DEPTH_ROOT = ROOT / "third_party" / "Depth-Anything-V2"
RESTORE_MODEL = ROOT / "third_party" / "restore" / "realesr-animevideov3.pth"
DEPTH_CKPT = DEPTH_ROOT / "checkpoints" / "depth_anything_v2_vits.pth"


# 빠른 동작 처리 모드별 기준: (이동량 한계[화면 폭 비율], 추적 실패율 한계). None = 보호 안 함
FAST_MOTION_MODES = {
    "auto": (0.20, 0.45),
    "ai": None,
    "hold": (0.10, 0.20),
    # 선 안정 우선: 작은 움직임만 보간. 큰 움직임은 원래 그림을 그대로 둬서 선이 따로 노는 느낌을 없앤다
    "lines": (0.06, 0.15),
}
# 화면의 넓은 부분이 한 번에 바뀌고 추적도 안 되면 '순간 전환' → 모드와 상관없이 보간하지 않는다
# (실측: 이런 경우 실제 중간 그림은 앞 그림과 같았고, AI는 두 장면을 뭉갠 그림을 만들었다)
INSTANT_CHANGE_AREA = 0.35
INSTANT_CHANGE_RESIDUAL = 0.30


def ffmpeg_bin(name: str = "ffmpeg") -> str:
    """프로젝트 tools/ffmpeg 우선, 없으면 PATH에서 찾는다."""
    local = FFMPEG_DIR / f"{name}.exe"
    if local.exists():
        return str(local)
    return shutil.which(name) or name


# (표시 이름, ffmpeg 인코더) - 컨테이너별 선택 가능한 비디오 코덱
CODECS: dict[str, list[tuple[str, str]]] = {
    "mp4": [
        ("H.264 · NVIDIA GPU (빠름)", "h264_nvenc"),
        ("H.265/HEVC · NVIDIA GPU (빠름, 고효율)", "hevc_nvenc"),
        ("H.264 · CPU x264 (호환성 최고)", "libx264"),
        ("H.265/HEVC · CPU x265 (고효율)", "libx265"),
        ("AV1 · CPU SVT-AV1 (최고 효율, 느림)", "libsvtav1"),
    ],
    "mkv": [
        ("H.264 · NVIDIA GPU (빠름)", "h264_nvenc"),
        ("H.265/HEVC · NVIDIA GPU (빠름, 고효율)", "hevc_nvenc"),
        ("H.264 · CPU x264", "libx264"),
        ("H.265/HEVC · CPU x265", "libx265"),
        ("AV1 · CPU SVT-AV1", "libsvtav1"),
    ],
    "mov": [
        ("ProRes 422 HQ (편집용, 대용량)", "prores_ks"),
        ("H.264 · NVIDIA GPU", "h264_nvenc"),
        ("H.264 · CPU x264", "libx264"),
    ],
    "webm": [
        ("VP9 (웹용)", "libvpx-vp9"),
        ("AV1 · SVT-AV1 (웹용, 고효율)", "libsvtav1"),
    ],
}

CONTAINER_LABELS = {
    "mp4": "MP4 (가장 범용)",
    "mkv": "MKV (자막·다중 오디오 보존)",
    "mov": "MOV (편집 프로그램용)",
    "webm": "WebM (웹 업로드용)",
}


@dataclass
class Settings:
    input_path: str = ""
    output_path: str = ""

    # 프레임
    target_fps: int = 60
    backend: str = "rife"  # rife | opencv | blend
    rife_scale: float = 1.0
    fp16: bool = True

    # 분석 (프로그램 기반, AI 아님)
    analysis_width: int = 384
    dup_pixel_ratio: float = 0.0005  # 변화 픽셀 비율이 이 값 미만이면 같은 그림
    dup_mean_diff: float = 1.2       # 전체 평균 차이가 이 값 미만이어야 같은 그림 (페이드 보호)
    scene_hist_corr: float = 0.55    # 히스토그램 상관이 이 값 미만이면 장면 전환
    max_interp_frames: int = 4       # 이보다 긴 정지(홀드)는 끝부분만 보간

    # 빠른 동작 처리: 보간 AI가 새 포즈를 그리지 못하는(잔상이 생기는) 그림 쌍을 어떻게 할지
    #   auto : 실제로 잔상이 날 쌍만 원본 유지 (인빈시블 실측으로 기준 설정)
    #   ai   : 전부 AI 보간 (끊김 없음, 잔상 감수)
    #   hold : 조금이라도 위험하면 원본 유지 (잔상 없음, 끊김 감수)
    fast_motion_mode: str = "auto"

    # AI 장면 감독 (로컬 Ollama)
    use_ai_director: bool = True
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5vl:7b"
    ai_min_scene_sec: float = 0.5

    # 원본 보정 (애니 전용 복원 AI): 선 정리·압축 노이즈 제거, 선택 시 해상도 올림
    restore: bool = False
    restore_res: str = "source"      # source | 1080 | 1440

    # 후처리 효과 (프레임 작업이 모두 끝난 뒤 적용)
    relight: bool = False            # 깊이 기반 입체 조명 (그림자·림 라이트·발광)
    relight_strength: float = 1.0    # 0~2 (200%)
    light_dir: str = "auto"          # auto | left | right | top | back
    light_color: str = "auto"        # auto | warm | cool | red | white
    shading: bool = False
    shading_strength: float = 0.5
    lighting: bool = False
    lighting_strength: float = 0.5
    vignette: bool = False

    # 한국어 AI 더빙 (변환이 끝난 영상에 한국어 음성 트랙 + 자막 추가)
    dub: bool = False
    dub_vc: bool = True              # 캐릭터 원래 목소리로 음색 변환
    dub_llm: str = "auto"            # 번역 AI (auto = 설치된 것 중 가장 좋은 것)
    dub_context: str = ""            # 작품·인물 메모 (예: 옴니맨(아빠)과 마크(아들). 서로 반말)
    dub_api_key: str = ""            # DeepSeek 클라우드 번역을 고른 경우에만 쓰는 사용자 API 키

    # 출력
    container: str = "mp4"
    video_codec: str = "h264_nvenc"
    quality: int = 18  # CRF/CQ: 낮을수록 고화질

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
