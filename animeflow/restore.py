"""원본 그림 보정 (애니메이션 전용 복원 AI).

Real-ESRGAN AnimeVideo v3 로 압축 노이즈·번진 선을 정리하고, 필요하면 해상도를 올린다.
중복 제거된 '고유 그림'에만 적용하므로 영상 전체 프레임 수보다 훨씬 적게 돈다.
보정된 그림으로 중간 프레임을 만들기 때문에 생성된 프레임도 깨끗해진다.
"""
from __future__ import annotations

import numpy as np

from .config import RESTORE_MODEL

# 출력 해상도 선택 → 세로 픽셀 수 (None = 원본 해상도 유지)
RESTORE_RES = {"source": None, "1080": 1080, "1440": 1440}


def restore_available() -> bool:
    return RESTORE_MODEL.exists()


def output_size(src_w: int, src_h: int, res: str) -> tuple[int, int]:
    target = RESTORE_RES.get(res)
    if not target or target <= src_h:
        return src_w, src_h
    w = int(round(src_w * target / src_h / 2)) * 2
    return w, target


class Restorer:
    def __init__(self, out_w: int, out_h: int, fp16: bool = True):
        import spandrel
        import torch

        self.torch = torch
        self.F = torch.nn.functional
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        desc = spandrel.ModelLoader().load_from_file(str(RESTORE_MODEL))
        net = desc.model.eval().to(self.device)
        self.dtype = torch.float16 if (fp16 and self.device.type == "cuda") else torch.float32
        net = net.to(self.dtype)
        if self.device.type == "cuda":
            net = net.to(memory_format=torch.channels_last)
        self.net = net
        self.out_w, self.out_h = out_w, out_h

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 → 보정된 BGR uint8 (출력 해상도)."""
        torch, F = self.torch, self.F
        with torch.no_grad():
            x = torch.from_numpy(bgr).to(self.device).permute(2, 0, 1)[None]
            x = x[:, [2, 1, 0]].to(self.dtype).div_(255)  # 모델은 RGB
            if self.device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            y = self.net(x)                                  # 4배
            y = F.interpolate(y.float(), (self.out_h, self.out_w), mode="area")  # 원하는 크기로 (GPU에서)
            y = y[:, [2, 1, 0]].clamp_(0, 1).mul_(255).round_().to(torch.uint8)
            return y[0].permute(1, 2, 0).contiguous().cpu().numpy()
