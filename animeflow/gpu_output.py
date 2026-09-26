"""GPU 출력 단계: (효과 적용) → uint8 변환 → 고정 메모리로 비동기 복사.

복사가 끝나기를 기다리지 않고 바로 다음 프레임 작업을 GPU에 넣기 때문에
GPU가 쉬지 않는다. 인코더 스레드가 PendingFrame.get() 할 때 필요한 만큼만 기다린다.
"""
from __future__ import annotations

import numpy as np

from .config import Settings
from .effects import TorchEffects, effects_enabled


class PendingFrame:
    __slots__ = ("buf", "event")

    def __init__(self, buf, event):
        self.buf, self.event = buf, event

    def get(self) -> np.ndarray:
        self.event.synchronize()
        return self.buf.numpy()


def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class GpuOutput:
    RING = 32  # 출력 큐(12) + 한 번에 만드는 묶음(최대 8) + 인코딩 중 프레임보다 넉넉하게

    def __init__(self, s: Settings, h: int, w: int, fp16: bool = True):
        import torch
        self.torch = torch
        self.device = torch.device("cuda")
        self.dtype = torch.float16 if fp16 else torch.float32
        self.h, self.w = h, w
        self.fx = TorchEffects(s, h, w, self.device, self.dtype) if effects_enabled(s) else None
        self.ring = [torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True) for _ in range(self.RING)]
        self.i = 0

    @property
    def has_effects(self) -> bool:
        return self.fx is not None

    def upload(self, frame: np.ndarray):
        x = self.torch.from_numpy(frame).to(self.device, non_blocking=True)
        return x.permute(2, 0, 1).unsqueeze(0).to(self.dtype).div_(255.0)

    def finish(self, x, key=None) -> PendingFrame:
        """x: (1,3,H,W) 0~1 텐서. 원본 텐서는 수정하지 않는다. key: 원본 그림 번호 (효과 계산 재사용용)."""
        torch = self.torch
        with torch.no_grad():
            if self.fx is not None:
                x = self.fx(x.to(self.dtype), key)
            out = x[0].clamp(0, 1).mul_(255).round_().to(torch.uint8).permute(1, 2, 0).contiguous()
        buf = self.ring[self.i]
        self.i = (self.i + 1) % self.RING
        buf.copy_(out, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        return PendingFrame(buf, ev)

    def close(self):
        self.ring = []
