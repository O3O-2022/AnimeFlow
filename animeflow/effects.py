"""후처리 효과 (명암 보정 / 광원 효과 / 비네팅).

모든 프레임 생성(보간)이 끝난 '최종 출력 프레임'에만 적용한다.
- GPU가 있으면 PyTorch로 GPU에서 처리 (프레임당 1~2ms)
- GPU가 없으면 같은 효과를 FFmpeg 필터로 인코딩 단계에서 처리 (CPU, 느림)
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .config import Settings


def effects_enabled(s: Settings) -> bool:
    return s.shading or s.lighting or s.vignette or s.relight


# ---------------------------------------------------------------- GPU (PyTorch)
class TorchEffects:
    """입력/출력: (1,3,H,W) BGR, 0~1 범위 텐서."""

    def __init__(self, s: Settings, h: int, w: int, device, dtype):
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        self.s = s
        self.h, self.w = h, w
        k = h / 1080.0
        self.sh = max(0.0, min(1.0, s.shading_strength))
        self.li = max(0.0, min(1.0, s.lighting_strength))
        self.lc_sigma = max(1.5, 10.0 * k)     # 명암: 로컬 대비 반경
        self.glow_sigma = max(3.0, 18.0 * k)   # 광원: 빛 번짐 반경
        self._kernels: dict[float, object] = {}
        self.device, self.dtype = device, dtype
        self.relighter = None
        if s.relight:
            from .relight import Relighter, depth_available
            if depth_available():
                self.relighter = Relighter(s.relight_strength, s.light_dir, device, dtype, self.blur,
                                           light_color=s.light_color)
        self.vmask = None
        if s.vignette:
            yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                                    torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
            d = torch.hypot(xx - w / 2, yy - h / 2) / math.hypot(w / 2, h / 2)
            c = torch.cos(math.pi / 5 * d)
            self.vmask = (c * c * c * c).to(dtype)[None, None]

    def _kernel(self, sigma: float):
        kern = self._kernels.get(sigma)
        if kern is None:
            r = max(1, int(sigma * 3))
            x = self.torch.arange(-r, r + 1, device=self.device, dtype=self.torch.float32)
            g = self.torch.exp(-(x * x) / (2 * sigma * sigma))
            kern = (g / g.sum()).to(self.dtype)
            self._kernels[sigma] = kern
        return kern

    def blur(self, x, sigma: float):
        """큰 반경 가우시안 블러: 축소 → 분리형 블러 → 확대 (빠름)."""
        F = self.F
        f = max(1, int(sigma // 3))
        y = F.avg_pool2d(x, f) if f > 1 else x
        kern = self._kernel(sigma / f).to(y.dtype)
        r = (kern.numel() - 1) // 2
        y = F.pad(y, (r, r, r, r), mode="replicate")
        y = F.conv2d(y, kern.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
        y = F.conv2d(y, kern.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)
        if f > 1:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y

    @staticmethod
    def luma(x):
        return 0.114 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.299 * x[:, 2:3]

    def __call__(self, x, cache_key=None):
        """cache_key: 같은 원본 그림이면 깊이 추정을 다시 하지 않도록 넘기는 키."""
        s = self.s
        if self.relighter is not None:
            x = self.relighter(x, cache_key)
        if s.shading and self.sh > 0:
            k = self.sh
            x = (x - 0.5) * (1 + 0.22 * k) + 0.5                        # 대비
            x = x.clamp(1e-4, 1).pow(1 - 0.06 * k)                        # 감마 (어두운 부분 깊게)
            lum = self.luma(x)
            x = lum + (x - lum) * (1 + 0.15 * k)                          # 채도
            x = x + (0.6 * k) * (x - self.blur(x, self.lc_sigma))         # 입체감(로컬 대비)
            x = x.clamp(0, 1)
        if s.lighting and self.li > 0:
            k = self.li
            th = 0.78 - 0.18 * k
            hi = ((x - th) / (1 - th)).clamp(0, 1)                        # 밝은 부분만 추출
            glow = self.blur(hi, self.glow_sigma)                          # 빛 번짐
            screen = 1 - (1 - x) * (1 - glow)
            x = x + (screen - x) * (0.25 + 0.55 * k)
            w = self.luma(x).pow(2)                                        # 하이라이트를 살짝 따뜻하게
            x = x + w * self.torch.tensor([-0.03 * k, 0.015 * k, 0.04 * k], device=x.device,
                                          dtype=x.dtype).view(1, 3, 1, 1)
            x = x.clamp(0, 1)
        if self.vmask is not None:
            x = x * self.vmask
        return x


def preview_jpeg(jpeg: bytes, s: Settings) -> bytes:
    """썸네일에 실제 렌더와 같은 효과를 적용 (CPU)."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None or not effects_enabled(s):
        return jpeg
    import torch
    h, w = img.shape[:2]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fx = TorchEffects(s, h, w, dev, torch.float32)
    x = torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        y = fx(x)
    out = (y[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    return cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


# ---------------------------------------------------------------- CPU 대체 (FFmpeg 필터)
def build_filter_chain(s: Settings, height: int) -> str:
    parts: list[str] = []
    if s.shading:
        k = max(0.0, min(1.0, s.shading_strength))
        parts.append(f"eq=contrast={1 + 0.22 * k:.3f}:saturation={1 + 0.15 * k:.3f}:gamma={1 - 0.06 * k:.3f}")
        parts.append(f"unsharp=lx=13:ly=13:la={0.6 * k:.3f}:cx=5:cy=5:ca=0")
    if s.lighting:
        k = max(0.0, min(1.0, s.lighting_strength))
        sigma = max(3.0, 18.0 * height / 1080.0)
        th = 0.78 - 0.18 * k
        parts.append(
            "split=2[lf_base][lf_hi];"
            f"[lf_hi]curves=all='0/0 {th:.3f}/0 1/1',gblur=sigma={sigma:.1f}:steps=2[lf_glow];"
            f"[lf_base][lf_glow]blend=all_mode=screen:all_opacity={0.25 + 0.55 * k:.3f}"
        )
        parts.append(f"colorbalance=rh={0.04 * k:.3f}:gh={0.015 * k:.3f}:bh={-0.03 * k:.3f}")
    if s.vignette:
        parts.append("vignette=angle=PI/5")
    return ",".join(parts)
