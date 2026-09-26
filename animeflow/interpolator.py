"""중간 프레임 생성 엔진.

- RIFE (로컬 AI, GPU): 품질 최고. third_party/Practical-RIFE/train_log 에 모델 필요.
- OpenCV 광학 흐름 (DIS): AI 없이 동작하는 대체 엔진.
- 단순 블렌드: 가장 빠름, 테스트용.
"""
from __future__ import annotations

import sys
from typing import Callable

import cv2
import numpy as np

from .config import RIFE_MODEL_DIR, RIFE_ROOT


class Interpolator:
    name = "base"
    gpu_native = False

    def interpolate(self, ia: int, fa: np.ndarray, ib: int, fb: np.ndarray, r: float) -> np.ndarray:
        raise NotImplementedError

    def close(self):
        pass


class BlendInterpolator(Interpolator):
    name = "단순 블렌드"

    def interpolate(self, ia, fa, ib, fb, r):
        return cv2.addWeighted(fa, 1.0 - r, fb, r, 0.0)


class FlowInterpolator(Interpolator):
    """양방향 광학 흐름으로 두 프레임을 t 위치로 워핑 후 합성. 키 쌍마다 흐름은 한 번만 계산."""
    name = "OpenCV 광학 흐름"

    def __init__(self, work_width: int = 960):
        self.work_width = work_width
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self._pair = None
        self._flows = None
        self._grid = None

    def _compute(self, fa, fb):
        h, w = fa.shape[:2]
        sw = min(self.work_width, w)
        sh = int(round(h * sw / w))
        ga = cv2.cvtColor(cv2.resize(fa, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(cv2.resize(fb, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        f01 = self.dis.calc(ga, gb, None)
        f10 = self.dis.calc(gb, ga, None)
        k = w / sw
        f01 = cv2.resize(f01, (w, h), interpolation=cv2.INTER_LINEAR) * k
        f10 = cv2.resize(f10, (w, h), interpolation=cv2.INTER_LINEAR) * k
        if self._grid is None or self._grid[0].shape != (h, w):
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._grid = (gx, gy)
        return f01, f10

    def interpolate(self, ia, fa, ib, fb, r):
        if self._pair != (ia, ib):
            self._flows = self._compute(fa, fb)
            self._pair = (ia, ib)
        f01, f10 = self._flows
        gx, gy = self._grid
        # Super SloMo 방식의 중간 시점 흐름 근사
        ft0 = -(1 - r) * r * f01 + r * r * f10
        ft1 = (1 - r) * (1 - r) * f01 - r * (1 - r) * f10
        wa = cv2.remap(fa, gx + ft0[..., 0], gy + ft0[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wb = cv2.remap(fb, gx + ft1[..., 0], gy + ft1[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.addWeighted(wa, 1.0 - r, wb, r, 0.0)


class RifeInterpolator(Interpolator):
    name = "RIFE (AI)"
    gpu_native = True  # 결과를 GPU 텐서로 바로 넘길 수 있음

    def __init__(self, scale: float = 1.0, fp16: bool = True, log: Callable[[str], None] = print):
        import torch
        import torch.nn.functional as F

        if not (RIFE_MODEL_DIR / "flownet.pkl").exists():
            raise FileNotFoundError(
                f"RIFE 모델이 없습니다: {RIFE_MODEL_DIR / 'flownet.pkl'}\n"
                "README의 안내대로 모델을 받아 train_log 폴더에 넣어주세요.")
        if str(RIFE_ROOT) not in sys.path:
            sys.path.insert(0, str(RIFE_ROOT))
        import train_log.IFNet_HDv3 as ifnet  # type: ignore
        from train_log.RIFE_HDv3 import Model  # type: ignore

        self.torch, self.F = torch, F
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda = self.device.type == "cuda"
        if not self.cuda:
            log("경고: CUDA GPU를 찾지 못해 CPU로 RIFE를 실행합니다 (매우 느림).")
        # 부분 영역(크기가 매번 다름)을 처리하므로 크기별 자동 튜닝은 끈다 (실측: 끄는 쪽이 같거나 빠름)
        torch.backends.cudnn.benchmark = False
        torch.set_grad_enabled(False)

        model = Model()
        if not hasattr(model, "version"):
            model.version = 0
        model.load_model(str(RIFE_MODEL_DIR), -1)
        model.eval()
        model.device()
        self.fp16 = fp16 and self.cuda
        self.dtype = torch.float16 if self.fp16 else torch.float32
        if self.fp16:
            # 워핑(grid_sample)만 FP32로 계산하도록 감싸서 모델 전체를 FP16으로 실행 (~20% 빠름)
            if not getattr(ifnet, "_af_patched", False):
                orig = ifnet.warp
                ifnet.warp = lambda inp, flow: orig(inp.float(), flow.float()).to(inp.dtype)
                ifnet._af_patched = True
            model.flownet.half()
        if self.cuda:
            model.flownet.to(memory_format=torch.channels_last)
        self.model = model
        self.scale = scale
        self._tensors: dict[int, object] = {}
        self._pad = None
        self.stats_full = 0  # 전체 화면 처리한 그림 쌍 수
        self.stats_crop = 0  # 바뀐 영역만 처리한 그림 쌍 수
        log(f"RIFE 모델 로드 완료 (버전 {model.version}, 장치 {self.device}, FP16 {self.fp16})")

    def _to_tensor(self, idx: int, frame: np.ndarray):
        t = self._tensors.get(idx)
        if t is not None:
            return t
        torch = self.torch
        h, w = frame.shape[:2]
        if self._pad is None:
            m = max(128, int(128 / self.scale))
            ph, pw = ((h - 1) // m + 1) * m, ((w - 1) // m + 1) * m
            self._pad = (0, pw - w, 0, ph - h)
            self._hw = (h, w)
        x = torch.from_numpy(frame).to(self.device, non_blocking=True)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.dtype).div_(255.0)
        x = self.F.pad(x, self._pad)
        if self.cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        # 최근 두 키만 GPU에 보관
        if len(self._tensors) >= 2:
            for k in sorted(self._tensors)[:-1]:
                del self._tensors[k]
        self._tensors[idx] = x
        return x

    def key_tensor(self, idx: int, frame: np.ndarray):
        """키 프레임 원본 (1,3,H,W) - GPU에 이미 올라가 있으면 재사용."""
        h, w = frame.shape[:2]
        return self._to_tensor(idx, frame)[:, :, :h, :w]

    def interpolate_tensor(self, ia, fa, ib, fb, r):
        i0 = self._to_tensor(ia, fa)
        i1 = self._to_tensor(ib, fb)
        if self.model.version >= 3.9:
            out = self.model.inference(i0, i1, float(r), self.scale)
        else:
            out = self.model.inference(i0, i1, self.scale)  # 구버전: 중간(0.5)만 지원
        h, w = self._hw
        return out[:, :, :h, :w]

    CROP_ALIGN = 128     # 모델 입력 크기 배수
    CROP_MARGIN = 48     # 바뀐 영역 주변으로 더 보는 여유 (px)
    FEATHER = 16         # 잘라낸 영역 경계를 부드럽게 섞는 폭 (px)
    CROP_MAX_AREA = 0.7  # 바뀐 영역이 이보다 크면 전체 화면 처리

    def _changed_box(self, i0, i1):
        """두 그림에서 실제로 바뀐 영역 (패딩된 좌표, 128 배수로 정렬). 없으면 None."""
        h, w = self._hw
        d = (i0[:, :, :h, :w] - i1[:, :, :h, :w]).abs().amax(dim=1, keepdim=True)
        # 4x4 평균으로 압축 노이즈를 누른 뒤 8x8 블록 단위로 '바뀐 곳' 표시
        d4 = self.F.avg_pool2d(d.float(), 4, ceil_mode=True)
        blk = self.F.max_pool2d(d4, 2, ceil_mode=True)[0, 0] > 0.045
        ys = blk.any(dim=1).nonzero()
        if ys.numel() == 0:
            return None
        xs = blk.any(dim=0).nonzero()
        y0, y1 = int(ys[0]) * 8, (int(ys[-1]) + 1) * 8
        x0, x1 = int(xs[0]) * 8, (int(xs[-1]) + 1) * 8
        m, a = self.CROP_MARGIN, self.CROP_ALIGN
        ph, pw = i0.shape[2], i0.shape[3]
        y0, x0 = max(0, (y0 - m) // a * a), max(0, (x0 - m) // a * a)
        y1, x1 = min(ph, -(-(y1 + m) // a) * a), min(pw, -(-(x1 + m) // a) * a)
        return y0, y1, x0, x1

    def _feather(self, y0, y1, x0, x1, ph, pw):
        """잘라낸 영역 가장자리(화면 테두리가 아닌 쪽)를 0→1 로 서서히 섞는 마스크."""
        torch, f = self.torch, self.FEATHER
        yy = torch.arange(y1 - y0, device=self.device, dtype=torch.float32)
        xx = torch.arange(x1 - x0, device=self.device, dtype=torch.float32)
        wy = torch.ones_like(yy)
        wx = torch.ones_like(xx)
        if y0 > 0:
            wy = torch.minimum(wy, (yy + 1) / f)
        if y1 < ph:
            wy = torch.minimum(wy, (y1 - y0 - yy) / f)
        if x0 > 0:
            wx = torch.minimum(wx, (xx + 1) / f)
        if x1 < pw:
            wx = torch.minimum(wx, (x1 - x0 - xx) / f)
        return (wy.clamp(0, 1)[:, None] * wx.clamp(0, 1)[None, :]).to(self.dtype)[None, None]

    def interpolate_batch(self, ia, fa, ib, fb, ratios):
        """같은 그림 쌍의 중간 프레임 여러 장을 한 번에 생성.

        - 한 번의 호출로 여러 장 → GPU 호출 오버헤드 감소
        - 바뀐 영역만 잘라서 처리 → 입만 움직이는 대사 장면 등에서 수 배 빠름
        반환: (B,3,H,W) 텐서
        """
        torch = self.torch
        i0 = self._to_tensor(ia, fa)
        i1 = self._to_tensor(ib, fb)
        h, w = self._hw
        n = len(ratios)
        ph, pw = i0.shape[2], i0.shape[3]
        box = self._changed_box(i0, i1)
        if box is None:
            return i0[:, :, :h, :w].expand(n, -1, -1, -1)
        y0, y1, x0, x1 = box
        full = (y1 - y0) * (x1 - x0) >= self.CROP_MAX_AREA * ph * pw
        if full:
            y0, y1, x0, x1 = 0, ph, 0, pw
        c0 = i0[:, :, y0:y1, x0:x1]
        c1 = i1[:, :, y0:y1, x0:x1]
        if not full:
            c0 = c0.contiguous(memory_format=torch.channels_last)
            c1 = c1.contiguous(memory_format=torch.channels_last)
        ts = torch.tensor(ratios, device=self.device, dtype=self.dtype).view(n, 1, 1, 1)
        if self.model.version >= 3.9:
            out = self.model.inference(c0.expand(n, -1, -1, -1), c1.expand(n, -1, -1, -1), ts, self.scale)
        else:
            out = torch.cat([self.model.inference(c0, c1, self.scale)] * n)
        self.stats_full += int(full)
        self.stats_crop += int(not full)
        if full:
            return out[:, :, :h, :w]
        res = i0.expand(n, -1, -1, -1).clone()
        base = i0[:, :, y0:y1, x0:x1]
        res[:, :, y0:y1, x0:x1] = base + (out.to(self.dtype) - base) * self._feather(y0, y1, x0, x1, ph, pw)
        return res[:, :, :h, :w]

    def interpolate(self, ia, fa, ib, fb, r):
        out = self.interpolate_tensor(ia, fa, ib, fb, r)
        out = out[0].float().clamp(0, 1).mul_(255).round_().to(self.torch.uint8).permute(1, 2, 0)
        return out.contiguous().cpu().numpy()

    def close(self):
        self._tensors.clear()
        if self.cuda:
            self.torch.cuda.empty_cache()


def rife_available() -> tuple[bool, str]:
    if not (RIFE_MODEL_DIR / "flownet.pkl").exists():
        return False, "RIFE 모델 파일 없음 (train_log/flownet.pkl)"
    try:
        import torch
    except ImportError:
        return False, "PyTorch 미설치"
    if not torch.cuda.is_available():
        return True, "사용 가능 (GPU 없음: CPU로 느리게 동작)"
    return True, f"사용 가능 · {torch.cuda.get_device_name(0)}"


def make_interpolator(backend: str, scale: float, fp16: bool, log: Callable[[str], None]) -> Interpolator:
    if backend == "rife":
        try:
            return RifeInterpolator(scale, fp16, log)
        except Exception as e:  # noqa: BLE001
            log(f"RIFE를 사용할 수 없어 OpenCV 광학 흐름으로 대체합니다: {e}")
            return FlowInterpolator()
    if backend == "opencv":
        return FlowInterpolator()
    return BlendInterpolator()
