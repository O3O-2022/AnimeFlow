"""깊이 기반 입체 조명 (2.5D 리라이팅).

1) Depth Anything V2 (Small) 로 장면의 앞뒤(깊이)를 추정
2) 깊이 → 표면 방향(노멀)
3) 광원 방향·색: 배경에서 가장 밝은 곳(노을 하늘 등)을 찾아 자동 추정 (또는 사용자가 지정)
4) 셀 그림자(광원 반대편을 차가운 색으로), 림 라이트(광원 쪽 윤곽을 광원 색으로),
   발광(고글·눈처럼 하얗게 밝은 부분이 빛나게), 전체 톤을 극적으로

모든 계산은 GPU의 저해상도(짧은 변 ~392px)에서 하고 결과만 원본 크기로 올려서 빠르다.
"""
from __future__ import annotations

import math
import sys

from .config import DEPTH_CKPT, DEPTH_ROOT

LIGHT_DIRS = {
    # (x, y, z): 화면 좌표계 (x 오른쪽, y 아래, z 화면 밖=카메라 쪽)
    "left": (-1.0, -0.6, 0.45),
    "right": (1.0, -0.6, 0.45),
    "top": (0.0, -1.0, 0.35),
    "back": (0.0, -0.35, -0.75),   # 역광: 인물 뒤에서 오는 빛 → 윤곽이 빛남
}


# 광원 색 (BGR). None = 장면에서 자동 추정
LIGHT_COLORS = {
    "auto": None,
    "warm": (0.35, 0.65, 1.0),   # 노을·불빛
    "cool": (1.0, 0.85, 0.55),   # 달빛·네온 블루
    "red": (0.45, 0.35, 1.0),    # 붉은 조명
    "white": (1.0, 1.0, 1.0),
}


# 톤 파라미터 (강도 100% 기준). 참고 이미지(before/after 쌍)로 자동 보정한 값.
DEFAULT_PARAMS = {
    "dark_fg": 0.02,     # 인물 어둡게 (거의 안 함: 참고 이미지들은 전체 밝기를 유지)
    "dark_bg": 0.35,     # 배경 어둡게
    "shadow": 1.0,       # 광원 반대편 그림자 깊이
    "tint": 0.1,         # 그림자에 광원 보색을 섞는 정도
    "lit": 0.25,         # 빛 받는 면 밝게
    "contrast": 0.35,    # 대비 (평균 밝기 유지)
    "highlight": 0.18,   # 하이라이트 밝게
    "sat": -0.12,        # 채도 (대비를 올리면 채도도 같이 올라가서 살짝 낮춤)
    "rim": 1.0,          # 림 라이트
}


def depth_available() -> bool:
    return DEPTH_CKPT.exists()


class DepthEstimator:
    def __init__(self, device, dtype, short_side: int = 336):
        import torch
        if str(DEPTH_ROOT) not in sys.path:
            sys.path.insert(0, str(DEPTH_ROOT))
        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

        self.torch = torch
        self.F = torch.nn.functional
        m = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        m.load_state_dict(torch.load(str(DEPTH_CKPT), map_location="cpu"))
        m = m.to(device).eval()
        if dtype == torch.float16:
            m = m.half()
        self.model, self.device, self.dtype = m, device, dtype
        self.short = short_side
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def size_for(self, h: int, w: int) -> tuple[int, int]:
        k = self.short / min(h, w)
        return max(14, round(h * k / 14) * 14), max(14, round(w * k / 14) * 14)

    def __call__(self, x):
        """x: (1,3,H,W) BGR 0~1 → (1,1,h,w) 역깊이 (클수록 가까움), 저해상도."""
        h, w = self.size_for(x.shape[2], x.shape[3])
        rgb = x[:, [2, 1, 0]].float()
        inp = self.F.interpolate(rgb, (h, w), mode="bilinear", align_corners=False)
        inp = ((inp - self.mean) / self.std).to(self.dtype)
        with self.torch.no_grad():
            d = self.model(inp)
        return d[:, None].float()


class Relighter:
    def __init__(self, strength: float, light_dir: str, device, dtype, blur_fn, light_color: str = "auto"):
        import torch
        self.torch = torch
        self.F = torch.nn.functional
        self.k = max(0.0, min(2.0, strength))  # 0~200%
        self.light_dir = light_dir
        self.light_color = light_color
        self.depth = DepthEstimator(device, dtype)
        self.device = device
        self.blur = blur_fn
        # 시간에 따라 부드럽게 변하도록 이전 값 유지 (깜빡임 방지)
        self._L = None
        self._color = None
        self._range = None
        self._depth_cache: dict = {}  # 원본 그림 번호 → 깊이 (최근 몇 장만)
        self._mean = None
        self.p = dict(DEFAULT_PARAMS)

    # ------------------------------------------------------------ 보조
    @staticmethod
    def _smooth(a, b, t):
        return b if a is None else a + (b - a) * t

    def _smoothstep(self, e0, e1, x):
        t = ((x - e0) / (e1 - e0)).clamp(0, 1)
        return t * t * (3 - 2 * t)

    def _estimate_light(self, xs, dn):
        """배경(먼 곳)의 가장 밝은 영역 → 광원 방향과 색."""
        torch = self.torch
        lum = 0.114 * xs[:, 0] + 0.587 * xs[:, 1] + 0.299 * xs[:, 2]  # (1,h,w)
        far = dn[:, 0] < 0.45
        cand = lum[far] if int(far.sum()) > 200 else lum.flatten()
        thr = torch.quantile(cand.float(), 0.97)
        m = ((lum >= thr) & (far if int(far.sum()) > 200 else torch.ones_like(far))).float()
        if float(m.sum()) < 10:
            m = (lum >= torch.quantile(lum.flatten().float(), 0.97)).float()
        h, w = lum.shape[1:]
        ys = torch.linspace(-1, 1, h, device=self.device).view(1, h, 1)
        xs_ = torch.linspace(-1, 1, w, device=self.device).view(1, 1, w)
        tot = m.sum().clamp(min=1)
        cx, cy = float((m * xs_).sum() / tot), float((m * ys).sum() / tot)
        color = (xs * m[:, None]).sum((2, 3)) / tot  # (1,3) BGR
        color = color / color.max().clamp(min=1e-3)
        # 밝은 곳이 배경 쪽이면 인물 뒤에서 오는 빛(역광) 성분을 섞는다
        L = torch.tensor([cx * 1.2, cy * 1.2 - 0.3, -0.25], device=self.device)
        return L / L.norm().clamp(min=1e-3), color.view(1, 3, 1, 1)

    # ------------------------------------------------------------ 메인
    def __call__(self, x, cache_key=None):
        torch, F = self.torch, self.F
        k = self.k
        H, W = x.shape[2], x.shape[3]
        # 깊이는 원본 그림마다 한 번만 추정 (중간 프레임은 가까운 원본 그림의 깊이를 재사용)
        d = self._depth_cache.get(cache_key) if cache_key is not None else None
        if d is None:
            d = self.depth(x)
            if cache_key is not None:
                if len(self._depth_cache) >= 3:
                    self._depth_cache.pop(next(iter(self._depth_cache)))
                self._depth_cache[cache_key] = d
        h, w = d.shape[2], d.shape[3]
        xs = F.interpolate(x.float(), (h, w), mode="area")

        # 깊이 정규화 (0=멀리, 1=가까이), 범위도 시간에 따라 부드럽게
        flat = d.flatten()
        rng = torch.stack([torch.quantile(flat, 0.02), torch.quantile(flat, 0.98)])
        self._range = self._smooth(self._range, rng, 0.15)
        lo, hi = self._range[0], self._range[1]
        dn = ((d - lo) / (hi - lo).clamp(min=1e-3)).clamp(0, 1)

        # 광원 방향·색
        est_L, est_color = self._estimate_light(xs, dn)
        if self.light_dir in LIGHT_DIRS:
            L = torch.tensor(LIGHT_DIRS[self.light_dir], device=self.device)
            L = L / L.norm()
        else:
            L = est_L
        color = LIGHT_COLORS.get(self.light_color)
        color = est_color if color is None else torch.tensor(color, device=self.device).view(1, 3, 1, 1)
        color = color / color.amax().clamp(min=1e-3)
        color = 0.35 + 0.65 * color  # 너무 탁한 색이 되지 않게
        self._L = self._smooth(self._L, L, 0.08)
        self._color = self._smooth(self._color, color, 0.08)
        L = self._L / self._L.norm().clamp(min=1e-3)
        color = self._color
        lxy = L[:2] / L[:2].norm().clamp(min=1e-3)

        fg = self._smoothstep(0.30, 0.60, dn)  # 인물·가까운 물체

        # --- 셀 그림자: 흐린 깊이로 만든 표면 방향 vs 광원
        db = self.blur(dn.expand(-1, 3, -1, -1).contiguous(), max(1.5, h / 150))[:, :1]
        gx = F.pad(db[:, :, :, 2:] - db[:, :, :, :-2], (1, 1, 0, 0), mode="replicate")
        gy = F.pad(db[:, :, 2:, :] - db[:, :, :-2, :], (0, 0, 1, 1), mode="replicate")
        bump = 2.2 * min(h, w) / 20  # 클수록 얼굴·몸의 굴곡에 그림자가 더 많이 생김
        n = torch.cat([-gx * bump, -gy * bump, torch.ones_like(gx)], 1)
        n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-4)
        ndl = (n * L.view(1, 3, 1, 1)).sum(1, keepdim=True)
        lit = self._smoothstep(0.10, 0.18, ndl)  # 경계가 또렷한 셀 그림자 (광원을 정면으로 보는 면만 밝게)
        shadow = (1 - lit) * fg

        # --- 림 라이트: 깊이가 급하게 바뀌는 윤곽 중 '광원 쪽을 향한 가까운 쪽'
        e = dn
        ex = F.pad(e[:, :, :, 2:] - e[:, :, :, :-2], (1, 1, 0, 0), mode="replicate")
        ey = F.pad(e[:, :, 2:, :] - e[:, :, :-2, :], (0, 0, 1, 1), mode="replicate")
        mag = torch.sqrt(ex * ex + ey * ey)
        out_x, out_y = -ex / mag.clamp(min=1e-4), -ey / mag.clamp(min=1e-4)  # 가까운 쪽 → 먼 쪽 방향
        facing = (out_x * lxy[0] + out_y * lxy[1]).clamp(0, 1)
        edge = self._smoothstep(0.05, 0.14, mag)
        rim = edge * facing ** 0.8 * self._smoothstep(0.25, 0.5, dn)

        p_rim = self.p["rim"]
        # --- 림 라이트의 은은한 번짐도 저해상도에서 계산
        rim_glow = self.blur(rim.expand(-1, 3, -1, -1).contiguous(), max(0.8, h / 260))[:, :1]
        r = (rim * 0.9 + rim_glow * (0.15 + 0.1 * k)).clamp(0, 1) * min(1.0, k) * p_rim * (1 + 0.6 * max(0.0, k - 1))

        # --- 저해상도 맵 4장을 한 번에 원본 크기로
        maps = F.interpolate(torch.cat([shadow, lit * fg, fg, r], 1), (H, W), mode="bilinear", align_corners=False)
        dt = x.dtype
        maps = maps.to(dt)
        shadow_u, lit_u, fg_u, r_u = maps[:, 0:1], maps[:, 1:2], maps[:, 2:3], maps[:, 3:4]
        color_d = color.to(dt)
        xf = x

        # 강도 k(0~2)가 커져도 새하얗게/새까맣게 무너지지 않도록 포화 곡선 사용
        p = self.p
        sat_k = lambda a: 1 - math.exp(-a * k)  # noqa: E731
        # 1) 배경은 조금 어둡게, 인물은 거의 그대로 (전체 밝기는 유지하는 게 핵심)
        dark = sat_k(p["dark_fg"]) + (sat_k(p["dark_bg"]) - sat_k(p["dark_fg"])) * (1 - fg_u)
        out = xf * (1 - dark)
        # 2) 대비: 평균 밝기는 두고 어두운 곳은 더 어둡게, 밝은 곳은 더 밝게
        lum_o = 0.114 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.299 * out[:, 2:3]
        self._mean = self._smooth(self._mean, lum_o.mean(), 0.1)
        out = (self._mean + (out - self._mean) * (1 + k * p["contrast"])).clamp(0, 1)
        # 3) 빛 받는 면은 밝게 + 하이라이트 강조 (그림자 쪽에는 적용 안 함)
        lum_o = 0.114 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.299 * out[:, 2:3]
        not_shadow = 1 - shadow_u
        out = (out * (1 + lit_u * (k * p["lit"]) * (0.7 + 0.3 * color_d))).clamp(0, 1)
        out = out + (1 - out) * self._smoothstep(0.45, 0.9, lum_o) * (k * p["highlight"]) * not_shadow
        # 4) 광원 반대편 셀 그림자 — 마지막에 곱해서 다른 보정이 되살리지 못하게 (역광이면 실루엣처럼)
        tint = ((1 - color) * p["tint"] + torch.tensor([0.21, 0.24, 0.31], device=self.device).view(1, 3, 1, 1))
        sh = shadow_u * sat_k(p["shadow"])
        out = out * (1 - sh * (1 - tint.clamp(0.15, 0.95).to(dt)))
        lum_s = 0.114 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.299 * out[:, 2:3]
        out = (lum_s + (out - lum_s) * (1 + 0.25 * sh)).clamp(0, 1)  # 그림자는 채도를 살짝 올려 따뜻하게 (그림 그릴 때처럼, 피부가 칙칙해지지 않게)
        # 5) 채도
        lum_o = 0.114 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.299 * out[:, 2:3]
        # 채도 조정은 배경에만 (인물 피부가 칙칙해지지 않도록 인물은 원래 채도 유지)
        out = (lum_o + (out - lum_o) * (1 + p["sat"] * k * (1 - fg_u))).clamp(0, 1)
        # 6) 밝은 부분이 하얗게 날아가지 않도록 부드럽게 눌러줌 (필름 숄더)
        knee = 0.78
        out = torch.where(out > knee, knee + (1 - knee) * (1 - torch.exp(-(out - knee) / (1 - knee) * 1.6)) / (1 - math.exp(-1.6)), out)
        # 7) 림 라이트 (광원 색)
        out = 1 - (1 - out) * (1 - r_u * color_d)

        # 6) 발광: 조명·창문·표시등처럼 '색이 있는 밝은 빛'과 '아주 작은 흰 점'(눈빛 등)만.
        #    흰 옷·흰 벽처럼 넓게 밝은 면은 빛나지 않게 한다. (절반 해상도에서 계산)
        xh = F.avg_pool2d(xf.float(), 2)
        Hh = H // 2
        lum = 0.114 * xh[:, 0:1] + 0.587 * xh[:, 1:2] + 0.299 * xh[:, 2:3]
        local = self.blur(lum.expand(-1, 3, -1, -1).contiguous(), max(3.0, Hh / 25))[:, :1]
        mx = xh.amax(1, keepdim=True)
        sat = (mx - xh.amin(1, keepdim=True)) / mx.clamp(min=1e-3)
        colored = (self._smoothstep(0.35, 0.6, sat) * self._smoothstep(0.6, 0.85, mx)
                   * self._smoothstep(0.05, 0.2, lum - local))
        # 흰 점은 주변보다 '확연히' 밝아야 함 (흰 옷의 작은 조각이 반짝이지 않도록)
        white = self._smoothstep(0.85, 0.97, lum) * self._smoothstep(0.30, 0.45, lum - local)
        ks = max(3, int(Hh / 110) | 1)  # 이보다 작은 흰 점만
        opened = F.max_pool2d(-F.max_pool2d(-white, ks, 1, ks // 2), ks, 1, ks // 2)
        # 색 있는 빛도 '작은' 것만 (조명·창문·표시등). 인물 옷이나 큰 물체처럼 넓은 색 면은 제외
        kc = max(5, int(Hh / 30) | 1)
        opened_c = F.max_pool2d(-F.max_pool2d(-colored, kc, 1, kc // 2), kc, 1, kc // 2)
        emit = torch.maximum((colored - opened_c).clamp(0, 1), (white - opened).clamp(0, 1) * 0.6)
        src = (xh * emit).contiguous()
        glow = self.blur(src, max(1.5, Hh / 160)) * 1.2 + self.blur(src, max(3.0, Hh / 45)) * 0.9
        glow = F.interpolate(torch.cat([glow, emit], 1), (H, W), mode="bilinear", align_corners=False).to(dt)
        out = 1 - (1 - out) * (1 - (glow[:, :3] * (1.4 * min(k, 1.3))).clamp(0, 1))
        out = torch.maximum(out, xf * glow[:, 3:4])  # 발광체 자체는 원래 밝기 유지
        return out.clamp(0, 1)
