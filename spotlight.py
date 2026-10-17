"""
spotlight.py
หัวใจของการประมวลผลภาพ

เทคนิคที่ใช้ (ตรงกับที่เลือกไว้ในฟอร์ม)
  - Image Enhancement / Restoration : CLAHE, Unsharp Masking, Gaussian Blur
  - Color Space Transformation      : BGR <-> LAB (contrast), BGR <-> HSV (dim / desaturate)

แนวคิด
  out(x,y) = M(x,y) * F(x,y) + (1 - M(x,y)) * B(x,y)
  โดย  F = บริเวณโฟกัส (คม สว่าง contrast สูง)
       B = พื้นหลัง (เบลอ มืดลง สีจางลง)
       M = soft mask ที่ไล่ระดับด้วย smoothstep  M = 3t^2 - 2t^3
"""

import cv2
import numpy as np

SHAPES = ("circle", "rect", "bar")

# ค่าพรีเซ็ต 1 = อ่อน (เหมาะกับสไลด์ตัวหนังสือ) ... 3 = เข้ม (เน้นจุดเดียวชัด ๆ)
PRESETS = {
    1: dict(radius=310, feather=190, dim_gamma=0.74, desat=0.06,
            blur_ksize=9,  blur_scale=0.50, sharp_amount=0.35,
            clahe_clip=1.4, bright_gain=1.05),
    2: dict(radius=260, feather=150, dim_gamma=0.58, desat=0.16,
            blur_ksize=15, blur_scale=0.40, sharp_amount=0.60,
            clahe_clip=1.8, bright_gain=1.08),
    3: dict(radius=200, feather=110, dim_gamma=0.36, desat=0.34,
            blur_ksize=23, blur_scale=0.30, sharp_amount=0.90,
            clahe_clip=2.4, bright_gain=1.15),
}


class SpotlightRenderer:
    # ---------------------------------------------------------------- init --
    def __init__(self,
                 radius=260, feather=150, shape="circle",
                 blur_ksize=15, blur_scale=0.40,
                 dim_gamma=0.58, desat=0.16,
                 sharp_amount=0.60, sharp_sigma=1.5,
                 clahe_clip=1.8, bright_gain=1.08,
                 enhance_on=True, ring=False):

        self.radius = radius              # รัศมีส่วนที่คมชัดเต็มที่
        self.feather = feather            # ระยะไล่ระดับขอบ
        self.shape = shape                # circle / rect / bar

        self.blur_ksize = blur_ksize      # ความเบลอของพื้นหลัง
        self.blur_scale = blur_scale      # ย่อก่อนเบลอ -> เร็วขึ้นมาก
        self.dim_gamma = dim_gamma        # ความสว่างพื้นหลัง (1.0 = ไม่หรี่)
        self.desat = desat                # ลดความอิ่มสีพื้นหลัง (0 = สีเดิม)

        self.sharp_amount = sharp_amount  # ความแรง unsharp mask
        self.sharp_sigma = sharp_sigma
        self.bright_gain = bright_gain    # เพิ่มความสว่างในวง

        self.enhance_on = enhance_on
        self.ring = ring

        self._clahe_clip = clahe_clip
        self._clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))

        self._patch = None
        self._patch_key = None

    # ------------------------------------------------------- 1) SOFT MASK --
    def _build_patch(self):
        """สร้าง mask สี่เหลี่ยมขนาด 2R x 2R ไว้ใช้ซ้ำ (แคชไว้จนกว่าค่าจะเปลี่ยน)"""
        key = (self.radius, self.feather, self.shape)
        if self._patch_key == key:
            return self._patch

        R = self.radius + self.feather
        size = 2 * R
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        c = R - 0.5
        dx, dy = xx - c, yy - c

        if self.shape == "circle":
            d = np.sqrt(dx * dx + dy * dy)
        elif self.shape == "rect":
            d = np.maximum(np.abs(dx), np.abs(dy))
        else:                                   # bar : แถบแนวนอน อ่านทีละบรรทัด
            d = np.abs(dy)

        t = np.clip((R - d) / max(self.feather, 1e-6), 0.0, 1.0)
        self._patch = (t * t * (3.0 - 2.0 * t)).astype(np.float32)   # smoothstep
        self._patch_key = key
        return self._patch

    def _mask_and_bbox(self, h, w, cx, cy):
        """คืน (mask, (x0, y0, x1, y1)) เฉพาะบริเวณรอบเคอร์เซอร์"""
        R = self.radius + self.feather
        patch = self._build_patch()

        if self.shape == "bar":
            y0 = max(0, cy - R)
            y1 = min(h, cy + R)
            if y1 <= y0:
                return None, None
            profile = patch[:, R][(y0 - (cy - R)):(y1 - (cy - R))]
            mask = np.repeat(profile[:, None], w, axis=1)
            return mask, (0, y0, w, y1)

        x0, y0 = max(0, cx - R), max(0, cy - R)
        x1, y1 = min(w, cx + R), min(h, cy + R)
        if x1 <= x0 or y1 <= y0:
            return None, None

        px, py = x0 - (cx - R), y0 - (cy - R)
        mask = patch[py:py + (y1 - y0), px:px + (x1 - x0)]
        return mask, (x0, y0, x1, y1)

    # ------------------------------------------------- 2) BACKGROUND LAYER --
    def _background(self, frame):
        """เบลอ + หรี่แสง + ลดความอิ่มสี  (Gaussian Blur + HSV transform)"""
        h, w = frame.shape[:2]

        small = cv2.resize(frame, None,
                           fx=self.blur_scale, fy=self.blur_scale,
                           interpolation=cv2.INTER_AREA)

        k = max(3, int(self.blur_ksize) | 1)        # บังคับเป็นเลขคี่
        small = cv2.GaussianBlur(small, (k, k), 0)

        # ---- Color Space Transformation : BGR -> HSV ----
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= (1.0 - self.desat)           # S : ลดความอิ่มสี
        hsv[..., 2] *= self.dim_gamma               # V : หรี่ความสว่าง
        np.clip(hsv, 0, 255, out=hsv)
        small = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------- 3) FOREGROUND LAYER --
    def _enhance(self, roi):
        """CLAHE บน L channel + Unsharp Masking + เพิ่มความสว่าง"""
        if not self.enhance_on or roi.size == 0:
            return roi

        # ---- Color Space Transformation : BGR -> LAB ----
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        # ---- Unsharp Masking :  g = (1+k)*f - k*(f * G) ----
        if self.sharp_amount > 0.01:
            blur = cv2.GaussianBlur(out, (0, 0), self.sharp_sigma)
            out = cv2.addWeighted(out, 1.0 + self.sharp_amount,
                                  blur, -self.sharp_amount, 0)

        # ---- เพิ่มความสว่างผ่าน V channel ----
        if abs(self.bright_gain - 1.0) > 0.01:
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 2] = np.clip(hsv[..., 2] * self.bright_gain, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return out

    # ------------------------------------------------------- 4) COMPOSITE --
    def render(self, frame, cursor):
        h, w = frame.shape[:2]
        cx, cy = int(cursor[0]), int(cursor[1])

        out = self._background(frame)
        mask, bbox = self._mask_and_bbox(h, w, cx, cy)
        if mask is None:
            return out

        x0, y0, x1, y1 = bbox
        fg = self._enhance(frame[y0:y1, x0:x1]).astype(np.float32)
        bg = out[y0:y1, x0:x1].astype(np.float32)
        m3 = mask[..., None]

        blended = fg * m3 + bg * (1.0 - m3)
        out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

        if self.ring and self.shape == "circle":
            cv2.circle(out, (cx, cy), self.radius, (255, 255, 255), 1, cv2.LINE_AA)

        return out

    # ------------------------------------------------------ live controls --
    def adjust(self, radius=0, feather=0, dim=0.0, blur=0, sharp=0.0):
        if radius:
            self.radius = int(np.clip(self.radius + radius, 40, 900))
            self._patch_key = None
        if feather:
            self.feather = int(np.clip(self.feather + feather, 5, 450))
            self._patch_key = None
        if dim:
            self.dim_gamma = float(np.clip(self.dim_gamma + dim, 0.05, 1.0))
        if blur:
            self.blur_ksize = int(np.clip(self.blur_ksize + blur, 3, 61))
        if sharp:
            self.sharp_amount = float(np.clip(self.sharp_amount + sharp, 0.0, 2.5))

    def apply_preset(self, level: int):
        for key, value in PRESETS[level].items():
            setattr(self, key, value)
        self._clahe = cv2.createCLAHE(clipLimit=self._clahe_clip,
                                      tileGridSize=(8, 8))
        self._patch_key = None

    def next_shape(self):
        self.shape = SHAPES[(SHAPES.index(self.shape) + 1) % len(SHAPES)]
        self._patch_key = None

    def status(self):
        return (f"r={self.radius} f={self.feather} dim={self.dim_gamma:.2f} "
                f"blur={self.blur_ksize} sharp={self.sharp_amount:.2f} "
                f"| {self.shape} | enhance={'ON' if self.enhance_on else 'OFF'}")