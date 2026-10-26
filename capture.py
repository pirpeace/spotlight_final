"""
capture.py
แหล่งข้อมูลภาพสำหรับโปรเจค Spotlight for Teaching
  - SnapshotSource : จับภาพหน้าจอครั้งเดียว (ไม่เกิด infinite mirror) -- แนะนำ
  - WebcamSource   : ภาพจากกล้องเว็บแคมแบบ real-time
  - VideoSource    : ไฟล์วิดีโอ (ใช้ทดสอบซ้ำ ๆ ตอนทำรายงาน)
  - ScreenSource   : จับภาพหน้าจอต่อเนื่อง (ต้องใช้ 2 จอเท่านั้น)
"""

import sys
import time

import cv2
import numpy as np

IS_WINDOWS = sys.platform.startswith("win")


# --------------------------------------------------------------------------- #
#  BASE
# --------------------------------------------------------------------------- #
class BaseSource:
    """โครงร่างกลางของทุกแหล่งภาพ"""

    uses_window_mouse = True      # True = ใช้ตำแหน่งเมาส์บนหน้าต่าง OpenCV

    def __init__(self):
        self.mouse = None

    def read(self):
        raise NotImplementedError

    def cursor(self, w, h):
        if self.mouse is None:
            return w // 2, h // 2
        x, y = self.mouse
        return int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))

    def release(self):
        pass


# --------------------------------------------------------------------------- #
#  1) SNAPSHOT  (แนะนำ : ใช้จอเดียวได้ ไม่มีภาพซ้อน)
# --------------------------------------------------------------------------- #
class SnapshotSource(BaseSource):
    def __init__(self, monitor: int = 1, delay: float = 3.0):
        super().__init__()
        self.monitor = monitor
        self.delay = delay
        self.frame = None
        self.capture()

    def capture(self):
        import mss

        print(f"[snapshot] เตรียมหน้าจอที่ต้องการให้พร้อม จะถ่ายใน {self.delay:.0f} วินาที ...")
        for i in range(int(self.delay), 0, -1):
            print(f"           {i}")
            time.sleep(1)

        with mss.mss() as sct:
            mon = sct.monitors[self.monitor]
            shot = np.array(sct.grab(mon))

        frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
        h, w = frame.shape[:2]
        if w > 2200:
            scale = 2200 / w
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        self.frame = frame
        print(f"[snapshot] ถ่ายภาพเรียบร้อย ({frame.shape[1]}x{frame.shape[0]})")

    def read(self):
        return True, self.frame.copy()


# --------------------------------------------------------------------------- #
#  2) WEBCAM  (ตรงกับที่เลือกไว้ในฟอร์ม : Real-time)
# --------------------------------------------------------------------------- #
class WebcamSource(BaseSource):
    """ภาพจากกล้องเว็บแคม ใช้ตำแหน่งเมาส์บนหน้าต่างเป็นตัวชี้"""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        super().__init__()
        backend = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"เปิดกล้อง index={index} ไม่ได้ "
                f"— ลองเปลี่ยนเป็น --cam 1 หรือปิดแอปที่ใช้กล้องอยู่"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)          # mirror ให้เหมือนส่องกระจก
        return ok, frame

    def release(self):
        self.cap.release()


# --------------------------------------------------------------------------- #
#  3) VIDEO FILE
# --------------------------------------------------------------------------- #
class VideoSource(BaseSource):
    """เล่นไฟล์วิดีโอวนลูป"""

    def __init__(self, path: str):
        super().__init__()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"เปิดไฟล์วิดีโอไม่ได้: {path}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        self.cap.release()


# --------------------------------------------------------------------------- #
#  LIVE SCREEN  (โหมดสอนจริง : ต้องใช้ Extended Display)
# --------------------------------------------------------------------------- #
class ScreenSource(BaseSource):
    """
    จับภาพหน้าจอแบบต่อเนื่อง สำหรับใช้สอนจริงกับโปรเจคเตอร์

    หลักการ : จับจอ A (จอที่เปิดสไลด์) แล้วแสดงผลบนจอ B (โปรเจคเตอร์)
              ทำให้ไม่เกิด infinite mirror และสไลด์เลื่อนหน้าได้อิสระ

    รองรับจอ Retina : mss คืนค่า monitor เป็น "points" แต่ grab ได้ "pixels"
                      จึงคำนวณ scale จากขนาดภาพจริงเสมอ
    """

    uses_window_mouse = False       # ใช้ตำแหน่งเมาส์จริงของระบบทั้งเครื่อง

    def __init__(self, monitor: int = 1, scale: float = 0.6):
        super().__init__()
        import mss
        import pyautogui

        pyautogui.FAILSAFE = False
        self.pyautogui = pyautogui

        self.sct = mss.mss()
        self.monitors = self.sct.monitors
        if monitor >= len(self.monitors):
            raise RuntimeError(
                f"ไม่พบจอหมายเลข {monitor} — เครื่องนี้มี {len(self.monitors) - 1} จอ "
                f"(ลองสั่ง --list-monitors เพื่อดูรายการ)"
            )

        self.index = monitor
        self.mon = self.monitors[monitor]
        self.scale = float(np.clip(scale, 0.25, 1.0))

        print(f"[screen] จับจอ #{monitor} "
              f"({self.mon['width']}x{self.mon['height']} pt) "
              f"| ลดขนาดประมวลผล {self.scale:.2f}x")
        if len(self.monitors) <= 2:
            print("[screen] คำเตือน : ตรวจพบจอเดียว "
                  "หากแสดงผลบนจอเดียวกับที่จับภาพ จะเกิดภาพซ้อน (infinite mirror)")

    # ---- รายการจอทั้งหมด (ใช้ตอนตั้งค่า) ----
    @staticmethod
    def list_monitors():
        import mss
        with mss.mss() as sct:
            for i, m in enumerate(sct.monitors):
                tag = "ทุกจอรวมกัน" if i == 0 else f"จอที่ {i}"
                print(f"  [{i}] {tag:14s} "
                      f"{m['width']}x{m['height']}  @ ({m['left']}, {m['top']})")

    def read(self):
        shot = np.array(self.sct.grab(self.mon))
        frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
        if self.scale < 0.999:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_AREA)
        return True, frame

    def cursor(self, w, h):
        """แปลงพิกัดเมาส์ของระบบ -> พิกัดในเฟรม (รองรับ Retina อัตโนมัติ)"""
        try:
            mx, my = self.pyautogui.position()
        except Exception:
            return w // 2, h // 2

        rel_x = (mx - self.mon["left"]) / max(self.mon["width"], 1)
        rel_y = (my - self.mon["top"]) / max(self.mon["height"], 1)
        x = int(np.clip(rel_x * w, 0, w - 1))
        y = int(np.clip(rel_y * h, 0, h - 1))
        return x, y

    def target_monitor(self, exclude: int):
        """คืนพิกัดของจออื่นที่ไม่ใช่จอที่กำลังจับภาพ — ใช้ย้ายหน้าต่างไปแสดง"""
        for i, m in enumerate(self.monitors):
            if i in (0, exclude):
                continue
            return m
        return None

    def release(self):
        self.sct.close()

# --------------------------------------------------------------------------- #
#  FACTORY
# --------------------------------------------------------------------------- #
def build_source(args):
    """สร้าง source ตาม argument ที่ผู้ใช้เลือก"""
    if args.source == "snapshot":
        return SnapshotSource(args.monitor, args.delay)
    if args.source == "screen":
        return ScreenSource(args.monitor, getattr(args, "scale", 0.6))
    if args.source == "video":
        return VideoSource(args.path)
    return WebcamSource(args.cam)