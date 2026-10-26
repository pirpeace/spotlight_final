"""
live_mac.py
โหมด real-time บน macOS "จอเดียว" ไม่ต้องแคปภาพนิ่ง และไม่เกิดภาพซ้อน (infinite mirror)

หลักการ
  1) สร้างหน้าต่างเต็มจอ ไร้ขอบ ลอยเหนือทุกแอป และ "คลิกทะลุ" ได้ (ignoresMouseEvents)
     -> ผู้สอนคลิก/เลื่อนสไลด์ที่อยู่ข้างล่างได้ตามปกติ
  2) จับภาพหน้าจอเฉพาะ "หน้าต่างที่อยู่ใต้หน้าต่างของเรา"
     (CGWindowListCreateImage + kCGWindowListOptionOnScreenBelowWindow)
     -> ภาพที่จับได้จึงไม่มีหน้าต่าง Spotlight ปนเข้ามาเลย ภาพซ้อนเกิดไม่ได้
  3) ส่งภาพเข้า SpotlightRenderer เดิม แล้ววาดผลลัพธ์กลับเป็นหน้าต่างในข้อ 1
  4) ตำแหน่งเมาส์ใช้ NSEvent.mouseLocation() (ไม่ต้องขอสิทธิ์เพิ่ม)
  5) ปุ่มลัดใช้ CGEventTap (แบบ active) รันใน thread แยก
     -> กดปุ่มลัดแล้วโปรแกรมทำงานทันที และ "กลืน" ปุ่มนั้นไว้ ไม่ส่งต่อไปยังแอปที่โฟกัสอยู่
        (เหมือนโหมด snapshot ที่ปุ่มไปถึงหน้าต่าง OpenCV อย่างเดียว)
     -> ปุ่มอื่นที่ไม่ใช่ปุ่มลัด (ลูกศร, Space, Page Up/Down) และปุ่มที่กดพร้อม Cmd/Ctrl/Option
        ส่งต่อให้สไลด์ตามปกติ
     -> ต้องเปิดสิทธิ์ Accessibility ให้ Terminal/IDE

ติดตั้ง:  pip install pyobjc-framework-Quartz pyobjc-framework-Cocoa
ทดสอบหลักการก่อนใช้จริง:  python live_mac.py --check
"""

import sys
import time
import threading
import collections

import cv2
import numpy as np

if sys.platform != "darwin":
    raise RuntimeError("live_mac.py ใช้ได้เฉพาะ macOS")

import objc
import Quartz
from AppKit import NSApplication, NSColor, NSEvent, NSScreen, NSWindow
from Foundation import NSData, NSDate, NSDefaultRunLoopMode
from CoreFoundation import (CFMachPortCreateRunLoopSource, CFRunLoopAddSource,
                            CFRunLoopGetCurrent, CFRunLoopRun, CFRunLoopStop,
                            kCFRunLoopCommonModes)

# ค่าคงที่ของ AppKit (ใส่เป็นตัวเลขตรง ๆ กันชื่อ constant ต่างกันตามเวอร์ชัน PyObjC)
_STYLE_BORDERLESS = 0
_BACKING_BUFFERED = 2
_POLICY_ACCESSORY = 1                 # ไม่มีไอคอนบน Dock
_JOIN_ALL_SPACES = 1 << 0
_STATIONARY = 1 << 4
_IGNORES_CYCLE = 1 << 6
_FULLSCREEN_AUXILIARY = 1 << 8        # ให้ลอยเหนือแอปที่เปิดแบบเต็มจอ (เช่น Keynote)

# BGRX 32 บิต (ตรงกับลำดับ byte ที่ OpenCV ใช้: B,G,R,+ช่องว่าง)
_BITMAP_BGRX = int(Quartz.kCGBitmapByteOrder32Little) | int(Quartz.kCGImageAlphaNoneSkipFirst)

KEY_ESC = 53
_EVENT_MASK_ANY = 0xFFFFFFFFFFFFFFFF

# รหัสปุ่มจริงบนคีย์บอร์ด -> ตัวอักษร (ใช้ได้แม้คีย์บอร์ดเป็นภาษาไทย)
KEYCODE_TO_CHAR = {
    12: "q", 31: "o", 4: "h", 1: "s",
    18: "1", 19: "2", 20: "3",
    24: "=", 27: "-", 33: "[", 30: "]",
    43: ",", 47: ".", 41: ";", 39: "'",
    40: "k", 37: "l", 11: "b", 14: "e", 15: "r",
}


# --------------------------------------------------------------------------- #
#  OVERLAY WINDOW + CAPTURE
# --------------------------------------------------------------------------- #
class LiveOverlay:
    def __init__(self):
        self.app = NSApplication.sharedApplication()
        self.app.setActivationPolicy_(_POLICY_ACCESSORY)
        self.app.finishLaunching()

        # จอหลัก (จอที่มีเมนูบาร์) ให้ตรงกับ CGMainDisplayID
        self.screen = NSScreen.screens()[0]
        self.screen_frame = self.screen.frame()
        self.bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())

        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            self.screen_frame, _STYLE_BORDERLESS, _BACKING_BUFFERED, False)
        win.setReleasedWhenClosed_(False)
        win.setOpaque_(True)
        win.setBackgroundColor_(NSColor.blackColor())
        win.setHasShadow_(False)
        win.setIgnoresMouseEvents_(True)              # คลิกทะลุ
        win.setLevel_(Quartz.CGWindowLevelForKey(Quartz.kCGScreenSaverWindowLevelKey))
        win.setCollectionBehavior_(
            _JOIN_ALL_SPACES | _STATIONARY | _IGNORES_CYCLE | _FULLSCREEN_AUXILIARY)
        win.contentView().setWantsLayer_(True)
        self.layer = win.contentView().layer()
        self.layer.setContentsGravity_("resize")
        self.win = win

        self.colorspace = Quartz.CGColorSpaceCreateDeviceRGB()
        self.visible = False
        self.show_window()
        self.win_id = int(win.windowNumber())

        # ---- ปุ่มลัด : CGEventTap ใน thread แยก ----
        self._keys = collections.deque()          # deque ปลอดภัยเมื่อใช้ข้าม thread
        self._tap = None
        self._tap_loop = None
        self._tap_ready = threading.Event()
        self._tap_cb = self._on_tap               # เก็บ reference กัน callback ถูก GC
        self._tap_thread = threading.Thread(target=self._run_tap, daemon=True)
        self._tap_thread.start()
        self._tap_ready.wait(2.0)
        self.keys_ok = self._tap is not None

    # ---- หน้าต่าง ----
    def show_window(self):
        self.win.orderFrontRegardless()
        self.visible = True

    def hide_window(self):
        self.win.orderOut_(None)
        self.visible = False

    def close(self):
        if self._tap is not None:
            Quartz.CGEventTapEnable(self._tap, False)   # คืนคีย์บอร์ดให้ระบบทันที
            self._tap = None
        if self._tap_loop is not None:
            CFRunLoopStop(self._tap_loop)
            self._tap_loop = None
        self.win.orderOut_(None)
        self.win.close()

    def pump(self, seconds=0.001):
        """ดึงอีเวนต์จากคิวแล้วส่งต่อ (ต้องทำ ไม่งั้น global key monitor ไม่ถูกเรียก)"""
        until = NSDate.dateWithTimeIntervalSinceNow_(seconds)
        while True:
            ev = self.app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                _EVENT_MASK_ANY, until, NSDefaultRunLoopMode, True)
            if ev is None:
                break
            self.app.sendEvent_(ev)
            until = NSDate.distantPast()   # ดึงที่เหลือในคิวโดยไม่ต้องรอ

    # ---- ปุ่มลัด ----
    def _run_tap(self):
        """สร้าง event tap แล้ววน run loop ของตัวเอง (thread นี้รับปุ่มได้ทันที
        ไม่ต้องรอเฟรมประมวลผลภาพเสร็จ คีย์บอร์ดทั้งเครื่องจึงไม่หน่วง)"""
        with objc.autorelease_pool():
            tap = Quartz.CGEventTapCreate(
                Quartz.kCGSessionEventTap,
                Quartz.kCGHeadInsertEventTap,
                Quartz.kCGEventTapOptionDefault,          # active : กลืนปุ่มได้
                Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown),
                self._tap_cb,
                None)
            if tap is None:                               # ยังไม่ได้รับสิทธิ์ Accessibility
                self._tap_ready.set()
                return

            src = CFMachPortCreateRunLoopSource(None, tap, 0)
            self._tap_loop = CFRunLoopGetCurrent()
            CFRunLoopAddSource(self._tap_loop, src, kCFRunLoopCommonModes)
            Quartz.CGEventTapEnable(tap, True)
            self._tap = tap
            self._tap_ready.set()
        CFRunLoopRun()

    def _on_tap(self, proxy, etype, event, refcon):
        try:
            # macOS ปิด tap เองถ้าตอบช้า -> เปิดกลับ
            if etype in (Quartz.kCGEventTapDisabledByTimeout,
                         Quartz.kCGEventTapDisabledByUserInput):
                if self._tap is not None:
                    Quartz.CGEventTapEnable(self._tap, True)
                return event
            if etype != Quartz.kCGEventKeyDown:
                return event

            # กดพร้อม Cmd / Ctrl / Option -> ไม่ใช่ปุ่มลัด ปล่อยผ่าน (Cmd+Tab, Ctrl+C ฯลฯ)
            flags = Quartz.CGEventGetFlags(event)
            if flags & (Quartz.kCGEventFlagMaskCommand
                        | Quartz.kCGEventFlagMaskControl
                        | Quartz.kCGEventFlagMaskAlternate):
                return event

            code = int(Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventKeycode))
            if code == KEY_ESC:
                self._keys.append((code, "esc"))
                return None                               # กลืนปุ่ม
            ch = KEYCODE_TO_CHAR.get(code)
            if ch is None:
                return event                              # ไม่ใช่ปุ่มลัด ส่งต่อให้สไลด์
            self._keys.append((code, ch))
            return None                                   # กลืนปุ่ม ไม่ให้ไปพิมพ์ในแอปอื่น
        except Exception:
            return event

    def pop_keys(self):
        keys = []
        while self._keys:
            keys.append(self._keys.popleft())
        return keys

    # ---- จับภาพ "เฉพาะสิ่งที่อยู่ใต้หน้าต่างเรา" ----
    def capture(self):
        img = Quartz.CGWindowListCreateImage(
            self.bounds,
            Quartz.kCGWindowListOptionOnScreenBelowWindow,
            self.win_id,
            Quartz.kCGWindowImageNominalResolution | Quartz.kCGWindowImageBoundsIgnoreFraming)
        if img is None:
            return None

        w = int(Quartz.CGImageGetWidth(img))
        h = int(Quartz.CGImageGetHeight(img))
        if w == 0 or h == 0:
            return None
        bpr = int(Quartz.CGImageGetBytesPerRow(img))

        data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
        buf = np.frombuffer(bytes(data), dtype=np.uint8)
        # แต่ละแถวอาจมี padding ท้ายแถว (bytesPerRow > width*4)
        rows = buf[: bpr * h].reshape(h, bpr // 4, 4)[:, :w, :]
        return np.ascontiguousarray(rows[..., :3])      # BGRA -> BGR

    # ---- วาดผลลัพธ์ลงหน้าต่าง ----
    def show(self, bgr):
        h, w = bgr.shape[:2]
        bgrx = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        raw = bgrx.tobytes()
        provider = Quartz.CGDataProviderCreateWithCFData(
            NSData.dataWithBytes_length_(raw, len(raw)))
        cg = Quartz.CGImageCreate(
            w, h, 8, 32, w * 4, self.colorspace, _BITMAP_BGRX,
            provider, None, False, Quartz.kCGRenderingIntentDefault)

        Quartz.CATransaction.begin()
        Quartz.CATransaction.setDisableActions_(True)
        self.layer.setContents_(cg)
        Quartz.CATransaction.commit()

    # ---- เมาส์ ----
    def cursor(self, w, h):
        loc = NSEvent.mouseLocation()               # origin มุมซ้ายล่าง หน่วย point
        fx = (loc.x - self.screen_frame.origin.x) / max(self.screen_frame.size.width, 1)
        fy = 1.0 - (loc.y - self.screen_frame.origin.y) / max(self.screen_frame.size.height, 1)
        return (int(np.clip(fx * w, 0, w - 1)),
                int(np.clip(fy * h, 0, h - 1)))


# --------------------------------------------------------------------------- #
#  KEY MAP (ตรงกับ main.py)
# --------------------------------------------------------------------------- #
def apply_key(sp, ch):
    if ch in ("1", "2", "3"):
        sp.apply_preset(int(ch))
    elif ch in ("+", "="):
        sp.adjust(radius=+20)
    elif ch in ("-", "_"):
        sp.adjust(radius=-20)
    elif ch == "[":
        sp.adjust(feather=-15)
    elif ch == "]":
        sp.adjust(feather=+15)
    elif ch == ",":
        sp.adjust(dim=-0.05)
    elif ch == ".":
        sp.adjust(dim=+0.05)
    elif ch == ";":
        sp.adjust(blur=-2)
    elif ch == "'":
        sp.adjust(blur=+2)
    elif ch == "k":
        sp.adjust(sharp=-0.1)
    elif ch == "l":
        sp.adjust(sharp=+0.1)
    elif ch == "b":
        sp.next_shape()
    elif ch == "e":
        sp.enhance_on = not sp.enhance_on
    elif ch == "r":
        sp.ring = not sp.ring


HELP_LINES = [
    "[1/2/3] preset   [+/-] size   [ [ / ] ] feather   [ , / . ] dim",
    "[ ; / ' ] blur   [k/l] sharpen   [b] shape   [e] enhance   [r] ring",
    "[o] show/hide spotlight   [s] save png   [h] hide this   [q/ESC] quit",
]


def _draw_text(img, text, org, scale=0.5, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _check_permission():
    preflight = getattr(Quartz, "CGPreflightScreenCaptureAccess", None)
    request = getattr(Quartz, "CGRequestScreenCaptureAccess", None)
    if preflight is None or preflight():
        return True
    if request is not None:
        request()
    print("[permission] ยังไม่ได้รับสิทธิ์ Screen Recording")
    print("             System Settings > Privacy & Security > Screen & System Audio Recording")
    print("             เปิดสิทธิ์ให้ Terminal (หรือ IDE ที่ใช้รัน) แล้วปิด-เปิดโปรแกรมใหม่")
    return False


# --------------------------------------------------------------------------- #
#  MAIN LOOP
# --------------------------------------------------------------------------- #
def run_live(args):
    from spotlight import SpotlightRenderer

    if not _check_permission():
        return

    sp = SpotlightRenderer()
    sp.apply_preset(args.preset)
    scale = float(np.clip(getattr(args, "live_scale", 1.0), 0.25, 1.0))
    frame_time = 1.0 / max(float(getattr(args, "fps", 30)), 1.0)

    ov = LiveOverlay()
    show_hud = False
    warned = False
    running = True
    frames, t_fps = 0, time.time()
    last_out = None

    print("[live] เริ่มทำงาน — กด q หรือ ESC เพื่อออก (หรือ Ctrl+C ใน Terminal)")
    if ov.keys_ok:
        print("[live] ปุ่มลัดพร้อมใช้ (ปุ่มลัดจะไม่ถูกพิมพ์ลงแอปอื่น)")
    else:
        request = getattr(Quartz, "CGRequestPostEventAccess", None)
        if request is not None:
            request()                                     # ให้ macOS เด้งหน้าขอสิทธิ์
        print("[live] ปุ่มลัดใช้ไม่ได้: ยังไม่ได้รับสิทธิ์ Accessibility")
        print("       System Settings > Privacy & Security > Accessibility")
        print("       เปิดสิทธิ์ให้ Terminal / VS Code แล้วปิด-เปิดแอปใหม่ "
              "(ตอนนี้ออกด้วย Ctrl+C ใน Terminal)")

    try:
        while running:
            t0 = time.time()
            with objc.autorelease_pool():
                ov.pump(0.001)

                for code, ch in ov.pop_keys():
                    ch = ch.lower() if ch.isalpha() else ch
                    if code == KEY_ESC or ch == "q":
                        running = False
                    elif ch == "o":
                        if ov.visible:
                            ov.hide_window()
                        else:
                            ov.show_window()
                    elif ch == "h":
                        show_hud = not show_hud
                    elif ch == "s" and last_out is not None:
                        name = f"spotlight_{int(time.time())}.png"
                        cv2.imwrite(name, last_out)
                        print(f"[save] {name}")
                    else:
                        apply_key(sp, ch)

                if not running:
                    break
                if not ov.visible:
                    time.sleep(0.03)
                    continue

                frame = ov.capture()
                if frame is None:
                    if not warned:
                        print("[warn] จับภาพหน้าจอไม่ได้ (ตรวจสิทธิ์ Screen Recording)")
                        warned = True
                    time.sleep(0.1)
                    continue

                if scale < 0.999:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
                out = sp.render(frame, ov.cursor(w, h))

                if show_hud:
                    _draw_text(out, sp.status(), (16, 28), 0.55, (0, 255, 200))
                    base = h - 22 * len(HELP_LINES) - 14
                    for i, line in enumerate(HELP_LINES):
                        _draw_text(out, line, (16, base + i * 22))

                ov.show(out)
                last_out = out

            frames += 1
            if frames >= 30:
                now = time.time()
                print(f"[live] {frames / (now - t_fps):5.1f} FPS  ({w}x{h})")
                frames, t_fps = 0, now

            spare = frame_time - (time.time() - t0)
            if spare > 0:
                time.sleep(spare)
    except KeyboardInterrupt:
        pass
    finally:
        ov.close()
        print("[live] ปิดโปรแกรมแล้ว")


# --------------------------------------------------------------------------- #
#  SELF CHECK : ยืนยันว่าภาพที่จับไม่มีหน้าต่างของเราปนอยู่
# --------------------------------------------------------------------------- #
def self_check():
    if not _check_permission():
        return

    ov = LiveOverlay()
    try:
        w = int(ov.screen_frame.size.width)
        h = int(ov.screen_frame.size.height)
        red = np.zeros((h, w, 3), np.uint8)
        red[:] = (0, 0, 255)                          # แดงล้วน (BGR)
        ov.show(red)
        for _ in range(30):                           # ให้หน้าต่างขึ้นจอจริงก่อน
            ov.pump(0.02)

        frame = ov.capture()
        if frame is None:
            print("[check] ล้มเหลว: CGWindowListCreateImage คืนค่า None")
            print("        อาจเพราะยังไม่ได้เปิดสิทธิ์ Screen Recording หรือ macOS รุ่นนี้ไม่รองรับ API นี้")
            return

        cv2.imwrite("check_capture.png", frame)
        b, g, r = (frame[..., i].astype(np.int16) for i in range(3))
        red_ratio = float(np.mean((r > 200) & (g < 60) & (b < 60)))
        print(f"[check] ขนาดภาพที่จับได้ {frame.shape[1]}x{frame.shape[0]} "
              f"| สัดส่วนพิกเซลแดง {red_ratio:.1%}")
        if red_ratio > 0.5:
            print("[check] ไม่ผ่าน: ภาพที่จับมีหน้าต่างของเราปนอยู่ (จะเกิดภาพซ้อน)")
        else:
            print("[check] ผ่าน: ภาพที่จับไม่มีหน้าต่างของเรา (บันทึกไว้ที่ check_capture.png)")
    finally:
        ov.close()


if __name__ == "__main__":
    if "--check" in sys.argv:
        self_check()
    else:
        print("ใช้งาน:  python main.py --source live")
        print("ทดสอบ:   python live_mac.py --check")
