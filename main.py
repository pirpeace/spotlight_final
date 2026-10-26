"""
main.py
Spotlight for Teaching — Image Processing Project

รันแบบแนะนำ (ใช้จอเดียวได้ ไม่มีภาพซ้อน):
    python main.py --source snapshot

โหมดเว็บแคมตามที่เลือกไว้ในฟอร์ม:
    python main.py --source webcam

โหมดจับหน้าจอสด (ต้องมี 2 จอเท่านั้น):
    python main.py --source screen --monitor 1 --windowed

โหมด real-time บนจอเดียว (macOS เท่านั้น ไม่เกิดภาพซ้อน):
    python main.py --source live
    (ทดสอบหลักการก่อน: python live_mac.py --check)
"""

import sys
sys.dont_write_bytecode = True   # ไม่สร้าง __pycache__ (ต้องอยู่ก่อน import ไฟล์ในโปรเจค)

import argparse
import time

import cv2
import numpy as np

from capture import build_source
from spotlight import SpotlightRenderer

WINDOW = "Spotlight for Teaching"

HELP_LINES = [
    "[1/2/3] preset soft-mid-strong   [+/-] size   [ [ / ] ] feather",
    "[ , / . ] dim bg   [ ; / ' ] blur   [ k / l ] sharpen",
    "[c] re-capture screen   [b] shape   [e] enhance   [r] ring   [f] fullscreen",
    "[s] save png   [h] hide help   [q / ESC] quit",
]


def draw_text(img, text, org, scale=0.55, color=(255, 255, 255)):
    """วาดข้อความพร้อมเส้นขอบดำ ให้อ่านออกทุกพื้นหลัง"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def parse_args():
    ap = argparse.ArgumentParser(description="Spotlight for Teaching")
    ap.add_argument("--source", default="snapshot",
                    choices=["snapshot", "webcam", "video", "screen", "live"],
                    help="แหล่งภาพ (ค่าเริ่มต้น: snapshot)")
    ap.add_argument("--monitor", type=int, default=1, help="หมายเลขจอสำหรับ snapshot/screen")
    ap.add_argument("--delay", type=float, default=3.0, help="หน่วงเวลาก่อนถ่าย snapshot (วินาที)")
    ap.add_argument("--cam", type=int, default=0, help="index ของกล้องเว็บแคม")
    ap.add_argument("--path", default="", help="พาธไฟล์วิดีโอ")
    ap.add_argument("--preset", type=int, default=2, choices=[1, 2, 3],
                    help="ความแรงเริ่มต้น 1=อ่อน 2=กลาง 3=เข้ม")
    ap.add_argument("--windowed", action="store_true",
                    help="เปิดแบบหน้าต่าง ไม่เต็มจอ")
    ap.add_argument("--scale", type=float, default=0.6,
                    help="สัดส่วนย่อภาพก่อนประมวลผล (0.25-1.0) ยิ่งน้อยยิ่งลื่น")
    ap.add_argument("--display", type=int, default=0,
                    help="ย้ายหน้าต่างผลลัพธ์ไปจอหมายเลขนี้ (0 = อัตโนมัติ)")
    ap.add_argument("--live-scale", type=float, default=1.0,
                    help="โหมด live: สัดส่วนย่อภาพก่อนประมวลผล (0.25-1.0) ลดลงถ้าภาพกระตุก")
    ap.add_argument("--fps", type=float, default=30,
                    help="โหมด live: เพดาน FPS")
    ap.add_argument("--list-monitors", action="store_true",
                    help="แสดงรายการจอทั้งหมดแล้วจบการทำงาน")

    return ap.parse_args()


def main():
    args = parse_args()
    if args.list_monitors:
        from capture import ScreenSource
        print("จอที่ตรวจพบ:")
        ScreenSource.list_monitors()
        return

    if args.source == "live":
        try:
            from live_mac import run_live
        except Exception as err:
            print(f"[error] เปิดโหมด live ไม่ได้: {err}")
            print("        โหมดนี้ใช้ได้เฉพาะ macOS และต้อง: "
                  "pip install pyobjc-framework-Quartz pyobjc-framework-Cocoa")
            return
        run_live(args)
        return

    try:
        src = build_source(args)
    except Exception as err:
        print(f"[error] {err}")
        return

    sp = SpotlightRenderer()
    sp.apply_preset(args.preset)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    # --- ย้ายหน้าต่างไปจออื่นก่อนสั่งเต็มจอ (โหมด screen เท่านั้น) ---
    if args.source == "screen":
        target = None
        if args.display > 0:
            target = src.monitors[args.display]
        else:
            target = src.target_monitor(exclude=args.monitor)

        if target is not None:
            cv2.moveWindow(WINDOW, target["left"] + 40, target["top"] + 40)
            cv2.resizeWindow(WINDOW, target["width"] - 80, target["height"] - 120)
            print(f"[window] ย้ายหน้าต่างไปจอที่ ({target['left']}, {target['top']}) แล้ว")
            time.sleep(0.4)
        else:
            print("[window] ไม่พบจอที่สอง — เปิดแบบหน้าต่าง "
                  "กรุณาลากไปจอโปรเจคเตอร์แล้วกด f เพื่อเต็มจอ")
            args.windowed = True

    fullscreen = not args.windowed
    cv2.setWindowProperty(
        WINDOW, cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    def on_mouse(event, x, y, flags, param):
        src.mouse = (x, y)

    if src.uses_window_mouse:
        cv2.setMouseCallback(WINDOW, on_mouse)

    show_help = True
    fps, frames, t_start = 0.0, 0, time.time()

    while True:
        ok, frame = src.read()
        if not ok or frame is None:
            print("[warn] อ่านภาพไม่สำเร็จ — ออกจากโปรแกรม")
            break

        h, w = frame.shape[:2]
        out = sp.render(frame, src.cursor(w, h))

        frames += 1
        if frames >= 10:
            elapsed = time.time() - t_start
            fps = frames / elapsed if elapsed > 0 else 0.0
            frames, t_start = 0, time.time()

        draw_text(out, f"FPS {fps:5.1f} | {sp.status()}", (16, 30),
                  0.58, (0, 255, 200))

        if show_help:
            base = h - 22 * len(HELP_LINES) - 14
            for i, line in enumerate(HELP_LINES):
                draw_text(out, line, (16, base + i * 22), 0.5)

        cv2.imshow(WINDOW, out)
        key = cv2.waitKey(1) & 0xFF

        if key == 255:
            continue

        if key in (ord('q'), 27):
            break
        elif key in (ord('1'), ord('2'), ord('3')):
            sp.apply_preset(key - ord('0'))
        elif key in (ord('+'), ord('=')):
            sp.adjust(radius=+20)
        elif key in (ord('-'), ord('_')):
            sp.adjust(radius=-20)
        elif key == ord('['):
            sp.adjust(feather=-15)
        elif key == ord(']'):
            sp.adjust(feather=+15)
        elif key == ord(','):
            sp.adjust(dim=-0.05)
        elif key == ord('.'):
            sp.adjust(dim=+0.05)
        elif key == ord(';'):
            sp.adjust(blur=-2)
        elif key == ord("'"):
            sp.adjust(blur=+2)
        elif key == ord('k'):
            sp.adjust(sharp=-0.1)
        elif key == ord('l'):
            sp.adjust(sharp=+0.1)
        elif key == ord('b'):
            sp.next_shape()
        elif key == ord('e'):
            sp.enhance_on = not sp.enhance_on
        elif key == ord('r'):
            sp.ring = not sp.ring
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WINDOW, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord('s'):
            name = f"spotlight_{int(time.time())}.png"
            cv2.imwrite(name, out)
            print(f"[save] {name}")
        elif key == ord('c') and args.source == "snapshot":
            cv2.destroyWindow(WINDOW)
            for _ in range(5):
                cv2.waitKey(30)          # ให้ระบบปิดหน้าต่างจริงก่อน
            src.capture()
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                WINDOW, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(WINDOW, on_mouse)

    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
