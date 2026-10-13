"""
camera_source.py — 鏡頭擷取抽象層
=================================
讓同一份程式在兩個環境都能跑：

    電腦 (測試階段) : cv2.VideoCapture(0)     ← 現在用的
    車子 (RPi 5)    : Picamera2               ← 之後移植用

程式只要呼叫 CameraSource.read() 就會拿到 (ok, frame)，
底下用哪一種完全不需要知道 —— 這是「先在電腦測，之後搬上車」的關鍵設計。

使用方法
--------
    from camera_source import open_camera

    cam = open_camera()          # 自動挑可用的來源
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        ...
    cam.release()
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional, Tuple

# 必須在 import cv2 之前設定：OpenCV 只在初始化時讀一次這個變數。
# 沒設的話，攝影機開不起來時 cv2 會噴一堆 FFMPEG 錯誤到 stderr，
# 蓋掉我們自己較清楚的錯誤訊息。
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np


def _silence_opencv_logs() -> None:
    """
    關閉 OpenCV 自己的錯誤輸出。

    cv2 內部在「攝影機開不起來」時會直接寫 stderr，會蓋掉我們自己
    較清楚的錯誤訊息。這裡把 OpenCV 的 log level 降到 SILENT。
    在模層級呼叫，確保早於任何 VideoCapture() 嘗試。
    """
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        # 舊版 OpenCV 沒有 utils.logging，忽略即可
        pass


_silence_opencv_logs()


class CameraSource:
    """
    統一包裝不同攝影機來源的介面。

    屬性
    ----
    name     : str，當前實際使用的來源名稱（顯示用）
    frame_size : (w, h)，解析度
    """

    def __init__(self, name: str, frame_size: Tuple[int, int] = (640, 480)):
        self.name = name
        self.frame_size = frame_size
        self._cap: Optional[cv2.VideoCapture] = None
        self._picam = None
        self._stream = None

    # ── 讀取 ──────────────────────────────────────────────
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """讀取一幀。回傳 (ok, frame)；frame 為 BGR 三通道 ndarray。"""
        if self._cap is not None:
            return self._cap.read()

        if self._picam is not None:
            frame = self._picam.capture_array()
            if frame is None:
                return False, None
            # Picamera2 預設回傳 RGB，OpenCV 要 BGR
            return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return False, None

    # ── 釋放 ──────────────────────────────────────────────
    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


# ------------------------------------------------------------
# 開啟邏輯
# ------------------------------------------------------------

def _silence_opencv_logs() -> None:
    """
    關閉 OpenCV 自己的錯誤輸出。

    cv2 內部在「攝影機開不起來」時會直接寫 stderr，會蓋掉我們自己
    較清楚的錯誤訊息。這裡把 OpenCV 的 log level 調到只留 FATAL。
    """
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        # 舊版 OpenCV 沒有 utils.logging，忽略即可
        pass


def _try_opencv(index: int = 0, frame_size: Tuple[int, int] = (640, 480),
                warmup: int = 10) -> Optional[CameraSource]:
    """嘗試用 cv2.VideoCapture 開攝影機（含 Windows 常見的後端）。"""
    _silence_opencv_logs()

    # Windows 上 DSHOW / MSMF 涵蓋所有常見內建與 USB 攝影機。
    # 刻意不放 CAP_ANY：它會退回 FFMPEG 後端，而該後端的錯誤訊息
    # 會繞過 setLogLevel 噴到 stderr，且它本來也開不了攝影機。
    if sys.platform.startswith("win"):
        backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]
    else:
        backends = [(cv2.CAP_ANY, "ANY")]

    for backend, bname in backends:
        try:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

            # 部分驅動需要丟掉前幾幀才會穩定
            ok = False
            for _ in range(warmup):
                ok, _ = cap.read()
                if not ok:
                    break

            if not ok:
                cap.release()
                continue

            cam = CameraSource(f"OpenCV/{bname} (index={index})", frame_size)
            cam._cap = cap
            return cam
        except cv2.error:
            continue

    return None


def _try_picamera2(frame_size: Tuple[int, int] = (640, 480)) -> Optional[CameraSource]:
    """嘗試用 Picamera2（樹莓派專用，未安裝會直接跳過）。"""
    try:
        from picamera2 import Picamera2
    except ImportError:
        return None

    try:
        picam = Picamera2()
        config = picam.create_preview_configuration(main={"size": frame_size})
        picam.configure(config)
        picam.start()

        cam = CameraSource("Picamera2 (Raspberry Pi)", frame_size)
        cam._picam = picam
        return cam
    except Exception:
        return None


def open_camera(prefer: str = "auto",
                index: int = 0,
                frame_size: Tuple[int, int] = (640, 480)) -> CameraSource:
    """
    開啟攝影機。

    參數
    ----
    prefer : "auto" | "opencv" | "picamera2"
        auto → 在車端優先 Picamera2，其餘用 OpenCV
        （車子移植時這裡不用改，auto 會自己挑）

    index  : int，OpenCV 的攝影機編號（外接 USB 攝影機常用 1）

    Raises
    ------
    RuntimeError : 所有來源都開不起來時
    """
    order = []
    if prefer == "picamera2":
        order = [_try_picamera2, lambda: _try_opencv(index, frame_size)]
    elif prefer == "opencv":
        order = [lambda: _try_opencv(index, frame_size), _try_picamera2]
    else:
        # 車子上優先 Picamera2，電腦上優先 OpenCV
        if _is_raspberry_pi():
            order = [_try_picamera2, lambda: _try_opencv(index, frame_size)]
        else:
            order = [lambda: _try_opencv(index, frame_size), _try_picamera2]

    for factory in order:
        cam = factory()
        if cam is not None:
            return cam

    raise RuntimeError(
        "無法開啟任何攝影機！\n"
        "可能原因：\n"
        "  1. 沒有攝影機，或攝影機被其他程式（Teams/Zoom/瀏覽器）佔用\n"
        "  2. Windows 隱私權設定阻擋：設定 > 隱私權與安全性 > 相機\n"
        "  3. 外接攝影機請試 --index 1"
    )


def _is_raspberry_pi() -> bool:
    """判斷是否在樹莓派上執行。"""
    try:
        with open("/proc/device-tree/model", "r", encoding="utf-8") as f:
            return "raspberry pi" in f.read().lower()
    except Exception:
        return False


if __name__ == "__main__":
    """自我測試：開啟攝影機並存一張測試圖到 logs/preview.jpg"""
    import os

    print("嘗試開啟攝影機 ...")
    try:
        cam = open_camera()
    except RuntimeError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    print(f"[OK] 使用來源：{cam.name}")

    for _ in range(10):
        ok, frame = cam.read()
        if not ok:
            break
        time.sleep(0.05)

    if ok and frame is not None:
        os.makedirs("logs", exist_ok=True)
        from face_core import imwrite_safe
        imwrite_safe("logs/preview.jpg", frame)
        print(f"[OK] 解析度：{frame.shape[1]}x{frame.shape[0]}")
        print(f"[OK] 已存測試圖 → logs/preview.jpg")
    else:
        print("[WARN] 讀不到影像")

    cam.release()