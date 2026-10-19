"""
yolo_detector.py — YOLOv8 物件偵測模組
======================================
本週主題「結合 Flask 與 YOLOv8」的偵測核心。

設計重點（依課程簡報〈TROUBLE SHOOTING〉的建議）
------------------------------------------------
1. **降低解析度**：使用 imgsz=320。在樹莓派 5 實測，
   640 → 約 5 FPS，320 → 約 11 FPS，差距近兩倍。
   物件偵測是「看大方向」用，320 綽綽有餘。

2. **跳幀處理（Skip Frames）**：YOLO 推論很慢，若每一幀都跑，
   影像串流會直接卡死。改成每 N 幀才推論一次，
   中間的畫幀沿用上一次的結果。

3. **非同步狀態更新**：推論結果存在物件屬性裡，
   前端另外用 API 抓取，不要把推論塞進串流的迴圈中。

4. **只偵測五類**：本作業只要求 PERSON / CAR / BUS /
   MOTORCYCLE / BICYCLE，其他類別丟掉，既省後處理又減少畫面雜訊。

效能實測（Raspberry Pi 5 / Debian 13 / torch 2.14.1+cpu）
----------------------------------------------------------
    imgsz=640   184.5 ms →  5.42 FPS
    imgsz=416    92.6 ms → 10.80 FPS
    imgsz=320    91.9 ms → 10.89 FPS   ← 建議採用

相依套件
--------
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install ultralytics

    ⚠️ 樹莓派沒有 NVIDIA GPU，務必用上面的 CPU 索引。
       直接 `pip install ultralytics` 會去抓 500MB+ 的 CUDA 套件而失敗。

使用
----
    from yolo_detector import YoloDetector

    det = YoloDetector("yolo_model.pt", imgsz=320, skip_frames=3)
    for frame in stream:
        det.tick(frame)                    # 每幀呼叫，只有跳幀時才真的推論
        out = det.draw(frame)              # 畫上偵測結果
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parent

# ────────────────────────────────────────────────────────────
#  常數
# ────────────────────────────────────────────────────────────

#: 本作業要求的五類物件（中文顯示名稱）
TARGET_CLASSES = {
    "person": "人",
    "car": "汽車",
    "bus": "公車",
    "motorcycle": "機車",
    "bicycle": "自行車",
}

#: 框線配色（BGR），對應 TARGET_CLASSES 的順序
CLASS_COLORS = [
    (0, 255, 255),    # person    黃
    (255, 180, 0),    # car       藍
    (0, 200, 255),    # bus       橘
    (0, 255, 100),    # motorcycle 綠
    (255, 0, 255),    # bicycle   紫紅
]

#: 預設模型路徑：優先找專案內的 yolo_model.pt
DEFAULT_MODEL_PATH = BASE_DIR / "yolo_model.pt"


def is_available() -> bool:
    """
    檢查 YOLO 環境是否可用。

    沒有安裝 ultralytics 時回傳 False，讓整個系統能降級運作
    （人臉辨識照常跑，只是沒有物件偵測）。
    """
    try:
        import ultralytics  # noqa: F401
        import torch        # noqa: F401
        return True
    except ImportError:
        return False


class YoloDetector:
    """
    YOLOv8 偵測器。

    參數
    ----
    model_path : str | Path
        模型檔位置；不存在時會嘗試從 ultralytics 自動下載。

    imgsz : int
        推論解析度。樹莓派上建議 320。

    conf : float
        信心分數門檻，低於此值不報。

    skip_frames : int
        每 N 幀才推論一次。1 = 每幀都跑（電腦上可用），
        樹莓派建議 2~3。

    屬性
    ----
    results  : list[dict]，最近一次推論結果
    infer_ms : float，最近一次推論耗時（毫秒）
    fps      : float，實際推論幀率
    enabled  : bool，模型是否成功載入
    """

    def __init__(self,
                 model_path: str | Path = DEFAULT_MODEL_PATH,
                 imgsz: int = 320,
                 conf: float = 0.30,
                 skip_frames: int = 3):
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.skip_frames = max(1, int(skip_frames))

        self._model = None
        self._class_ids: list[int] = []
        self._names: dict[int, str] = {}
        self._color_by_id: dict[int, tuple] = {}

        self._counter = 0          # 跳幀計數
        self._lock = threading.Lock()

        # ── 對外狀態 ──
        self.enabled = False
        self.load_error: str | None = None
        self.results: list[dict] = []
        self.infer_ms: float = 0.0
        self.fps: float = 0.0
        self._recent: list[float] = []

    # ── 載入 ──────────────────────────────────────────────

    def load(self) -> bool:
        """
        載入模型。成功回傳 True。

        載入失敗不會拋例外，只把錯誤記在 load_error，
        讓 Web 介面能顯示「YOLO 未啟用」而不是整個系統掛掉。
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            self.load_error = (
                "尚未安裝 ultralytics。"
                "樹莓派請用："
                "pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cpu "
                "&& pip install ultralytics"
            )
            print(f"[yolo] {self.load_error}（原始錯誤：{e}）")
            return False

        try:
            path = self.model_path
            if not path.exists():
                # 沒有就讓 ultralytics 自己抓官方權重
                print(f"[yolo] 找不到 {path}，改用 ultralytics 內建模型")
                path = "yolov8n.pt"

            self._model = YOLO(str(path))

            # 取出模型內的類別表，建立「只留五類」的過濾清單
            names = self._model.names
            self._names = dict(names)
            self._class_ids = [cid for cid, n in names.items()
                               if n in TARGET_CLASSES]

            for i, cid in enumerate(sorted(self._class_ids)):
                self._color_by_id[cid] = CLASS_COLORS[i % len(CLASS_COLORS)]

            self.enabled = True
            self.load_error = None
            print(f"[yolo] 模型載入完成：{path.name}")
            print(f"[yolo] imgsz={self.imgsz}  conf={self.conf}  "
                  f"跳幀={self.skip_frames}")
            print(f"[yolo] 五類過濾（COCO 編號）：{self._class_ids}")
            return True

        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            print(f"[yolo] 模型載入失敗：{self.load_error}")
            return False

    # ── 推論 ──────────────────────────────────────────────

    def tick(self, frame: np.ndarray) -> list[dict]:
        """
        丟一幀進來，依跳幀設定決定要不要真的推論。

        每幀都要呼叫，內部會自動判斷是否該推論。
        回傳目前的偵測結果（可能是這次新算的，也可能是沿用上次的）。

        參數
        ----
        frame : np.ndarray，BGR 三通道影像

        回傳
        ----
        list[dict]，每筆為 {"class","cn","conf","rect"}
        """
        if not self.enabled or frame is None:
            return []

        self._counter += 1
        if (self._counter - 1) % self.skip_frames != 0:
            # 這幀跳過，直接回傳上次的結果
            with self._lock:
                return list(self.results)

        try:
            t0 = time.perf_counter()
            res = self._model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.conf,
                classes=self._class_ids or None,
                verbose=False,
            )[0]
            elapsed = (time.perf_counter() - t0) * 1000

            detections = self._parse(res)

            with self._lock:
                self.results = detections
                self.infer_ms = elapsed
                self._recent.append(time.perf_counter())
                if len(self._recent) > 20:
                    self._recent.pop(0)
                if len(self._recent) >= 2:
                    span = self._recent[-1] - self._recent[0]
                    if span > 0:
                        self.fps = (len(self._recent) - 1) / span
                return list(detections)

        except Exception as e:
            print(f"[yolo] 推論失敗：{type(e).__name__}: {e}")
            return list(self.results)

    def _parse(self, res) -> list[dict]:
        """把 ultralytics 的結果物件轉成簡單的 dict 清單。"""
        out: list[dict] = []
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out

        for b in boxes:
            cid = int(b.cls[0])
            name = self._names.get(cid, str(cid))
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append({
                "class": name,
                "cn": TARGET_CLASSES.get(name, name),
                "conf": float(b.conf[0]),
                "rect": [int(x1), int(y1), int(x2), int(y2)],
            })
        return out

    # ── 繪圖 ──────────────────────────────────────────────

    def draw(self, frame: np.ndarray,
             font_scale: float = 0.45,
             thickness: int = 2) -> np.ndarray:
        """
        在畫面上畫出偵測結果。

        參數
        ----
        frame : np.ndarray

        回傳
        ----
        np.ndarray：畫上框線與標籤的影像（會複製，不改原圖）
        """
        import cv2

        if frame is None:
            return frame

        with self._lock:
            detections = list(self.results)

        canvas = frame.copy()

        for d in detections:
            x1, y1, x2, y2 = d["rect"]
            color = self._color_by_id.get(
                next((c for c, n in self._names.items() if n == d["class"]), -1),
                (0, 255, 255))

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

            # 標籤：「人 0.91」
            label = f"{d['cn']} {d['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            ty = y1 - 6 if y1 - th - 10 > 0 else y1 + th + 8
            cv2.rectangle(canvas, (x1, ty - th - 6),
                          (x1 + tw + 8, ty + 4), color, -1)
            cv2.putText(canvas, label, (x1 + 4, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (20, 20, 20), thickness, cv2.LINE_AA)

        return canvas

    # ── 狀態查詢 ──────────────────────────────────────────

    def status(self) -> dict:
        """回傳給 /api/ai_status 的狀態資料。"""
        with self._lock:
            return {
                "enabled": self.enabled,
                "loaded": self.enabled,
                "error": self.load_error,
                "imgsz": self.imgsz,
                "conf": self.conf,
                "skip_frames": self.skip_frames,
                "targets": TARGET_CLASSES,
                "class_ids": list(self._class_ids),
                "infer_ms": round(self.infer_ms, 1),
                "fps": round(self.fps, 2),
                "count": len(self.results),
                "detections": list(self.results),
            }

    @property
    def count(self) -> int:
        """最近一次推論偵測到的五類物件數量。"""
        with self._lock:
            return len(self.results)


# ────────────────────────────────────────────────────────────
#  命令列自我測試
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if not is_available():
        print("ultralytics 未安裝，無法測試。")
        print("樹莓派安裝指令：")
        print("  pip install torch torchvision "
              "--index-url https://download.pytorch.org/whl/cpu")
        print("  pip install ultralytics")
        sys.exit(1)

    print("YOLO 偵測模組自我測試")
    print("-" * 50)
    det = YoloDetector(imgsz=320, skip_frames=1)

    if not det.load():
        print(f"載入失敗：{det.load_error}")
        sys.exit(1)

    # 用攝影機抓一幀
    sys.path.insert(0, str(BASE_DIR))
    try:
        from camera_source import open_camera
        cam = open_camera()
        frame = None
        for _ in range(20):
            ok, f = cam.read()
            if ok:
                frame = f
                time.sleep(0.05)
        cam.release()
    except RuntimeError:
        frame = None

    if frame is None:
        print("攝影機開不起來，改用測試影像")
        frame = np.random.randint(60, 200, (480, 640, 3), dtype=np.uint8)

    print(f"\n測試影像：{frame.shape}")

    results = det.tick(frame)
    print(f"推論耗時：{det.infer_ms:.1f} ms")
    print(f"偵測結果：{len(results)} 個")

    for r in results:
        print(f"  {r['cn']:<4s} ({r['class']:<10s}) "
              f"信心 {r['conf']:.2f}  位置 {r['rect']}")

    out = det.draw(frame)
    cv2_path = str(BASE_DIR / "yolo_selftest.jpg")
    import cv2
    cv2.imwrite(cv2_path, out)
    print(f"\n已存測試結果：{cv2_path}")

    # 驗證跳幀確實有效
    print("\n跳幀行為驗證：")
    for skip in (1, 3, 5):
        det.skip_frames = skip
        det._counter = 0
        calls = [0]
        orig = det._model.predict

        def counting(*a, _orig=orig, _c=calls, **k):
            _c[0] += 1
            return _orig(*a, **k)

        det._model.predict = counting
        for _ in range(20):
            det.tick(frame)
        det._model.predict = orig
        print(f"  20 幀、跳幀={skip} → 實際推論 {calls[0]} 次")