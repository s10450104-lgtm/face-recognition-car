"""
recognize_lbph.py — Step 3：即時人臉辨識
=========================================
對應作業：啟動 Webcam 進行即時人臉偵測、特徵提取與閾值比對
         作業要求：請另一位同學進行辨識

執行
----
    python recognize_lbph.py                # 開內建鏡頭
    python recognize_lbph.py --index 1      # 外接 USB 攝影機
    python recognize_lbph.py --source file  # 用圖片/影片測試
    python recognize_lbph.py --threshold 55 # 調整嚴格程度

操作
----
    q / ESC  → 結束
    s        → 截圖存到 logs/
    t        → 顯示/切換統計資訊

辨識原理
--------
偵測到的每張臉，丟進已訓練的 LBPH 模型 → 回傳 (預測 ID, 距離 distance)

distance 是「紋理特徵的差異值」，越小越像：
    distance <= 60  → 綠框，判定為已註冊的人
    distance <= 85  → 黃框，可能是同一人（環境光較差）
    distance >  85  → 橘框，陌生人

（本機端用滑鼠拉大 / 縮小視窗來找最佳閾值值）
"""

from __future__ import annotations

import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# face_core 必須先於 cv2 匯入：它會設定 OPENCV_LOG_LEVEL 與主控台編碼
import face_core
import cv2
import numpy as np

from camera_source import open_camera
from face_core import (
    LOG_DIR,
    THRESHOLD_STRONG,
    THRESHOLD_WEAK,
    TrackState,
    build_face_detector,
    draw_hud,
    draw_result,
    identify_faces,
    imread_safe,
    imwrite_safe,
    load_trained_model,
)

# 人臉追蹤的分格大小（像素）。太小會因晃動而失效，太大會把兩張臉混在一起
TRACK_GRID = 60


def color_for(name: str, distance: float) -> tuple:
    """依辨識結果決定框的顏色（BGR）。"""
    from face_core import COLOR_KNOWN, COLOR_UNKNOWN

    if name == "Unknown":
        return COLOR_UNKNOWN
    if distance <= THRESHOLD_STRONG:
        return COLOR_KNOWN
    return (0, 215, 255)   # 黃色：疑似


def run_recognition(cam, detector, recognizer, labels: list[str],
                    source_label: str = "WEBCAM") -> int:
    """
    主辨識迴圈。

    回傳
    ----
    int : 本次執行辨識到的人臉次數統計（畫面右上角顯示）
    """
    tracks: dict[tuple, TrackState] = {}
    hit_counter: deque = deque(maxlen=120)   # 最近 120 幀的 FPS
    fps = 0.0
    show_stats = True
    frame_no = 0

    try:
        while True:
            t0 = time.time()
            ok, frame = cam.read()
            if not ok:
                print("  [!] 影像讀取中斷")
                break

            frame = cv2.flip(frame, 1)       # 鏡像，符合使用者直觉
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── 偵測 + 辨識 ──
            results = identify_faces(gray, detector, recognizer, labels)

            # ── 連續幀確認，避免閃爍 ──
            # 用「人臉中心的量化格子」當 key，而非 bbox：
            # bbox 會跟著呼吸/晃動每幀都變，當 key 等於永遠累積不到確認幀數。
            seen: set = set()
            for r in results:
                x, y, w, h = r["rect"]
                key = ((x + w // 2) // TRACK_GRID, (y + h // 2) // TRACK_GRID)
                seen.add(key)
                state = tracks.setdefault(key, TrackState())
                r["name"], r["distance"] = state.update(r["name"], r["distance"])

            # 清掉已經消失的人臉追蹤
            tracks = {k: v for k, v in tracks.items() if k in seen}

            # ── 繪製結果 ──
            draw_result(frame, results, show_distance=True)
            draw_hud(frame,
                     title="FACE RECOGNITION (LBPH)",
                     subtitle=source_label)

            # 顏色分級重繪
            for r in results:
                if r["distance"] <= THRESHOLD_WEAK and r["is_known"]:
                    x, y, w, h = r["rect"]
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  color_for(r["name"], r["distance"]), 3)

            # ── 統計資訊 ──
            frame_no += 1
            dt = time.time() - t0
            hit_counter.append(dt)
            if len(hit_counter) > 10:
                fps = len(hit_counter) / sum(hit_counter)

            known = sum(1 for r in results if r["is_known"])
            total = len(results)

            if show_stats:
                # 左下角資訊面板
                panel_h = 88
                cv2.rectangle(frame, (0, frame.shape[0] - panel_h),
                              (300, frame.shape[0]), (25, 25, 25), -1)
                lines = [
                    f"FPS: {fps:.1f}",
                    f"Faces: {total}   Known: {known}",
                    f"Registered: {len(labels)}",
                ]
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (10, frame.shape[0] - panel_h + 22 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF

            # ── 按鍵 ──
            if key in (ord("q"), 27):        # q / ESC
                print("\n  結束辨識")
                break
            elif key == ord("s"):            # 截圖
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = LOG_DIR / f"recognize_{ts}.jpg"
                imwrite_safe(path, frame)
                print(f"  [截圖] {path}")
            elif key == ord("t"):            # 切換統計
                show_stats = not show_stats

    except KeyboardInterrupt:
        print("\n  中斷")
    finally:
        cv2.destroyAllWindows()

    return frame_no


def run_on_file(path: Path, detector, recognizer, labels: list[str]) -> int:
    """用圖片或影片檔測試（方便驗證、也方便截圖交作業）。"""
    is_video = path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  [ERROR] 無法開啟 {path}")
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    frame_no = 0

    if is_video:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = identify_faces(gray, detector, recognizer, labels)
            draw_result(frame, results)
            draw_hud(frame, f"VIDEO: {path.name}", f"Frame {frame_no}")

            cv2.imshow("Face Recognition (file)", frame)
            if cv2.waitKey(25) & 0xFF in (ord("q"), 27):
                break
        cap.release()
    else:
        frame = imread_safe(path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"  [ERROR] 無法讀取圖片 {path}")
            cap.release()
            return 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = identify_faces(gray, detector, recognizer, labels)
        draw_result(frame, results)
        draw_hud(frame, f"IMAGE: {path.name}", "")

        out = LOG_DIR / f"result_{path.stem}.jpg"
        imwrite_safe(out, frame)
        print(f"  [OK] 結果已存：{out}")

        # 等使用者按鍵關閉
        cv2.imshow("Face Recognition (file)", frame)
        while cv2.waitKey(50) & 0xFF not in (ord("q"), 27, 13, 10):
            pass
        cap.release()

    cv2.destroyAllWindows()
    return frame_no


def main() -> int:
    """命令列進入點。"""
    args = sys.argv[1:]

    # ── 參數解析 ──
    cam_index = 0
    source = "camera"
    file_path: Path | None = None

    if "--index" in args:
        i = args.index("--index")
        if i + 1 < len(args):
            cam_index = int(args[i + 1])
    if "--source" in args:
        i = args.index("--source")
        if i + 1 < len(args):
            source = args[i + 1]
    if "--file" in args:
        i = args.index("--file")
        if i + 1 < len(args):
            file_path = Path(args[i + 1])

    print("=" * 62)
    print("  即時人臉辨識 (OpenCV + HAAR + LBPH)")
    print("=" * 62)

    # ── 載入模型 ──
    try:
        recognizer, labels = load_trained_model()
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return 1

    detector = build_face_detector()

    print(f"  已載入模型：{len(labels)} 人 → {', '.join(labels)}")
    print(f"  閾值設定：強={THRESHOLD_STRONG}  弱={THRESHOLD_WEAK}")
    print()

    # ── 檔案模式 ──
    if file_path:
        print(f"  模式：檔案測試 → {file_path}")
        run_on_file(file_path, detector, recognizer, labels)
        return 0

    # ── 鏡頭模式 ──
    print("  正在開啟攝影機 ...")
    try:
        cam = open_camera(index=cam_index)
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        return 1

    print(f"  攝影機：{cam.name}")
    print()
    print("  【操作】")
    print("    q / ESC → 結束")
    print("    s       → 截圖存到 logs/")
    print("    t       → 切換統計面板")
    print()
    print("  請換另一位同學站到鏡頭前進行辨識測試。")
    print("-" * 62)

    try:
        run_recognition(cam, detector, recognizer, labels,
                        source_label=cam.name.split("(")[0].strip())
    finally:
        cam.release()

    print("  Bye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())