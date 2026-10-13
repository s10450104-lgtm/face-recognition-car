"""
self_test.py — 專案自我檢查
=============================
不需要攝影機，檢查環境與各階段程式是否能正常運作。

執行
----
    python self_test.py

檢查項目
--------
    1. Python / OpenCV / NumPy 版本與必要套件
    2. HAAR cascade 載入
    3. LBPH 辨識器建立（相容 OpenCV 4.x / 5.x）
    4. 攝影機可否開啟（不可用時只提示，不視為失敗）
    5. 資料集狀態
    6. 完整訓練 → 存檔 → 載入 → 辨識 流程
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# face_core 必須先於 cv2 匯入：它會設定 OPENCV_LOG_LEVEL 與主控台編碼
import face_core as fc
import cv2
import numpy
import numpy as np

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
_results: list[tuple[str, str, str]] = []


def check(label: str, ok: bool, detail: str = "", warn_only: bool = False):
    status = WARN if warn_only else (PASS if ok else FAIL)
    _results.append((status, label, detail))
    print(f"  [{status}] {label}" + (f"   {detail}" if detail else ""))
    return ok


def section(title: str):
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)


def synthetic_face(seed: int, size: int = 200) -> np.ndarray:
    """
    產生有紋理的合成人臉。

    注意：這只用來驗證「管線能不能跑」，不代表真實辨識率。
    真實辨識率必須用 collect_faces.py 收集的照片來測。
    """
    rng = np.random.default_rng(seed)
    img = rng.normal(150, 18, (size, size)).astype(np.float32)
    for _ in range(180):
        x, y = rng.integers(0, size, 2)
        cv2.circle(img, (int(x), int(y)), 1, float(rng.normal(150, 25)), -1)
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    img = np.clip(img, 0, 255).astype(np.uint8)

    cx, cy = size // 2, size // 2
    cv2.ellipse(img, (cx, cy + 35), (58, 42), 0, 0, 360, 100, -1)
    cv2.circle(img, (cx - 26, cy - 14), 10, 35, -1)
    cv2.circle(img, (cx + 26, cy - 14), 10, 35, -1)
    cv2.ellipse(img, (cx, cy + 4), (7, 17), 0, 0, 360, 45, -1)
    cv2.ellipse(img, (cx, cy + 40), (24, 10), 0, 0, 180, 40, -1)
    return fc.normalize_face(img)


def main() -> int:
    print()
    print("=" * 62)
    print("  人臉辨識系統自我檢查 (self_test.py)")
    print("=" * 62)

    # ── 1. 環境 ──
    section("1. 執行環境")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  OpenCV  : {cv2.__version__}")
    print(f"  NumPy   : {numpy.__version__}")
    print(f"  Python 路徑: {sys.executable}")
    print()

    check("OpenCV 版本 >= 4.5", tuple(map(int, cv2.__version__.split(".")[:2])) >= (4, 5),
          f"→ {cv2.__version__}")

    # ── 2. HAAR ──
    section("2. HAAR Cascade 人臉偵測")
    try:
        detector = fc.build_face_detector()
        check("載入 haarcascade_frontalface_alt.xml", not detector.empty())
        check("cascade 路徑存在",
              (Path(cv2.data.haarcascades) / fc.FACE_DETECTOR_PATH).exists(),
              f"→ {cv2.data.haarcascades}")

        blank = np.full((480, 640), 128, np.uint8)
        res = fc.detect_faces(blank, detector)
        check("空白影像回傳空結果（不崩潰）", res is not None and len(res) == 0)
    except Exception as e:
        check("HAAR 載入", False, f"→ {type(e).__name__}: {e}")

    # ── 3. LBPH ──
    section("3. LBPH 辨識器")
    try:
        rec = fc.build_recognizer()
        check("建立 LBPH recognizer", rec is not None, f"→ {type(rec).__name__}")
        check("LBPH 參數正確", fc.LBPH_PARAMS == dict(radius=1, neighbors=5, grid_x=8, grid_y=8),
              f"→ {fc.LBPH_PARAMS}")
    except Exception as e:
        check("建立 LBPH recognizer", False, f"→ {type(e).__name__}: {e}")
        rec = None

    # ── 4. 攝影機 ──
    section("4. 攝影機")
    try:
        from camera_source import open_camera
        cam = open_camera()
        ok, frame = cam.read()
        check("攝影機可開啟", cam is not None, f"→ {cam.name}", warn_only=True)
        if ok and frame is not None:
            check("可讀取影像幀", True, f"→ {frame.shape[1]}x{frame.shape[0]}")
        cam.release()
    except RuntimeError as e:
        check("攝影機可開啟", False, "→ 無法開啟（見下方說明）", warn_only=True)
        for line in str(e).split("\n"):
            print(f"        {line}")
    except Exception as e:
        check("攝影機可開啟", False, f"→ {type(e).__name__}: {e}", warn_only=True)

    # ── 5. 資料集 ──
    section("5. 訓練資料集")
    samples = fc.collect_samples(fc.DATA_DIR)
    if not samples:
        check("data/ 有訓練資料", False,
              "→ 目前沒有資料，請執行 python collect_faces.py", warn_only=True)
    else:
        total = sum(len(v) for v in samples.values())
        for label, items in sorted(samples.items()):
            flag = "OK" if len(items) >= fc.REQUIRED_SAMPLES else "數量不足"
            print(f"    {label:<16s} {len(items):>3d} 張   {flag}")
        check("每個人照片數達 30 張", fc.validate_dataset(samples)[0],
              f"→ 共 {len(samples)} 人 / {total} 張", warn_only=True)

    # ── 6. 完整流程 ──
    section("6. 完整流程（訓練 → 存檔 → 載入 → 辨識）")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        data_dir = tmp / "data"
        model_dir = tmp / "models"

        # 造兩個人各 30 張
        for person, base in (("student01", 1000), ("student02", 2000)):
            d = data_dir / person
            d.mkdir(parents=True, exist_ok=True)
            for i in range(1, 31):
                cv2.imwrite(str(d / f"{person}_{i:03d}.jpg"),
                            synthetic_face(base + i))

        tmp_samples = fc.collect_samples(data_dir)
        check("掃描資料集 → 2 人 × 30 張",
              len(tmp_samples) == 2 and all(len(v) == 30 for v in tmp_samples.values()),
              f"→ { {k: len(v) for k, v in tmp_samples.items()} }")

        try:
            trained, labels_arr, label_map = fc.train_lbph_model(
                tmp_samples, model_dir, verbose=False)
            check("LBPH 訓練完成", len(labels_arr) == 60, f"→ {len(labels_arr)} 個樣本")
            check("模型檔已輸出", (model_dir / "trainer.yml").exists()
                  and (model_dir / "trainer.json").exists())
            check("Label 對應正確", sorted(label_map) == ["student01", "student02"],
                  f"→ {label_map}")

            loaded, loaded_labels = fc.load_trained_model(model_dir)
            check("模型載入成功", sorted(loaded_labels) == sorted(label_map))

            # 用 student01 的樣本測試，應命中自己
            probe = cv2.imread(str(tmp_samples["student01"][0].path), cv2.IMREAD_GRAYSCALE)
            name, dist = fc.identify_face(probe, loaded, loaded_labels)
            check("辨識命中 student01", name == "student01", f"→ name={name}, dist={dist:.2f}")

            # 連續幀確認機制
            t = fc.TrackState()
            r1, _ = t.update("student01", 10.0)
            r2, _ = t.update("student01", 10.0)
            r3, _ = t.update("student01", 10.0)
            check("TrackState 前 2 幀不確認",
                  r1 == "Unknown" and r2 == "Unknown", f"→ {r1}, {r2}")
            check(f"第 {fc.CONFIRM_FRAMES} 幀確認",
                  r3 == "student01", f"→ {r3}")

            # 影像處理工具
            frame = np.full((480, 640, 3), 60, np.uint8)
            vis = fc.draw_result(frame.copy(), [{
                "rect": (200, 150, 120, 120), "name": "student01",
                "distance": 35.0, "is_known": True}])
            check("結果標註繪製", vis.shape == (480, 640, 3) and not np.array_equal(vis, frame))

            hud = fc.draw_hud(vis.copy(), "SELF TEST", "subtitle")
            check("HUD 繪製", hud.shape == (480, 640, 3))

        except Exception as e:
            check("完整流程", False, f"→ {type(e).__name__}: {e}")

    # ── 總結 ──
    section("總結")
    npass = sum(1 for s, _, _ in _results if s == PASS)
    nwarn = sum(1 for s, _, _ in _results if s == WARN)
    nfail = sum(1 for s, _, _ in _results if s == FAIL)

    print(f"  通過 {npass}   提示 {nwarn}   失敗 {nfail}")
    print()

    if nfail == 0:
        print("  [OK] 所有必要檢查通過，系統可正常使用。")
    else:
        print("  [ERROR] 有檢查未通過，請看上面的 [FAIL] 項目。")

    if nwarn:
        print()
        print("  有提示項目（非致命）：")
        for s, label, detail in _results:
            if s == WARN:
                print(f"    - {label}  {detail}")

    print()
    print("  下一步：")
    print("    python collect_faces.py       # 收集照片")
    print("    python train_lbph.py          # 訓練模型")
    print("    python recognize_lbph.py      # 即時辨識")
    print()

    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())