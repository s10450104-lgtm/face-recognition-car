"""
collect_faces.py — Step 1：收集人臉訓練照片
=============================================
對應作業：收集 30 張人臉灰階照片到 data/<name>/ 目錄

執行
----
    python collect_faces.py                 # 互動輸入姓名
    python collect_faces.py student01       # 直接指定姓名
    python collect_faces.py student01 50    # 指定姓名與張數

操作
----
    對準鏡頭，按空白鍵拍照（也可設自動連拍）
    按 q 結束

設計重點
--------
* 每張照片都會再跑一次 HAAR 驗證「真的有臉」，沒臉就不存
* 存的是「裁切 + 直方圖均衡化」後的正規化人臉 → 訓練效果比原圖穩
* 自動擺位提示：改變頭部角度/表情，讓 LBPH 有足夠的紋理變化
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# face_core 必須先於 cv2 匯入：它會設定 OPENCV_LOG_LEVEL 與主控台編碼
import face_core
import cv2
import numpy as np

from camera_source import open_camera
from face_core import (
    DATA_DIR,
    REQUIRED_SAMPLES,
    build_face_detector,
    crop_face,
    detect_faces,
    imwrite_safe as fc_imwrite,
    normalize_face,
)

# 收集時的預處理輸出尺寸（訓練時會再統一縮成 LBPH 要的格式）
SAVE_SIZE = 200


def collect_faces(name: str, num_samples: int = REQUIRED_SAMPLES,
                  cam_index: int = 0) -> int:
    """
    收集指定人臉的訓練照片。

    回傳
    ----
    int : 實際成功儲存的張數
    """
    # ── 建立輸出目錄 ──
    out_dir = DATA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.jpg")))
    if existing:
        print(f"  [!] {name} 已有 {existing} 張，將繼續編號追加")

    detector = build_face_detector()
    cam = open_camera(index=cam_index)

    print("=" * 62)
    print(f"  收集人臉照片：{name}")
    print("=" * 62)
    print(f"  攝影機：{cam.name}")
    print(f"  目標張數：{num_samples} 張")
    print()
    print("  【操作方式】")
    print("    對準鏡頭讓臉在畫面中央 → 按 [空白] 拍照")
    print("    請稍微左右轉頭、改變表情、微笑 → 讓紋理特徵更豐富")
    print("    按 [q] 結束")
    print()

    saved = 0
    attempts = 0
    t0 = time.time()

    try:
        while saved < num_samples:
            ok, frame = cam.read()
            if not ok:
                print("  [!] 讀不到影像，嘗試重新連線...")
                cam.release()
                time.sleep(1)
                try:
                    cam = open_camera(index=cam_index)
                except RuntimeError as e:
                    print(f"  [FAIL] {e}")
                    return saved
                continue

            # 偵測用的灰階影像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, detector)

            # ── 畫面預覽 ──
            preview = frame.copy()
            h, w = preview.shape[:2]

            # 進度條
            progress = int((saved / num_samples) * (w - 20))
            cv2.rectangle(preview, (10, h - 30), (w - 10, h - 22), (60, 60, 60), -1)
            cv2.rectangle(preview, (10, h - 30), (10 + progress, h - 22), (0, 200, 0), -1)

            cv2.rectangle(preview, (0, 0), (w, 42), (25, 25, 25), -1)
            cv2.putText(preview, f"Collect: {name}", (12, 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            cv2.putText(preview, f"{saved}/{num_samples}",
                        (w - 150, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

            if len(faces) == 0:
                cv2.putText(preview, "No face detected - please face the camera",
                            (w // 2 - 215, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 165, 255), 2)
            else:
                for (x, y, fw, fh) in faces:
                    cv2.rectangle(preview, (x, y), (x + fw, y + fh), (0, 220, 0), 2)
                # 提示拍最大的一張臉
                bx, by, bw, bh = max(faces, key=lambda r: r[2] * r[3])
                cv2.putText(preview, "Press SPACE to capture",
                            (w // 2 - 105, h - 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 2)

            cv2.imshow("Collect Faces", preview)
            key = cv2.waitKey(1) & 0xFF

            # q → 結束
            if key == ord("q"):
                print(f"\n  結束收集，共 {saved} 張")
                break

            # 空白 → 拍照
            if key == ord(" ") or key == 32:
                attempts += 1

                if len(faces) == 0:
                    print(f"  [{attempts:>3d}] 沒偵測到人臉，請再試一次")
                    continue

                # 取畫面中最大的臉
                x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])

                if fw < 100:
                    print(f"  [{attempts:>3d}] 人臉太小({fw}px)，請靠近一點")
                    continue

                # 裁切 → 正規化 → 存檔
                roi = crop_face(gray, (x, y, fw, fh), out_size=SAVE_SIZE)
                roi = normalize_face(roi)

                idx = len(list(out_dir.glob("*.jpg"))) + 1
                path = out_dir / f"{name}_{idx:03d}.jpg"
                fc_imwrite(path, roi)

                saved += 1
                print(f"  [{attempts:>3d}] 已儲存 {path.name}  ({time.time()-t0:.0f}s)")

    except KeyboardInterrupt:
        print("\n  中斷")
    finally:
        cam.release()
        cv2.destroyAllWindows()

    return saved


def main() -> int:
    """命令列進入點。"""
    print()

    # ── 取得姓名 ──
    if len(sys.argv) > 1:
        name = sys.argv[1].strip()
    else:
        name = input("請輸入你的識別名稱（如 student01）：").strip()

    if not name:
        print("[ERROR] 姓名不可為空")
        return 1

    # 檔名安全檢查：擋掉 Windows 不允許的字元即可，中文是允許的
    # （寫檔走 face_core.imwrite_safe，支援 Unicode 路徑）
    bad = set('<>:"/\\|?*')
    if any(c in bad for c in name) or name.endswith((" ", ".")):
        print(f"[ERROR] 姓名不能包含 < > : \" / \\ | ? * 等字元：{name}")
        return 1
    if len(name) > 40:
        print(f"[ERROR] 姓名過長（最多 40 字元）：{name}")
        return 1

    # ── 取得張數 ──
    num = REQUIRED_SAMPLES
    if len(sys.argv) > 2:
        try:
            num = int(sys.argv[2])
        except ValueError:
            print(f"[WARN] 張數格式錯誤，改用預設 {num}")

    # ── 取得攝影機編號 ──
    cam_index = 0
    if "--index" in sys.argv:
        i = sys.argv.index("--index")
        if i + 1 < len(sys.argv):
            cam_index = int(sys.argv[i + 1])

    saved = collect_faces(name, num, cam_index)

    print()
    print("=" * 62)
    if saved >= REQUIRED_SAMPLES:
        print(f"  [OK] 完成！共 {saved} 張 → data/{name}/")
        print("  下一步：python train_lbph.py")
    else:
        print(f"  [WARN] 只收集到 {saved} 張（建議 {REQUIRED_SAMPLES} 張以上）")
        print(f"  執行 python collect_faces.py {name} 補足數量")
    print("=" * 62)

    return 0


if __name__ == "__main__":
    sys.exit(main())