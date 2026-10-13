"""
train_lbph.py — Step 2：訓練 LBPH 人臉辨識模型
=================================================
對應作業：自動讀取 data/ 下所有人臉資料，計算 LBPH 紋理特徵，
         輸出模型到 models/ 目錄

執行
----
    python train_lbph.py

輸出
----
    models/trainer.yml   ← LBPH 模型本體
    models/trainer.json  ← label 對照表（ID ↔ 姓名）

LBPH 原理（作業報告要寫的「資訊處理流程」）
-------------------------------------------
1. 人臉裁切成固定大小（200x200）
2. 直方圖均衡化 + CLAHE → 消除明暗影響
3. 切成 8x8 = 64 個格子
4. 每格算 3x3 鄰域的 LBP 碼：中心像素比鄰居大記 1、小記 0
   → 每格得到一個 0~255 的「紋理簽章」
5. 64 個簽章各累 加一個直方圖 → 串成 64*256 = 16384 維特徵向量
6. 把這些向量餵給 LBPH 訓練器 → 產出可存檔的模型

優點：訓練只要幾秒、不需 GPU、對光線變化有一定容忍度
"""

from __future__ import annotations

import sys
import time
from datetime import datetime

from face_core import (
    DATA_DIR,
    MODEL_DIR,
    REQUIRED_SAMPLES,
    collect_samples,
    train_lbph_model,
    validate_dataset,
)


def main() -> int:
    """命令列進入點。"""
    print("=" * 62)
    print("  LBPH 人臉辨識模型訓練")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print(f"  資料目錄：{DATA_DIR}")
    print(f"  輸出目錄：{MODEL_DIR}")
    print()

    # ── 步驟 0：檢查資料集 ──
    samples = collect_samples(DATA_DIR)

    if not samples:
        print("  [ERROR] data/ 目錄沒有任何照片！")
        print("  請先執行：python collect_faces.py")
        return 1

    total = sum(len(v) for v in samples.values())
    print(f"  找到 {len(samples)} 個人、{total} 張照片")
    for label, items in sorted(samples.items()):
        bar = "#" * min(40, len(items))
        print(f"    {label:<16s} {len(items):>3d} 張  {bar}")
    print()

    # ── 資料品質檢查 ──
    ok, warnings = validate_dataset(samples, REQUIRED_SAMPLES)
    if warnings:
        print("  資料檢查：")
        for w in warnings:
            print(w)
        if ok or all(len(v) >= 10 for v in samples.values()):
            print("  仍會繼續訓練，但辨識率可能受影響。")
        else:
            print("  [ERROR] 樣本過少，請先執行 collect_faces.py 收集照片")
            return 1
        print()

    # ── 訓練 ──
    t0 = time.time()
    try:
        recognizer, labels_array, label_map = train_lbph_model(samples, MODEL_DIR)
    except RuntimeError as e:
        print(f"  [ERROR] 訓練失敗：{e}")
        return 1

    elapsed = time.time() - t0

    print("=" * 62)
    print("  訓練完成！")
    print("=" * 62)
    print(f"  耗時      ：{elapsed:.2f} 秒")
    print(f"  模型大小  ：{(MODEL_DIR / 'trainer.yml').stat().st_size:,} bytes")
    print(f"  已註冊人員：{', '.join(label_map)}")
    print()
    print("  下一步：python recognize_lbph.py")
    print("=" * 62)

    return 0


if __name__ == "__main__":
    sys.exit(main())