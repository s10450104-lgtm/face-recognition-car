"""
face_core.py — 人臉辨識共用核心模組
=====================================
專案：從 Local 到 Web — 結合 Flask 與 YOLOv8 AI 自走車（作業一：人臉辨識）
底層：OpenCV + HAAR Cascade（偵測）+ LBPH（辨識）

本模組把「偵測 / 對齊 / 正規化 / 模型訓練 / 辨識比對」集中起來，
讓 collect_faces.py、train_lbph.py、recognize_lbph.py 共用同一套邏輯，
避免三支程式各自寫一份而參數不一致（作業要求：資訊處理流程清楚）。

對應到自走車的部分：
    本機 (PC) : cv2.VideoCapture(0)  ← 一般筆電內建/web 鏡頭
    車端 (RPi): Picamera2             ← 由 camera_source.py 自動偵測切換
    兩邊共用同一份辨識核心，測試成功後可直接搬上車。
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import sys

# ── 必須在 import cv2 之前設定 ──
# 1) OPENCV_LOG_LEVEL：攝影機開不起來時 cv2 會噴一堆 FFMPEG 錯誤到 stderr，
#    蓋掉我們自己較清楚的錯誤訊息。OpenCV 只在初始化時讀一次，所以要提早設。
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

# 2) Windows 主控台中文亂碼：PowerShell/cmd 預設 Big5(cp950)，
#    Python 輸出 UTF-8 會變亂碼。強制 console 與輸出流都用 UTF-8。
if os.name == "nt":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass

for _stream in ("stdout", "stderr"):
    _s = getattr(sys, _stream, None)
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import cv2
import numpy as np

# OpenCV 匯入後再降一次 log level（部分版本的 VideoCapture 建構時會重置）
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass


# ============================================================
# 全域設定（集中管理，方便調參）
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"          # 人臉訓練照片
MODEL_DIR = BASE_DIR / "models"       # LBPH 模型輸出
LOG_DIR = BASE_DIR / "logs"           # 執行紀錄

# HAAR Cascade：使用 OpenCV 內建檔案（隨套件安裝，不需額外下載）
FACE_DETECTOR_PATH = "haarcascade_frontalface_alt.xml"

# LBPH 參數：這 4 個值決定「紋理特徵向量」怎麼抽
#   radius / neighbors / grid_x / grid_y
#   調高 radius、neighbors 會讓特徵更細緻但更慢（樹莓派上要注意）
LBPH_PARAMS = dict(radius=1, neighbors=5, grid_x=8, grid_y=8)

# 訓練照片設定
REQUIRED_SAMPLES = 30          # 作業要求：每人 30 張
MIN_FACE_SIZE = 100            # 人臉框最小邊長，過濾掉遠距離的雜訊臉

# 辨識閾值（LBPH 回傳的是「距離」，越小越像）
THRESHOLD_STRONG = 60.0        # 低於此值 → 判定為同一人
THRESHOLD_WEAK = 85.0          # 高於此值 → 判定為陌生人

# 連續幾幀都認到才更新畫面顯示，避免畫面閃爍
CONFIRM_FRAMES = 3


# ============================================================
# 資料結構
# ============================================================

@dataclass
class FaceSample:
    """單一張人臉訓練樣本。"""
    label: str
    index: int
    path: Path
    face_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def name(self) -> str:
        return f"{self.label}_{self.index:03d}.jpg"


@dataclass
class TrackState:
    """
    連續幀確認機制，避免單幀誤判造成畫面閃爍。

    運作方式：
      pending_label  暫存「這一幀辨識到誰」
      pending_hits   連續幾幀都是同一個候選
      current_label  「已確認」要顯示給人看的結果

      只有當 pending_label 連續出現 CONFIRM_FRAMES 幀，
      才會把它升級成 current_label；中途換人則重新計數。
    """
    current_label: str = "Unknown"
    current_score: float = 0.0
    pending_label: str = ""
    pending_hits: int = 0
    last_update: float = field(default_factory=time.time)

    def update(self, label: str, score: float) -> Tuple[str, float]:
        """
        餵入這一幀的辨識結果。

        回傳
        ----
        (confirmed_label, score) : 已確認的人名 + 這一幀的距離
        """
        if label == self.pending_label:
            self.pending_hits += 1
        else:
            self.pending_label = label
            self.pending_hits = 1

        # 連續達標 → 正式確認
        if self.pending_hits >= CONFIRM_FRAMES:
            self.current_label = self.pending_label

        self.current_score = score
        self.last_update = time.time()
        return self.current_label, score

    def reset(self) -> None:
        """人臉消失時清空計數。"""
        self.pending_label = ""
        self.pending_hits = 0


# ============================================================
# 0. 檔案 I/O（處理 Windows 中文路徑）
# ============================================================

def imwrite_safe(path, image) -> bool:
    """
    存圖，但支援 Windows 上的中文／非 ASCII 路徑。

    問題：cv2.imwrite() 在 Windows 走的是 ANSI 路徑 API，
          路徑含中文時會直接回傳 False 且不寫檔、也不丟例外
          （實測 `cv2.imwrite('中文測試.jpg', img)` → False）。
          訓練資料夾用中文命名時就會整批存不下來。

    解法：改用 imencode() 在記憶體中編碼，再用 numpy.tofile()
          走寬字元路徑寫檔。

    參數
    ----
    path  : str | Path，目標路徑
    image : np.ndarray，影像

    回傳
    ----
    bool : 是否寫入成功
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 副檔名決定編碼器；沒有副檔名時預設 jpg
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False

    buf.tofile(str(path))     # numpy.tofile 走寬字元 API，中文 OK
    return True


def imread_safe(path, flags=cv2.IMREAD_GRAYSCALE):
    """
    讀圖，支援 Windows 上的中文／非 ASCII 路徑（cv2.imread 的對應問題）。

    cv2.imread() 遇到中文路徑同樣會回傳 None，且不丟例外。
    """
    path = Path(path)
    if not path.exists():
        return None

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None

    if data.size == 0:
        return None

    return cv2.imdecode(data, flags)


# ============================================================
# 1. HAAR Cascade 人臉偵測
# ============================================================

def build_face_detector(cascade_path: Optional[str] = None) -> cv2.CascadeClassifier:
    """
    建立 HAAR 人臉偵測器。

    參數
    ----
    cascade_path : str | None
        指定 cascade XML 路徑；None 則使用 OpenCV 內建的
        haarcascade_frontalface_alt.xml。

    回傳
    ----
    cv2.CascadeClassifier

    Raises
    ------
    FileNotFoundError : 找不到 cascade 檔案時
    """
    path = Path(cascade_path) if cascade_path else Path(cv2.data.haarcascades) / FACE_DETECTOR_PATH

    if not path.exists():
        raise FileNotFoundError(f"找不到 HAAR cascade 檔案：{path}")

    detector = cv2.CascadeClassifier(str(path))
    if detector.empty():
        raise RuntimeError(f"HAAR cascade 載入失敗（檔案損毀？）：{path}")

    return detector


def detect_faces(gray: np.ndarray, detector: cv2.CascadeClassifier,
                  scale_factor: float = 1.1, min_neighbors: int = 5,
                  min_size: int = MIN_FACE_SIZE) -> np.ndarray:
    """
    在灰階影像中偵測人臉。

    HAAR 的三個關鍵參數（作業要求的「參數處理流程」重點）：
      scale_factor = 1.1 : 影像每次縮小的比例。越小越精確但越慢
                           （樹莓派上可放寬到 1.3 提速）
      min_neighbors = 5  : 周圍鄰居數量門檻。越大越嚴格、誤判越少
      min_size           : 最小人臉邊長，過濾遠距離雜訊

    回傳
    ----
    np.ndarray : shape (N, 4)，每列為 [x, y, w, h]
    """
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
    )

    # detectMultiScale 在新版 OpenCV 回傳空 tuple（非空時是 ndarray）
    if faces is None or len(faces) == 0:
        return np.empty((0, 4), dtype=np.int32)

    return np.asarray(faces, dtype=np.int32)


# ============================================================
# 2. 人臉裁切與正規化
# ============================================================

def crop_face(image: np.ndarray, rect: Tuple[int, int, int, int],
              padding: float = 0.25, out_size: int = 200) -> np.ndarray:
    """
    從原影像裁出人臉並放大成固定尺寸。

    參數
    ----
    padding : float
        往外擴張的比例，讓臉部不只包含五官而是有點上下文，
        對 LBPH 紋ure特徵的穩定性有幫助。

    out_size : int
        輸出尺寸（LBPH 內部會再轉 100x100，這裡先統一較大的輸入）。
    """
    x, y, w, h = rect
    pad = int(w * padding)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return np.empty((0, 0), dtype=np.uint8)

    return cv2.resize(roi, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def normalize_face(gray_face: np.ndarray) -> np.ndarray:
    """
    LBPH 訓練用的標準化處理。

    步驟：灰階 → 直方圖均衡化（抵消 lighting 差異）
         → CLAHE（進一步處理側光/逆光）

    這是「讓同一個人在明暗不同的照片裡仍能被辨識」的關鍵步驟。
    """
    if gray_face.size == 0:
        return gray_face

    eq = cv2.equalizeHist(gray_face)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(eq)


# ============================================================
# 3. 資料集管理（data/ 目錄掃描與檢查）
# ============================================================

def label_from_filename(filename: str) -> str:
    """
    從檔名解析 label。

    規則：student01_003.jpg → student01
          001_王大明.jpg   → 001
          student01.jpg    → student01
    """
    stem = Path(filename).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def collect_samples(data_dir: Path = DATA_DIR) -> dict[str, list[FaceSample]]:
    """
    掃描 data/ 目錄，依 label 分組所有訓練樣本。

    支援的目錄結構：
        data/student01/student01_001.jpg
        data/student01/001.jpg          ← 較舊寫法也吃
        data/student01_001.jpg          ← 扁平寫法也吃

    回傳
    ----
    dict[str, list[FaceSample]] : label → 樣本清單（依 index 排序）
    """
    if not data_dir.exists():
        return {}

    grouped: dict[str, list[FaceSample]] = {}

    # 情況 A：data/<label>/*.jpg
    for sub_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = sub_dir.name
        files = sorted(f for f in sub_dir.glob("*.jpg"))
        for f in files:
            idx = _extract_index(f.stem, len(grouped.get(label, [])))
            grouped.setdefault(label, []).append(FaceSample(label, idx, f))

    # 情況 B：data/<label>_<idx>.jpg（扁平）
    for f in sorted(data_dir.glob("*.jpg")):
        label = label_from_filename(f.name)
        idx = _extract_index(f.stem, len(grouped.get(label, [])))
        grouped.setdefault(label, []).append(FaceSample(label, idx, f))

    for label in grouped:
        grouped[label].sort(key=lambda s: s.index)
    return grouped


def _extract_index(stem: str, fallback: int) -> int:
    """嘗試從檔名尾端解析數字序號，失敗則用現有筆數。"""
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else fallback


def validate_dataset(samples: dict[str, list[FaceSample]],
                     required: int = REQUIRED_SAMPLES) -> Tuple[bool, list[str]]:
    """
    檢查資料集是否符合訓練要求。

    回傳
    ----
    (ok, warnings)
        ok       : bool，每個 label 是否都達到數量門檻
        warnings : list[str]，不足的人員說明
    """
    warnings: list[str] = []
    for label, items in sorted(samples.items()):
        if len(items) < required:
            warnings.append(f"  [!] {label}: 只有 {len(items)} 張（建議 {required} 張以上）")

    return (len(warnings) == 0), warnings


# ============================================================
# 4. LBPH 模型訓練
# ============================================================

def _lbph_factory():
    """
    取得 LBPH 辨識器的建立函式。

    OpenCV 4.x  : cv2.FaceRecognizerLBPH_create()
    OpenCV 5.x  : 搬到 cv2.face 底下 → cv2.face.LBPHFaceRecognizer_create()

    車子(RPi)通常是 4.x、電腦可能是 5.x，兩邊都要能跑。
    """
    if hasattr(cv2, "FaceRecognizerLBPH_create"):
        return cv2.FaceRecognizerLBPH_create
    if hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        return cv2.face.LBPHFaceRecognizer_create
    raise RuntimeError("此 OpenCV 版本不支援 LBPH，請安裝 opencv-contrib-python")


def build_recognizer():
    """
    建立 LBPH 辨識器（已套用 LBPH_PARAMS 的空模型）。

    回傳
    ----
    LBPH 辨識器物件 : 已套用 LBPH_PARAMS 的空模型

    說明
    ----
    LBPH = Local Binary Patterns Histograms（局部二元模式直方圖）
    原理：把臉部切成 8x8 的格子，每格算「中心像素與鄰居的大小關係」
          （比鄰居大記 1、小記 0），串成一串 0/1 數列當作該格的紋理簽章，
          再把 64 格簽章合成一個直方圖向量，就是這張臉的「特徵向量」。
    優點：不吃光線變化、比純像素比對更穩、訓練超快（幾秒）
    缺點：轉頭或戴眼鏡效果下降；非 1:1 比對時用閾值控制
    """
    return _lbph_factory()(**LBPH_PARAMS)


def train_lbph_model(samples: dict[str, list[FaceSample]],
                     model_path: Path = MODEL_DIR,
                     verbose: bool = True,
                     re_detect: bool = False):
    """
    訓練 LBPH 模型。

    流程（作業要求的「資訊處理流程」）：
        讀取照片 → 灰階 → 正規化（均衡化 + CLAHE）
        → update() 累積特徵向量 → write() 輸出模型

    參數
    ----
    re_detect : bool
        False（預設）：直接使用照片的完整內容。
            這是 collect_faces.py 存檔的格式（已裁切的人臉），
            也是辨識時送進模型的格式 —— 兩者必須一致。
        True：照片若是未裁切的原圖，會先跑 HAAR 偵測再裁切人臉。

    回傳
    ----
    (recognizer, labels, processed) : 訓練好的模型、label 陣列、成功處理數
    """
    if not samples:
        raise RuntimeError("找不到任何訓練資料！請先執行 collect_faces.py 收集照片。")

    detector = build_face_detector()
    recognizer = build_recognizer()

    all_faces: list[np.ndarray] = []
    all_labels: list[int] = []
    label_map: list[str] = []
    label_to_id: dict[str, int] = {}

    if verbose:
        print("=" * 62)
        print("  步驟 1/3：讀取照片並偵測人臉")
        print("=" * 62)

    for label in sorted(samples):
        label_id = label_to_id.setdefault(label, len(label_to_id))
        label_map.append(label)

        ok_count = 0
        for sample in samples[label]:
            img = imread_safe(sample.path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                if verbose:
                    print(f"  [!] 無法讀取：{sample.path.name}")
                continue

            # ── 關鍵：裁切一致性 ──
            # collect_faces.py 存的已經是「HAAR 框 + padding 後」的標準人臉，
            # 辨識時 identify_faces() 做的也是同一件事。
            # 所以訓練時直接用整張圖即可；若再跑一次 HAAR 重新裁切，
            # HAAR 的框通常只涵蓋眼睛到下巴，會切掉已存的臉部邊緣，
            # 造成「訓練輸入」與「辨識輸入」不一致 → 距離被灌大、
            # 明明是同一人卻被判成 Unknown。
            #
            # re_detect=True 保留給「原始照片未裁切」的情況。
            if re_detect:
                faces = detect_faces(img, detector)
                if len(faces):
                    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                    roi = crop_face(img, (x, y, w, h))
                else:
                    roi = img
            else:
                roi = img

            roi = normalize_face(roi)

            if roi.size == 0:
                continue

            recognizer.update([roi], np.array([label_id], dtype=np.int32))
            all_faces.append(roi)
            all_labels.append(label_id)
            ok_count += 1

        if verbose:
            status = "OK" if ok_count >= REQUIRED_SAMPLES else f"（建議 {REQUIRED_SAMPLES} 張）"
            print(f"  [+] {label:<16s} ID={label_id}  成功 {ok_count:>3d} 張 {status}")

    if not all_faces:
        raise RuntimeError("訓練失敗：所有照片都偵測不到人臉，請檢查照片品質。")

    label_array = np.array(all_labels, dtype=np.int32)

    if verbose:
        print()
        print("=" * 62)
        print("  步驟 2/3：模型資訊")
        print("=" * 62)
        print(f"  訓練樣本總數 : {len(all_faces)}")
        print(f"  類別數       : {len(label_map)}  ({', '.join(label_map)})")
        print(f"  LBPH 參數    : {LBPH_PARAMS}")
        print()

    # ── 步驟 3：輸出模型檔 ──
    model_path.mkdir(parents=True, exist_ok=True)
    yml_path = model_path / "trainer.yml"
    json_path = model_path / "trainer.json"

    recognizer.write(str(yml_path))

    # 額外存一份 label 對照表（recognize 時要用）
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": label_map,
                "label_to_id": label_to_id,
                "params": LBPH_PARAMS,
                "threshold_strong": THRESHOLD_STRONG,
                "threshold_weak": THRESHOLD_WEAK,
                "sample_count": len(all_faces),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if verbose:
        print("=" * 62)
        print("  步驟 3/3：儲存模型")
        print("=" * 62)
        print(f"  [+] 模型     : {yml_path.name}")
        print(f"  [+] Label 表 : {json_path.name}")
        print()

    return recognizer, label_array, label_map


def load_trained_model(model_dir: Path = MODEL_DIR):
    """
    載入已訓練的模型與 label 對照表。

    Raises
    ------
    FileNotFoundError : 模型不存在（提醒先跑 train_lbph.py）
    """
    yml_path = model_dir / "trainer.yml"
    json_path = model_dir / "trainer.json"

    if not yml_path.exists():
        raise FileNotFoundError(f"找不到模型 {yml_path}，請先執行：python train_lbph.py")

    recognizer = build_recognizer()
    recognizer.read(str(yml_path))

    labels = []
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            labels = json.load(f).get("labels", [])

    return recognizer, labels


# ============================================================
# 5. 辨識（單張影像 / 即時串流共用）
# ============================================================

def identify_face(gray_face: np.ndarray,
                  recognizer,
                  labels: list[str]) -> Tuple[str, float]:
    """
    對單一人臉做辨識比對。

    回傳
    ----
    (label, distance)
        label    : 辨識出的名字，失敗回 "Unknown"
        distance : LBPH 距離。越小越像
    """
    roi = normalize_face(cv2.resize(gray_face, (200, 200)))

    try:
        pred_id, distance = recognizer.predict(roi)
        distance = float(distance)
    except cv2.error:
        # 模型尚未訓練，或訓練時沒有任何 sample（sample 數為 0 會丟例外）
        return "Unknown", float("inf")

    # 沒有 label 對照表就無法把 ID 轉回名字
    if not labels:
        return "Unknown", distance

    # 超過弱閾值一律視為陌生人
    if distance > THRESHOLD_WEAK or not np.isfinite(distance):
        return "Unknown", distance

    pred_id = int(pred_id)
    name = labels[pred_id] if 0 <= pred_id < len(labels) else "Unknown"
    return name, distance


def identify_faces(gray: np.ndarray,
                   detector: cv2.CascadeClassifier,
                   recognizer,
                   labels: list[str]) -> list[dict]:
    """
    在整張灰階影像上偵測 + 辨識所有人臉。

    回傳
    ----
    list[dict] : 每個元素含 rect / name / distance / is_known
    """
    results: list[dict] = []
    for (x, y, w, h) in detect_faces(gray, detector):
        roi = normalize_face(cv2.resize(crop_face(gray, (x, y, w, h)), (200, 200)))

        try:
            pred_id, distance = recognizer.predict(roi)
            distance = float(distance)
        except cv2.error:
            pred_id, distance = -1, float("inf")

        if distance <= THRESHOLD_WEAK and 0 <= int(pred_id) < len(labels):
            name = labels[int(pred_id)]
        else:
            name = "Unknown"

        results.append({
            "rect": (x, y, w, h),
            "name": name,
            "distance": distance,
            "is_known": name != "Unknown",
        })

    return results


# ============================================================
# 6. 標註繪圖（UI）
# ============================================================

# BGR 色碼
COLOR_KNOWN = (0, 220, 0)      # 綠色：認識的人
COLOR_UNKNOWN = (0, 165, 255)   # 橘色：陌生人
COLOR_TEXT_BG = (30, 30, 30)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_result(frame: np.ndarray, results: list[dict],
                show_distance: bool = True) -> np.ndarray:
    """
    把辨識結果畫到彩色畫面上。

    綠框 = 已註冊的人、橘框 = 陌生人
    """
    for r in results:
        x, y, w, h = r["rect"]
        color = COLOR_KNOWN if r["is_known"] else COLOR_UNKNOWN
        label = r["name"]

        if show_distance and np.isfinite(r["distance"]):
            label = f"{label} ({r['distance']:.0f})"

        # 人臉框
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # 文字底板
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 2)
        ty = max(0, y - 5)
        cv2.rectangle(frame, (x, ty - th - 8), (x + tw + 10, ty + 4), COLOR_TEXT_BG, -1)
        cv2.putText(frame, label, (x + 5, ty - 2), FONT, 0.55, color, 2)

    return frame


def draw_hud(frame: np.ndarray, title: str = "FACE RECOGNITION",
             subtitle: str = "") -> np.ndarray:
    """畫面上方的狀態列。"""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 46), (25, 25, 25), -1)
    cv2.putText(frame, title, (12, 30), FONT, 0.75, (0, 220, 255), 2)
    if subtitle:
        cv2.putText(frame, subtitle, (w - 10 - 8 * len(subtitle), 30),
                    FONT, 0.45, (180, 180, 180), 1)
    return frame