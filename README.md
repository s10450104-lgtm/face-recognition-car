# 人臉辨識與遙控系統（OpenCV + HAAR + LBPH）

> 課程作業一：人臉識別與辨識
> 技術：OpenCV HAAR Cascade（人臉偵測） + LBPH（人臉辨識）+ Flask

先在電腦內建鏡頭測試，測試成功後可直接移植到樹莓派自走車。

**組別**：第六組 ｜ **成員**：411206117 陳信安、411206137 郭振宇、411206157 徐睿澤

📄 **完整作業報告**：[`site/index.html`](site/index.html)（直接用瀏覽器開啟）

---

## 目錄

- [快速開始](#快速開始)
- [Web 介面](#web-介面)
- [專案架構](#專案架構)
- [三個階段說明](#三個階段說明)
- [資訊處理流程](#資訊處理流程)
- [參數調整](#參數調整)
- [移植到樹莓派](#移植到樹莓派)
- [疑難排解](#疑難排解)
- [實測結果](#實測結果)

---

## 快速開始

### 環境需求

```bash
pip install -r requirements.txt
```

> `opencv-contrib-python` **必須安裝**，LBPH 辨識器在這個套件裡面。

### 方式 A：Web 介面（完整功能，推薦）

```bash
python app.py
```

開瀏覽器進入 **http://127.0.0.1:5000** ，一次搞定：

- 即時影像串流 + 人臉辨識標註
- 9 向方向鍵遙控 + 油門滑桿 + 雲台角度
- **在網頁上直接註冊新的人臉**（不用開終端機）
- 線上重新訓練模型
- 系統日誌

### 方式 B：命令列

```bash
# Step 1：收集人臉照片（會開啟鏡頭，依提示按空白鍵拍照）
python collect_faces.py

# Step 2：訓練模型（讀取 data/ 全部照片，輸出到 models/）
python train_lbph.py

# Step 3：即時辨識（請另一位同學站到鏡頭前）
python recognize_lbph.py
```

### 環境自我檢查

```bash
python self_test.py
```

檢查 Python / OpenCV / NumPy 版本、HAAR 載入、LBPH 建立、攝影機可用性，
並用內建測試資料跑一次完整的「訓練 → 存檔 → 載入 → 辨識」流程。

---

## Web 介面

### 啟動

```bash
python app.py
```

預設監聽 `0.0.0.0:5000`，所以同一個 WiFi 下別人也可以用
`http://<你的IP>:5000` 連進來（查 IP：`ipconfig`）。

### 介面配置

```
┌─────────────────────────────────────────┬──────────────────┐
│  LIVE FEED                              │  DRIVE CONTROL   │
│  ┌───────────────────────────────────┐  │  ┌────────────┐  │
│  │                                   │  │  │ 油門滑桿   │  │
│  │      即時影像 + 辨識標註           │  │  └────────────┘  │
│  │                                   │  │  ┌──┬──┬──┐      │
│  └───────────────────────────────────┘  │  │◤ │▲ │◥ │      │
│  已註冊 │ 臉孔 │ 辨識成功 │ 閾值         │  │◀ │中心│▶ │      │
├─────────────────────────────────────────┤  │◣ │▼ │◢ │      │
│                                          │  └──┴──┴──┘      │
│                                          │  STOP │ EMERGENCY │
│                                          │  雲台 X / Y 滑桿   │
├─────────────────────────────────────────┼──────────────────┤
│                                          │  FACE REGISTRY   │
│                                          │  名稱 + 張數      │
│                                          │  註冊 / 訓練 / 截圖 │
│                                          │  已註冊人員清單     │
└─────────────────────────────────────────┴──────────────────┘
┌───────────────────────────────────────────────────────────┐
│  SYSTEM LOG                                                │
└───────────────────────────────────────────────────────────┘
```

### 框的顏色意義

| 顏色 | 意義 |
|------|------|
| 綠色 | 已註冊的人，距離 ≤ 60 |
| 黃色 | 疑似同一人，距離 ≤ 85 |
| 橘色 | 陌生人，距離 > 85 |

### 鍵盤操作

| 按鍵 | 功能 |
|------|------|
| `↑` `W` | 前進 |
| `↓` `S` | 後退 |
| `←` `A` | 左轉 |
| `→` `D` | 右轉 |
| `空白鍵` | 停止 |
| `R` | 重新載入模型 |

### 線上註冊人臉（重點功能）

不需要開終端機，直接在網頁上操作：

1. 在 `IDENTITY / 名稱` 輸入 `student01`（支援中文，例如 `王小明`）
2. 設定要拍幾張（預設 30）
3. 按 **開始註冊**，對著鏡頭保持不動
   - 系統會自動跳過「沒偵測到人臉」或「人臉太小」的畫面
   - 可以稍微左右轉頭、改變表情，讓紋理特徵更豐富
   - 進度條會即時顯示 `12 / 30`
4. 拍完後按 **重新訓練模型**，新的人就會被辨識出來
5. 已註冊清單可單獨刪除某個人

> 再次註冊同一個名字會**從現有張數接續追加**，不會覆蓋舊資料。

### 馬達模式

目前是 **SIMULATION（模擬）** 模式，右上角會標示 `SIMIMULATION`：
指令會照常計算左右輪轉速並顯示在畫面上，但**不會驅動任何硬體**。
適合先把網頁介面、執行緒、API 都測穩定，之後再上車。

上車後只要改一行：

```python
motor = MotorControl(mode=MODE_RPI)     # 原本是 MODE_SIMULATION
```

`app.py` 的其他部分完全不用動。

### API

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/` | 主頁面 |
| GET | `/video_feed` | MJPEG 影像串流 |
| GET | `/video_snapshot` | 單張 JPEG 快照 |
| GET | `/api/status` | 系統狀態（FPS、臉孔、馬達狀態） |
| GET | `/api/people` | 已註冊人員清單 |
| POST | `/api/motor/<action>` | 馬達動作（forward/backward/left/.../stop） |
| POST | `/api/motor/speed` | 設定速度 `{"speed": 60}` |
| POST | `/api/motor/servo` | 雲台角度 `{"x": 90, "y": 90}` |
| POST | `/api/motor/stop` | 停止 |
| POST | `/api/motor/emergency` | 緊急停止 |
| POST | `/api/register/start` | 開始註冊 `{"label": "student01", "count": 30}` |
| POST | `/api/register/cancel` | 取消註冊 |
| POST | `/api/train` | 重新訓練模型 |
| POST | `/api/load_model` | 載入模型 |
| POST | `/api/delete/<label>` | 刪除某人 |

範例：

```bash
curl -X POST http://127.0.0.1:5000/api/motor/forward -H "Content-Type: application/json" -d '{"speed":60}'
curl http://127.0.0.1:5000/api/status
```

### 效能設計

對應課堂 TROUBLE SHOOTING 的做法：

- **背景執行緒擷取**：攝影機迴圈獨立跑，HTTP 請求只取「最新那一幀」，
  辨識再慢也不會卡住串流
- **跳幀**：`DETECT_EVERY_N_FRAMES` 控制辨識頻率（車上建議設 2~3）
- **threading.Lock**：背景執行緒與多個 HTTP 執行緒共用狀態，全部加鎖
- **背景訓練**：`/api/train` 不阻塞請求，訓練在背景執行緒進行
- **馬達鎖**：`MotorControl` 內部的 `threading.Lock` 保護 I2C 操作

---

## 專案架構

```
face/
├── app.py                 # ★ Flask Web 主程式（遙控 + 辨識 + YOLO 整合）
├── motor_control.py       # ★ 馬達控制（模擬模式 / 樹莓派真實模式）
├── yolo_detector.py       # ★ YOLOv8 物件偵測（第二階段）
├── yolo_model.pt          #   YOLO 權重（.gitignore 排除，6.3 MB）
├── templates/
│   └── index.html         # ★ Web 介面（Cyberpunk 風格）
├── static/
│   ├── css/
│   │   └── style.css      # ★ 介面樣式
│   └── js/
│       └── main.js        # ★ 前端邏輯（串流、輪詢、控制、註冊）
├── collect_faces.py       # Step 1：收集人臉照片（命令列）
├── train_lbph.py          # Step 2：訓練 LBPH 模型（命令列）
├── recognize_lbph.py      # Step 3：即時人臉辨識（命令列 + 視窗）
├── face_core.py           # 核心模組（偵測/裁切/正規化/訓練/辨識）
├── camera_source.py       # 鏡頭抽象層（PC webcam ↔ RPi Picamera2）
├── LOBOROBOT2.py          # 課程提供的馬達控制庫（真實模式才會用到）
├── self_test.py           # 環境與流程自我檢查
├── site/                  # ★ 作業報告（靜態 HTML，瀏覽器直接開啟）
│   ├── index_template.html  #   報告模板（改這個）
│   ├── index.html           #   產生的單檔版（圖片 base64 內嵌）
│   ├── make_standalone.py   #   模板 → 單檔版
│   ├── upload_neocities.py  #   上傳到 Neocities
│   └── assets/
│       ├── report.css
│       └── shot-*.jpg     #   實作截圖
├── data/                  # 訓練照片（.gitignore 排除）
│   ├── 17/  17_001.jpg ~ 17_030.jpg
│   ├── 37/  37_001.jpg ~ 37_030.jpg
│   └── student01/
├── models/                # 訓練結果（.gitignore 排除）
│   ├── trainer.yml        # LBPH 模型本體
│   └── trainer.json       # label 對照表
├── logs/                  # 截圖與預覽（.gitignore 排除）
├── requirements.txt
└── README.md
```

> `data/` 與 `models/` 存放個人臉部影像與訓練結果，已透過 `.gitignore` 排除，不會進入版控。

★ = Web 介面相關

### 為什麼要多這幾個模組

**`face_core.py`** — 把偵測、裁切、正規化、訓練、辨識集中在一處。
如果三個腳本各自寫一份，參數很容易不一致；而 LBPH 只要訓練和辨識的
預處理有一點點不同，辨識率就會明顯下降。集中管理可以避免這個問題。

**`camera_source.py`** — 統一包裝攝影機來源，程式只要呼叫
`cam.read()` 就拿到影像。電腦用 `cv2.VideoCapture`、車子用 `Picamera2`，
上層程式完全不需要知道差別，這是「先在電腦測、之後搬上車」的關鍵。

**`motor_control.py`** — 唯一碰硬體的檔案。上層只呼叫
`motor.move("forward")`，模擬模式就只算速度、真實模式才驅動 I2C。
現在沒車也能把整個 Web 介面測完，上車時只改一個參數。

---

## 三個階段說明

### Step 1 — `collect_faces.py` 收集人臉照片

啟動攝影機，對準鏡頭後按空白鍵拍照，收集 30 張。

拍照時會自動：

1. 跑 HAAR 確認真的有偵測到人臉，沒偵測到就不存
2. 檢查人臉太小（小於 100px）就跳過，避免遠距離雜訊
3. 裁切人臉 → 直方圖均衡化 + CLAHE → 存成 200×200 灰階圖

> **拍照技巧**：稍微左右轉頭、改變表情、微笑。讓 LBPH 有足夠的紋理
> 變化，之後換姿勢時才認得出來。全都一樣的直視照片效果反而較差。

支援中文姓名（如 `王小明`）。

### Step 2 — `train_lbph.py` 訓練模型

自動掃描 `data/` 下所有人臉資料並訓練。

```
讀取照片 → 灰階 → 正規化（均衡化 + CLAHE）
        → update() 累積紋理特徵向量 → write() 輸出模型
```

會檢查每個人是否達到 30 張，不足會提示但仍可繼續訓練。
輸出 `models/trainer.yml`（模型）與 `models/trainer.json`（label 對照表）。

### Step 3 — `recognize_lbph.py` 即時辨識

開啟攝影機，即時偵測並辨識所有人臉。

```
讀取畫面 → 鏡像翻轉 → 灰階 → HAAR 偵測
        → 裁切正規化 → LBPH 比對 → 連續幀確認 → 標註顯示
```

框的顏色意義：

| 顏色 | 意義 |
|------|------|
| 綠色 | 已註冊的人，距離 ≤ 60 |
| 黃色 | 疑似同一人，距離 ≤ 85 |
| 橘色 | 陌生人，距離 > 85 |

---

## 資訊處理流程

這是作業要求的「資訊處理流程」，完整一條龍：

```
攝影機擷取 (640×480 BGR)
    ↓
鏡像翻轉 cv2.flip          ← 符合使用者直覺
    ↓
轉灰階 cvtColor(BGR2GRAY)  ← 去除顏色，專注亮度紋理
    ↓
HAAR Cascade 偵測人臉       ← detectMultiScale
    ↓
裁切人臉 crop_face          ← HAAR 框往外擴 25%，統一縮成 200×200
    ↓
直方圖均衡化 equalizeHist  ← 抵消明暗差異
    ↓
CLAHE 局部均衡化            ← 處理側光、逆光
    ↓
LBPH 紋理特徵擷取           ← 8×8 網格，每格算 0~255 的紋理簽章
    ↓
與訓練樣本比對 predict     ← 回傳最近的樣本與距離
    ↓
閾值判斷 + 連續 3 幀確認
    ↓
標註框與名字，顯示在畫面
```

### HAAR 三個關鍵參數

```python
detector.detectMultiScale(
    gray,
    scaleFactor=1.1,   # 每次縮小的比例：越小越精確但越慢
    minNeighbors=5,    # 鄰居門檻：越大越嚴格、誤判越少
    minSize=(100, 100) # 最小人臉：過濾遠距離雜訊
)
```

### LBPH 原理

LBPH = Local Binary Patterns Histograms（局部二元模式直方圖）：

1. 把臉切成 8×8 = 64 個格子
2. 每格取 3×3 鄰域，中心像素比鄰居大記 1、比鄰居小記 0
3. 得到一個 0~255 的「紋理簽章」
4. 64 個簽章各累加一個直方圖 → 合成 **16384 維**特徵向量
5. 訓練時把所有向量存進模型；辨識時找最近的向量，回傳距離

**優點**：訓練只要幾秒、不需 GPU、對光線變化有一定容忍度
**缺點**：轉頭或戴眼鏡效果下降；需要用閾值控制陌生人判定

### 連續幀確認機制

單幀的辨識結果會抖動（某一幀突然認錯）。因此採用：

> 連續 **3 幀**都辨識到同一個人，才顯示那個名字。

中途換人則重新計數。追蹤 key 用「人臉中心的量化格子」而非 bbox，
因為 bbox 會跟著呼吸晃動每幀都變，當 key 等於永遠累積不到確認幀數。

---

## 參數調整

全部集中在 `face_core.py` 頂端：

```python
LBPH_PARAMS = dict(radius=1, neighbors=5, grid_x=8, grid_y=8)
REQUIRED_SAMPLES = 30
MIN_FACE_SIZE = 100
THRESHOLD_STRONG = 60.0     # 低於此值 → 判定同一人
THRESHOLD_WEAK = 85.0       # 高於此值 → 判定陌生人
CONFIRM_FRAMES = 3
```

### 怎麼調閾值

`python recognize_lbph.py` 執行時，畫面上會顯示每個人的 `distance`。

1. 找**已註冊的人**在各種角度/光線下的距離 → 這是上界
2. 找**陌生人**的距離 → 這是下界
3. 閾值設在兩者之間

實測參考（單人模型，30 張樣本）：

| 情況 | 距離 |
|------|------|
| 已知的人（正常亮度） | ~19 |
| 已知的人（偏暗 0.75x） | ~21 |
| 已知的人（偏亮 1.25x） | ~18 |
| 已知的人（距離較遠） | ~19 |
| 已知的人（較近、佔畫面大） | ~53 |

> 人臉佔畫面比例變化時距離會明顯上升（19 → 53），因為裁切時的相對位置
> 改變了。**建議訓練時也拍幾張不同距離的照片**，讓模型涵蓋更多尺度。

### 樹莓派效能調整

樹莓派 CPU 效能有限，建議調整：

```python
# 偵測參數放寬，減少縮放搜尋次數
detect_faces(gray, detector, scale_factor=1.3, min_neighbors=3)

# LBPH 網格加密取樣（更慢但更準，依需求取捨）
LBPH_PARAMS = dict(radius=1, neighbors=5, grid_x=6, grid_y=6)
```

---

## 移植到樹莓派

`camera_source.py` 已經做好抽象層，上層程式**不需要修改**。

### 1. 車端安裝套件

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install Flask ultralytics opencv-python opencv-contrib-python numpy
sudo apt install python3-picamera2     # 攝影機套件
```

### 2. 複製程式

把整個 `face/` 資料夾複製到車上（`data/`、`models/` 一起帶過去，
這樣不用重新收集照片）。

### 3. 執行

```bash
python self_test.py               # 先確認環境 OK
python recognize_lbph.py
```

`open_camera()` 會自動偵測是否在樹莓派上：

- 樹莓派 → 優先用 `Picamera2`
- 一般電腦 → 用 `cv2.VideoCapture`

### 車上效能建議

課堂提到 RPi 5 跑 YOLO 會當機，主因是 CPU 負載與馬達啟動的電流尖峰。
人臉辨識的對應做法：

- **解析度**：640×480 夠用，不要開 1080p
- **跳幀**：辨識不必每幀都做，可每 2~3 幀一次
- **執行緒鎖**：若同時用 Flask 推影像流，辨識放背景執行緒，
  用 `threading.Lock` 保護共用狀態

---

## YOLOv8 物件偵測（第二階段）

在既有的 HAAR + LBPH 人臉辨識之外，加上了 YOLOv8n 物件偵測，
偵測本作業指定的五類：**人 / 汽車 / 公車 / 機車 / 自行車**。

### 安裝

```bash
# ⚠️ 樹莓派一定要用 CPU 索引，理由見下方疑難排解
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics

# 模型檔（6.3 MB，不入版控）
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt \
     -O yolo_model.pt
```

沒有安裝也沒關係，系統會自動降級成「只做人臉辨識，不做物件偵測」。

### 效能實測（Raspberry Pi 5 / Debian 13 / torch 2.14.1+cpu）

| imgsz | 推論耗時 | FPS |
|------:|--------:|----:|
| 640 | 184.5 ms | 5.42 |
| 416 | 92.6 ms | 10.80 |
| **320** | **91.9 ms** | **10.89** ← 採用 |
| 256 | 57.7 ms | 17.34 |

**為什麼是 320？** 640 的耗時是 320 的兩倍，但物件偵測是看大方向用，
解析度低一點完全夠用，省下的時間給影像串流。

### 設計：為什麼推論要放獨立執行緒

樹莓派 4 個核心，YOLO 單次推論要 90 ms 以上。若直接在串流迴圈裡推論，
影像串流會掉到 5 FPS 以下（實測）。

改成獨立背景執行緒後：

| 項目 | 數值 |
|---|---|
| 影像串流 FPS | 14.9（只有人臉辨識時是 20）|
| YOLO 推論 FPS | 10.0 |
| 跳幀設定 | 每 3 幀推論一次 |

串流與推論透過 `state` 交換資料：`yolo_loop()` 讀 `state.raw_frame`
推論後寫 `state.ai_results`，串流迴圈只負責把結果畫上去。
前端另外用 `/api/ai_status` 抓取，不會卡住串流。

### 可調參數

都在 `app.py` 開頭：

```python
YOLO_ENABLED = True        # 沒裝 ultralytics 時可設 False
YOLO_IMGSZ = 320           # 想更流暢可降到 256
YOLO_CONF = 0.30           # 信心門檻
YOLO_SKIP_FRAMES = 3       # 每 N 幀推論一次
```

串流和 AI 在搶 CPU。若覺得卡，先把 `YOLO_SKIP_FRAMES` 調到 4~5，
或把 `YOLO_IMGSZ` 降到 256。

### API

```
GET /api/ai_status
```

```json
{
  "enabled": true,
  "imgsz": 320,
  "skip_frames": 3,
  "class_ids": [0, 1, 2, 3, 5],
  "infer_ms": 99.8,
  "fps": 10.0,
  "count": 1,
  "detections": [
    { "class": "person", "cn": "人", "conf": 0.93, "rect": [0, 0, 617, 473] }
  ]
}
```

`class_ids` 對應 COCO 編號：

| 編號 | 類別 | 中文 |
|----:|------|------|
| 0 | person | 人 |
| 1 | bicycle | 自行車 |
| 2 | car | 汽車 |
| 3 | motorcycle | 機車 |
| 5 | bus | 公車 |

### 命令列測試

```bash
python yolo_detector.py
```

會開攝影機抓一幀、跑一次推論、印出結果，
並驗證跳幀設定是否真的有效果。

---

## 疑難排解

### pip install ultralytics 失敗：incomplete-download

**症狀**

```
error: incomplete-download
? Download failed after 6 attempts because not enough bytes were received
?   (195.8 MB/542.8 MB)
? URL: .../nvidia_cublas-13.1.1.3-py3-none-manylinux_2_27_aarch64.whl
```

**原因**

`ultralytics` 會連帶安裝 NVIDIA 的 CUDA 套件，
其中 `nvidia_cublas` 一個就有 **542 MB**。

樹莓派沒有 NVIDIA GPU，這些套件根本用不到；
加上檔案太大，連線中斷後 pip 續傳也傳不完，就會失敗。

**解決**

改用 PyTorch 官方的 **CPU 專用索引**：

```bash
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

CPU 版不會拉任何 CUDA 套件，torch 本體只有 99 MB，安裝順利。

### 安裝 YOLO 後人臉辨識壞掉：cv2 has no attribute 'CascadeClassifier'

**症狀**

```
[FAIL] 完整流程  → AttributeError: module 'cv2' has no attribute 'CascadeClassifier'
```

**原因**

用 `pip install --system-site-packages .venv` 建立虛擬環境時，
`ultralytics` 把 **opencv-python 5.0** 裝進 venv 裡，
**蓋掉了系統原本的 OpenCV 4.10**。

OpenCV 5.0 把 `CascadeClassifier` 等舊 API 搬走了，
而 `face_core.py` 是照 4.10 寫的，於是人臉辨識整個掛掉。

**解決**

把 venv 裡那份 OpenCV 移除，靠 `--system-site-packages` 回到系統版本：

```bash
source .venv/bin/activate
pip uninstall -y opencv-python
python -c "import cv2; print(cv2.__version__, cv2.__file__)"
```

確認輸出的是系統路徑（`/usr/lib/python3/dist-packages/cv2...`）而不是
`.../.venv/lib/python3.13/site-packages/cv2/`。

ultralytics 只要求 opencv >= 4.6，4.10 完全夠用。

### 攝影機開不起來

```bash
python -c "from camera_source import open_camera; print(open_camera().name)"
```

依序檢查：

1. **被其他程式佔用** — Teams / Zoom / 瀏覽器（Chrome、Edge 常在背景用鏡頭），全部關掉
2. **Windows 隱私權** — 設定 → 隱私權與安全性 → 相機 → 開啟
3. **攝影機被停用** — 裝置管理員 → 影像裝置 → 啟用裝置
4. **外接攝影機編號不同** — `python recognize_lbph.py --index 1`
5. **重開機** — 驅動更新後常需要重開機才生效

### 中文顯示成亂碼

程式已內建修正（自動切換 console 到 UTF-8）。若仍有問題：

```powershell
chcp 65001
```

PowerShell 建議設定：`設定 → 外觀 → 字型 → Cascadia Mono`，
並在設定檔加入：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

### 訓練時照片被跳過

```
[!] xxx.jpg 偵測不到人臉，略過
```

代表照片裡沒有清楚的人臉，可能是模糊、太暗或角度太偏。
重新收集即可。

> 注意：`collect_faces.py` 存的是**已裁切的人臉**，所以 `train_lbph.py`
> 不會再對它們跑 HAAR 重新裁切。這是刻意的 —— HAAR 的框通常只涵蓋
> 眼睛到下巴，重新裁切會切掉臉部邊緣，造成訓練輸入與辨識輸入不一致，
> 導致「明明是同一人卻被判定為陌生人」。
>
> 如果你的 `data/` 放的是**未裁切的原始照片**，可以這樣訓練：
>
> ```python
> train_lbph_model(samples, MODEL_DIR, re_detect=True)
> ```

### 辨識率不高

依序檢查：

1. **照片數量不足** — 每人至少 30 張，愈多愈準
2. **照片變化不夠** — 全部都是同樣的直視照片，模型學不到變化
3. **角度、光線差異** — 收集時加入不同角度與光線
4. **距離變化** — 人臉佔畫面比例變大時距離會上升（實測 19 → 53），
   建議也拍幾張不同距離的
5. **閾值不對** — 對照 `distance` 實際數值重新設定
6. **戴眼鏡或側臉** — LBPH 對這些情況較弱，可多收這類照片

### OpenCV 版本錯誤

```python
AttributeError: module 'cv2' has no attribute 'FaceRecognizerLBPH_create'
```

這是 OpenCV 4.x 的 API。程式已做相容處理：

- OpenCV 4.x → `cv2.FaceRecognizerLBPH_create()`
- OpenCV 5.x → `cv2.face.LBPHFaceRecognizer_create()`

兩種都能跑。但若報「不支援 LBPH」，代表裝的是純 `opencv-python`，
必須改裝 `opencv-contrib-python`。

```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

### 中文檔名存不下來

已用 `face_core.imwrite_safe()` 處理（`cv2.imwrite` 在 Windows 上
遇到中文路徑會靜默失敗）。若你自己寫的程式也有存圖需求，記得改用：

```python
from face_core import imwrite_safe, imread_safe
imwrite_safe("輸出/中文名稱.jpg", img)
```

---

## 實測結果

以下是用 `self_test.py` 的測試資料跑出的結果：

```
1. 執行環境
  Python  : 3.12.10
  OpenCV  : 5.0.0
  NumPy   : 2.5.3

2. HAAR Cascade 人臉偵測
  [PASS] 載入 haarcascade_frontalface_alt.xml
  [PASS] cascade 路徑存在
  [PASS] 空白影像回傳空結果（不崩潰）

3. LBPH 辨識器
  [PASS] 建立 LBPH recognizer   → LBPHFaceRecognizer
  [PASS] LBPH 參數正確

6. 完整流程（訓練 → 存檔 → 載入 → 辨識）
  [PASS] 掃描資料集 → 2 人 × 30 張
  [PASS] LBPH 訓練完成   → 60 個樣本
  [PASS] 模型檔已輸出
  [PASS] Label 對應正確  → ['student01', 'student02']
  [PASS] 模型載入成功
  [PASS] 辨識命中 student01
  [PASS] TrackState 前 2 幀不確認
  [PASS] 第 3 幀確認
  [PASS] 結果標註繪製
  [PASS] HUD 繪製

  通過 16   提示 2   失敗 0
```

真實人臉測試（不同亮度、縮放、位置的 8 種情況）：

```
  OK 正常亮度          -> 辨識成功  distance=19.94
  OK 偏暗 0.75x        -> 辨識成功  distance=20.67
  OK 偏亮 1.25x        -> 辨識成功  distance=18.43
  OK 較遠 scale0.75    -> 辨識成功  distance=19.14
  OK 較近 scale1.2     -> 辨識成功  distance=53.14
  OK 偏左              -> 辨識成功  distance=18.98
  OK 偏右              -> 辨識成功  distance=19.28
  OK 偏暗+較小         -> 辨識成功  distance=47.11

  距離統計: min=18.43 max=53.14 mean=27.09
  閾值: 強=60.0  弱=85.0
```

全部 8 種情況都正確辨識，距離都落在閾值內。
非人臉影像則直接被 HAAR 偵測階段擋掉，不會進入辨識。

---

## 開發過程記錄

### 遇到的問題與解決

**1. OpenCV 5.0 API 變更**

```
AttributeError: module 'cv2' has no attribute 'FaceRecognizerLBPH_create'
```

5.0 把 LBPH 搬到 `cv2.face` 底下。寫成相容兩版的_factory()，
車子（RPi 通常 4.x）與電腦（5.x）都能跑同一份程式。

**2. 訓練時照片被整批跳過**

原本 `train_lbph.py` 會對每張照片再跑一次 HAAR 重新裁切，
但 HAAR 常常偵測不到已裁切的臉，導致 60 張全部被丟棄。
改成直接使用照片完整內容。

**3. 同一人被誤判為陌生人（重要）**

進一步發現：即使偵測到了，HAAR 的框只涵蓋眼睛到下巴，
對已裁切的人臉再裁一次會切掉臉部邊緣，造成**訓練輸入與辨識輸入不一致**，
距離被灌大到 85 以上，超過閾值。

修正方式是確立「收集與辨識走完全相同的裁切流程」這個契約：
收集時存 `crop_face(HAAR框, padding=0.25)`，辨識時用同樣的
`crop_face(HAAR框, padding=0.25)`，訓練時直接使用存下的影像不再二次裁切。

**4. `TrackState` 確認機制無效**

原本 `update()` 的兩個分支都回傳 `self.current_label`，
讓「連續 3 幀確認」形同虛設。改為 `pending_label` + `pending_hits`
的兩階段設計，只有連續達標才升級為顯示結果。

**5. Windows 中文檔名靜默失敗**

```python
cv2.imwrite("中文測試.jpg", img)   # → False，檔案沒產生，也不丟例外
```

`cv2.imwrite`/`imread` 在 Windows 走 ANSI 路徑 API，中文會失敗。
改用 `imencode()` + `numpy.tofile()` 實作 `imwrite_safe()` / `imread_safe()`，
並放寬姓名檢查讓中文姓名可用。

**6. 中文訊息亂碼**

Windows 主控台預設 Big5(cp950)，Python 輸出 UTF-8 會亂碼。
在 `face_core.py` 匯入時呼叫 `SetConsoleOutputCP(65001)` 修正。

**7. OpenCV 錯誤訊息干擾**

攝影機開不起來時 cv2 會噴 FFMPEG 錯誤到 stderr，蓋掉自己的訊息。
設定 `OPENCV_LOG_LEVEL` 與 `setLogLevel(LOG_LEVEL_SILENT)` 靜音；
另外在 Windows 上移除 `CAP_ANY`（它會退回 FFMPEG 後端，
該後端的訊息會繞過 log level）。

### Web 介面開發時遇到的問題

**8. 線上註冊只存到 1 張照片**

註冊 API 只在開始時算一次檔名（`tester01_001.jpg`），之後每張都覆寫
同一個檔案。改成依「目前已擷取數量」即時計算檔名。

```python
n = task["count"] + 1
path = Path(task["dir"]) / f"{task['label']}_{n:03d}.jpg"
```

**9. 舊任務會誤殺新任務**

註冊完成時直接 `state.register_task = None`，但區域變數 `task` 可能已經
指向被新請求取代的舊 dict，會把新任務清掉。改成先確認
`state.register_task is task` 才動作。

**10. STOP 按鈕完全沒反應**

`bindDpad()` 用 `$$('.dbtn')` 綁事件，但 STOP 按鈕的 class 是
`.btn.btn-stop`，不是 `.dbtn`，所以從未被綁到。改成用
`$$('[data-action]')`，以資料屬性綁定就與 class 無關。

**11. Flask 3.x 的 JSON 中文變成 `\uXXXX`**

`app.config["JSON_AS_ASCII"]` 在 Flask 3.x 已被移除，改成
`app.json.ensure_ascii = False`。

**12. 主控台看不到註冊進度**

`Start-Process` 重導向輸出時 stdout 會被區塊緩衝，`print()` 不會即時顯示。
在 `print()` 加 `flush=True`，並用 `python -u app.py` 啟動。

---

## 參考資料

- OpenCV HAAR Cascade 官方文件
- OpenCV Face Recognition with LBPH
- PiCamera2 官方文件
- 課程簡報：`W5_Web_遙控暨辨識自走車20261007.pptx`