#!/bin/bash
# ================================================================
#  deploy.sh — 樹莓派一鍵部署腳本
#  在樹莓派上執行：bash deploy.sh
#
#  做什麼：
#    1. 檢查環境（I2C / 攝影機 / Python）
#    2. 安裝系統套件
#    3. 建立 Python 虛擬環境並安裝相依套件
#    4. 驗證 OpenCV + LBPH 可用
#    5. 啟動辨識系統並測試
#
#  選項：
#    bash deploy.sh              # 完整部署
#    bash deploy.sh --check      # 只檢查不改變任何東西
# ================================================================

set -u

PROJECT_DIR="${PROJECT_DIR:-$HOME/face}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[注意]${NC} $*"; }
err()  { echo -e "${RED}[錯誤]${NC} $*"; }
info() { echo -e "${CYAN}▸${NC} $*"; }

echo ""
echo "============================================================"
echo -e "  ${BOLD}樹莓派部署腳本${NC}"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# ────────────────────────────────────────────────────────────────
# 1. 環境檢查
# ────────────────────────────────────────────────────────────────
echo "─── [1/5] 環境檢查 ───────────────────────────────────────"

MODEL=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "未知")
OSNAME=$(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d'"' -f2)
echo "  型號：$MODEL"
echo "  系統：$OSNAME"
echo "  核心：$(uname -r)"
echo "  記憶體：$(free -h | awk '/^Mem:/{print $2" 總計 / "$7" 可用"}')"
echo "  IP　 ：$(hostname -I 2>/dev/null | tr ' ' '\n' | head -1)"
echo ""

# I2C / PCA9685
echo "  【I2C 檢查】"
PCA_OK=0
if command -v i2cdetect >/dev/null 2>&1; then
    if sudo i2cdetect -y 1 2>/dev/null | grep -q "40"; then
        ok "PCA9685 偵測到（I2C 0x40）"
        PCA_OK=1
    else
        warn "沒看到 PCA9685 (0x40)"
        warn "  → 馬達控制會無法運作，但人臉辨識不受影響"
        warn "  → 檢查接線，或：sudo raspi-config → Interface Options → I2C"
    fi
else
    warn "i2cdetect 未安裝（sudo apt install i2c-tools）"
fi
echo ""

# 攝影機
echo "  【攝影機檢查】"
CAMS=$(ls /sys/class/video4linux/video*/name 2>/dev/null | wc -l)
if [ "$CAMS" -gt 0 ]; then
    ok "偵測到 $CAMS 個影像裝置"
    for dev in /sys/class/video4linux/video*/name; do
        [ -e "$dev" ] && echo "      $(basename $(dirname $dev)): $(cat $dev)"
    done
else
    warn "沒有偵測到攝影機"
    warn "  → 確認排線插好、攝影機模組有接上"
fi
echo ""

# Python
echo "  【Python 檢查】"
PYV=$(python3 --version 2>&1)
info "$PYV"
echo ""

if [ $CHECK_ONLY -eq 1 ]; then
    echo "============================================================"
    echo "  僅檢查模式，未做任何變更"
    echo "============================================================"
    echo ""
    [ "$PCA_OK" -eq 1 ] && ok "PCA9685 已就緒" || warn "PCA9685 未偵測到（馬達不可用）"
    [ "$CAMS" -gt 0 ] && ok "攝影機已就緒" || warn "攝影機未偵測到"
    echo ""
    echo "  確認無誤後執行完整部署：bash deploy.sh"
    exit 0
fi

# ────────────────────────────────────────────────────────────────
# 2. 系統套件
# ────────────────────────────────────────────────────────────────
echo "─── [2/5] 安裝系統套件 ───────────────────────────────────"

echo "  更新套件清單..."
sudo apt-get update -qq 2>&1 | tail -2

INSTALL_PKGS="python3-picamera2 python3-smbus python3-dev i2c-tools python3-pip"
echo "  安裝：$INSTALL_PKGS"
sudo apt-get install -y -qq $INSTALL_PKGS 2>&1 | grep -v "^$" | tail -5
ok "系統套件完成"
echo ""

# ────────────────────────────────────────────────────────────────
# 3. 虛擬環境 + Python 套件
# ────────────────────────────────────────────────────────────────
echo "─── [3/5] Python 虛擬環境 ────────────────────────────────"

if [ ! -d "$PROJECT_DIR" ]; then
    err "找不到專案目錄：$PROJECT_DIR"
    err "請先把 face/ 資料夾整個複製過去再執行本腳本"
    exit 1
fi

cd "$PROJECT_DIR" || exit 1
info "專案目錄：$PROJECT_DIR"

VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    info "建立虛擬環境（--system-site-packages 可沿用系統套件）"
    python3 -m venv --system-site-packages "$VENV_DIR" 2>&1 | tail -3
    ok "虛擬環境建立完成"
else
    info "虛擬環境已存在，略過建立"
fi

source "$VENV_DIR/bin/activate"
info "Python：$(python --version)"

echo ""
info "安裝 Python 套件（opencv-contrib-python 較大，需 1-3 分鐘）..."
pip install --quiet --upgrade pip 2>&1 | tail -1
pip install --quiet -r requirements.txt 2>&1 | tail -5

# RPi 額外需要的
pip install --quiet adafruit-circuitpython-pca9685 gpiozero 2>&1 | tail -2

ok "Python 套件完成"
echo ""

# ────────────────────────────────────────────────────────────────
# 4. 驗證 OpenCV
# ────────────────────────────────────────────────────────────────
echo "─── [4/5] 驗證 OpenCV 與 LBPH ────────────────────────────"

python << 'PYEOF'
import sys
import cv2
import numpy as np

print(f"  OpenCV : {cv2.__version__}")
print(f"  NumPy  : {np.__version__}")

# LBPH：相容 OpenCV 4.x / 5.x
has_lbph = (hasattr(cv2, "FaceRecognizerLBPH_create") or
            (hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")))
if has_lbph:
    print("  LBPH   : 可用")
else:
    print("  LBPH   : 不可用 ← 需要 opencv-contrib-python")
    sys.exit(1)

# HAAR cascade
import os
path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt.xml")
det = cv2.CascadeClassifier(path)
if det.empty():
    print("  HAAR   : 載入失敗")
    sys.exit(1)
print("  HAAR   : 載入成功")

# 實際做一次偵測
blank = np.full((480, 640), 128, np.uint8)
faces = det.detectMultiScale(blank, 1.1, 5, minSize=(100, 100))
print(f"  空影像測試: {'通過' if len(faces) == 0 else '異常'}")
PYEOF

if [ $? -eq 0 ]; then
    ok "OpenCV 驗證通過"
else
    err "OpenCV 驗證失敗，請檢查上面的錯誤訊息"
    exit 1
fi
echo ""

# ────────────────────────────────────────────────────────────────
# 5. 自我檢查 + 總結
# ────────────────────────────────────────────────────────────────
echo "─── [5/5] 執行專案自我檢查 ──────────────────────────────"
echo ""

if python self_test.py; then
    ok "自我檢查通過"
else
    warn "自我檢查有項目未通過（請看上面的 [WARN] 或 [FAIL]）"
fi
echo ""

echo "============================================================"
echo "  部署完成"
echo "============================================================"
echo ""
echo "  下一步："
echo ""
echo "  【A】先測試鏡頭能不能開"
echo "     python camera_source.py"
echo ""
echo "  【B】啟動 Web 介面（馬達維持模擬模式）"
echo "     python app.py"
echo "     然後在筆電瀏覽器開："
echo "     http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "  【C】在車上直接看辨識視窗（需要接螢幕）"
echo "     python recognize_lbph.py"
echo ""
echo "  【D】訓練模型（如果還沒訓練過）"
echo "     python train_lbph.py"
echo ""
if [ "$PCA_OK" -eq 1 ]; then
    echo -e "  ${GREEN}PCA9685 已就緒${NC} — 馬達控制可以使用"
else
    echo -e "  ${YELLOW}PCA9685 未偵測到${NC} — 人臉辨識可正常使用，馬達暫時無法驅動"
fi
echo ""
echo "============================================================"
echo ""