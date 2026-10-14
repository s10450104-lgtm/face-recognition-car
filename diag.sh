#!/bin/bash
# ============================================================
#  樹莓派自我診斷腳本
#  在樹莓派終端機執行：bash diag.sh
#  會檢查所有部署前必要的條件
# ============================================================

echo "============================================================"
echo "  樹莓派部署前檢查"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

echo ""
echo "[1/8] 系統資訊"
echo "------------------------------------------------------------"
echo "  主機名稱 : $(hostname)"
echo "  型號     : $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo '讀取失敗')"
echo "  系統     : $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2)"
echo "  核心     : $(uname -r)"
echo "  架構     : $(uname -m)"
echo "  執行時間 : $(uptime -p)"

echo ""
echo "[2/8] 網路設定"
echo "------------------------------------------------------------"
echo "  所有 IP："
hostname -I 2>/dev/null | tr ' ' '\n' | grep -v '^$' | while read ip; do
    echo "    $ip"
done
echo "  WiFi 介面："
ip -4 addr show wlan0 2>/dev/null | grep inet || echo "    (wlan0 未連線)"
echo "  乙太網："
ip -4 addr show eth0 2>/dev/null | grep inet || echo "    (eth0 未連線或不存在)"

echo ""
echo "[3/8] I2C 狀態（馬達驅動 PCA9685 需要）"
echo "------------------------------------------------------------"
if command -v i2cdetect >/dev/null 2>&1; then
    echo "  i2cdetect 已安裝"
    echo "  I2C 裝置掃描（bus 1）："
    sudo i2cdetect -y 1 2>/dev/null | grep -E "^\s*[0-9a-f]{2}" || echo "    沒找到任何 I2C 裝置"
    echo ""
    if sudo i2cdetect -y 1 2>/dev/null | grep -q "40"; then
        echo "  ★ PCA9685 (0x40) 已偵測到 ✓"
    else
        echo "  ✗ 沒看到 PCA9685 (0x40)"
        echo "    → 檢查接線，或到主任辦公室確認模組"
        echo "    → 也可能 I2C 沒啟用：sudo raspi-config → Interface Options → I2C → Yes → reboot"
    fi
else
    echo "  ✗ i2cdetect 未安裝，請執行：sudo apt install i2c-tools"
fi

echo ""
echo "[4/8] 攝影機"
echo "------------------------------------------------------------"
echo "  攝影機型號："
for dev in /sys/class/video4linux/video*/name; do
    [ -e "$dev" ] && echo "    $(basename $(dirname $dev)): $(cat $dev 2>/dev/null)"
done
rpicam-out=$(libcamera-hello --help >/dev/null 2>&1 && echo "已安裝" || echo "未安裝")
echo "  rpicam-apps：$rpicam-out"
python3 -c "import picamera2; print('  Picamera2：已安裝')" 2>/dev/null || echo "  Picamera2：未安裝"

echo ""
echo "[5/8] Python 與套件"
echo "------------------------------------------------------------"
echo "  Python 系統版本： $(python3 --version 2>&1)"
if [ -d ~/face/.venv ]; then
    echo "  虛擬環境： ~/face/.venv"
    echo "  虛擬環境 Python： $(~/face/.venv/bin/python --version 2>&1)"
else
    echo "  虛擬環境：尚未建立"
fi
echo "  系統套件檢查："
for pkg in cv2 numpy flask smbus gpiozero picamera2; do
    r=$(python3 -c "import $pkg; print(getattr($pkg,'__version__','ok'))" 2>/dev/null)
    if [ -n "$r" ]; then
        echo "    $pkg : $r"
    else
        echo "    $pkg : 未安裝"
    fi
done

echo ""
echo "[6/8] OpenCV LBPH 支援（人臉辨識核心）"
echo "------------------------------------------------------------"
python3 -c "
import cv2
print('  OpenCV 版本:', cv2.__version__)
has = hasattr(cv2, 'FaceRecognizerLBPH_create') or (hasattr(cv2,'face') and hasattr(cv2.face,'LBPHFaceRecognizer_create'))
print('  LBPH 辨識器:', '可用' if has else '不可用（需安裝 opencv-contrib-python）')
import os
p = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt.xml')
print('  HAAR cascade:', '存在' if os.path.exists(p) else '找不到')
" 2>&1

echo ""
echo "[7/8] 記憶體與儲存空間"
echo "------------------------------------------------------------"
echo "  記憶體："
free -h | sed 's/^/    /'
echo "  磁碟："
df -h / | tail -1 | sed 's/^/    /'
echo "  CPU 溫度： $(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | awk '{printf "%.1f°C", $1/1000}')"

echo ""
echo "[8/8] 現有的部署目錄"
echo "------------------------------------------------------------"
if [ -d ~/face ]; then
    echo "  ~/face 已存在，內容："
    ls -la ~/face | sed 's/^/    /'
    echo ""
    echo "  訓練資料："
    if [ -d ~/face/data ]; then
        for d in ~/face/data/*/; do
            [ -d "$d" ] && echo "    $(basename $d): $(ls "$d" 2>/dev/null | wc -l) 張"
        done
    else
        echo "    (無 data 目錄)"
    fi
    echo "  模型："
    ls -la ~/face/models 2>/dev/null | sed 's/^/    /' || echo "    (無 models 目錄)"
else
    echo "  ~/face 尚未建立"
fi

echo ""
echo "============================================================"
echo "  檢查完成"
echo "============================================================"
echo ""
echo "  請把這份輸出拍照或複製給 AI 助手，"
echo "  就能知道缺什麼、怎麼部署。"
echo ""
echo "  重點確認："
echo "    [3/8] PCA9685 有沒有在 0x40"
echo "    [4/8] 攝影機有沒有被偵測到"
echo "    [6/8] LBPH 是否可用"