"""
app.py — Flask Web 主程式（遙控車版）
=====================================
把「遙控車馬達控制」包成網頁介面（方向鍵 D-pad UI）。

架構
----
    Flask 路由：  /              D-pad 控制頁
                 /api/motor/*   JSON 動作指令
                 /api/status    狀態輪詢

注意
----
    本資料夾只負責「遙控車馬達」。
    人臉辨識 / YOLO 在另一個資料夾 face_recognition（port 5000）。
    兩個服務互相獨立，可同時執行。

執行
----
    python3 app.py            # 前景執行，除錯用
    ./launch.sh               # 背景執行
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from motor_control import (ACTIONS, MODE_RPI, MODE_SIMULATION,
                           MotorControl)

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# Flask 3.x 起 JSON_AS_ASCII 設定已移除，改用 app.json 物件。
# 不關掉的話回傳的中文會變成 \uXXXX，前端顯示會很難閱讀。
app.json.ensure_ascii = False

# 模組層級先用模擬模式，避免模組載入時就因為馬達硬體問題而崩潰。
# 真實馬達模式會在 if __name__ == "__main__": 區塊內再次嘗試切換。
motor = MotorControl(mode=MODE_SIMULATION)

# 車控服務固定使用 5001，跟人臉辨識的 5000 分開。
PORT = 5001

# 人臉服務的位址：攝影機只能被一個程序開啟（Picamera2 限制），
# 相機由 5000 的人臉服務持有，車控頁的影像用代理轉發取得。
FACE_SERVICE = "http://127.0.0.1:5000"


# ============================================================
# 影像串流（代理自人臉服務）
# ============================================================

@app.route("/video_feed")
def video_feed():
    """
    把人臉服務（127.0.0.1:5000）的 MJPEG 串流轉發給瀏覽器。

    為什麼不自己開相機？
    ------------------
    Picamera2 一台相機只能被一個程序開啟。人臉服務（5000）已經
    持有相機並在畫面上疊了 YOLO 辨識框，車控頁直接轉發它的輸出，
    開車時就能同時看到影像和 AI 辨識結果。
    """
    try:
        upstream = urllib.request.urlopen(FACE_SERVICE + "/video_feed",
                                          timeout=10)
    except Exception as e:
        return Response(
            "camera upstream unavailable: " + str(e),
            status=503, mimetype="text/plain")

    def generate():
        try:
            while True:
                # 一次讀 64KB：MJPEG 一幀約 20~40KB，4096 會造成
                # 幾百次/秒的系統呼叫，徒增 CPU 負擔
                chunk = upstream.read(65536)
                if not chunk:
                    break
                yield chunk
        finally:
            upstream.close()

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_snapshot")
def video_snapshot():
    """單張截圖，同樣代理自人臉服務。"""
    try:
        with urllib.request.urlopen(FACE_SERVICE + "/video_snapshot",
                                    timeout=10) as r:
            data = r.read()
        return Response(data, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": "camera upstream unavailable: " + str(e)}), 503


# ============================================================
# 錄影（API 代理自人臉服務；相機與錄影都在 5000 那邊）
# ============================================================

def _proxy_record(path: str):
    """轉發錄影相關請求到人臉服務。"""
    try:
        if request.method == "POST":
            req = urllib.request.Request(FACE_SERVICE + path, data=b"{}",
                                         headers={"Content-Type":
                                                  "application/json"})
        else:
            req = urllib.request.Request(FACE_SERVICE + path)
        with urllib.request.urlopen(req, timeout=10) as r:
            return Response(r.read(), mimetype="application/json")
    except Exception as e:
        return jsonify({"error": f"人臉服務無回應：{e}"}), 503


@app.route("/api/record/start", methods=["POST"])
def api_record_start():
    return _proxy_record("/api/record/start")


@app.route("/api/record/stop", methods=["POST"])
def api_record_stop():
    return _proxy_record("/api/record/stop")


@app.route("/api/record/status")
def api_record_status():
    return _proxy_record("/api/record/status")


@app.route("/record/download/<name>")
def record_download(name):
    """影片檔下載（從人臉服務的 videos/ 目錄轉發）。"""
    try:
        with urllib.request.urlopen(
                f"{FACE_SERVICE}/record/download/{name}", timeout=30) as r:
            data = r.read()
        return Response(data, mimetype="video/x-msvideo",
                        headers={"Content-Disposition":
                                 f"attachment; filename={name}"})
    except Exception as e:
        return jsonify({"error": f"下載失敗：{e}"}), 503


# ============================================================
# 頁面
# ============================================================

@app.route("/")
def index():
    """D-pad 控制頁（複用人臉專案的模板，但只顯示遙控區塊）。"""
    return render_template("index.html",
                           actions=ACTIONS,
                           mode=motor.mode)


# ============================================================
# API：狀態查詢
# ============================================================

@app.route("/api/status")
def api_status():
    """系統總狀態（前端輪詢）。"""
    payload = {
        "running": True,
        "fps": 0.0,
        "frame_seq": 0,
        "faces": [],
        "face_count": 0,
        "known_count": 0,
        "model_loaded": False,
        "labels": [],
    }
    payload["motor"] = motor.state.to_dict()
    payload["ai"] = {"enabled": False, "count": 0}
    return jsonify(payload)


# ============================================================
# API：馬達控制
# ============================================================

@app.route("/api/motor/<action>", methods=["POST"])
def api_motor(action):
    """馬達動作指令。"""
    if action not in ACTIONS:
        return jsonify({"error": f"不支援的動作：{action}",
                        "valid": list(ACTIONS)}), 400

    data = request.get_json(silent=True) or {}
    speed = data.get("speed")

    if speed is not None:
        try:
            speed = int(speed)
        except (TypeError, ValueError):
            return jsonify({"error": "speed 必須是數字"}), 400
        if not 0 <= speed <= 100:
            return jsonify({"error": "speed 必須在 0~100"}), 400

    try:
        result = motor.move(action, speed)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        # lock 等待逾時：馬達控制忙碌，讓前端知道而不是 hang 住
        return jsonify({"error": str(e)}), 503

    return jsonify({"ok": True, "motor": result})


@app.route("/api/motor/speed", methods=["POST"])
def api_motor_speed():
    """調整速度。"""
    data = request.get_json(silent=True) or {}
    speed = data.get("speed")
    if speed is None:
        return jsonify({"error": "缺少 speed"}), 400
    try:
        speed = int(speed)
    except (TypeError, ValueError):
        return jsonify({"error": "speed 必須是數字"}), 400
    if not 0 <= speed <= 100:
        return jsonify({"error": "speed 必須在 0~100"}), 400
    return jsonify({"ok": True, "motor": motor.set_speed(speed)})


@app.route("/api/motor/servo", methods=["POST"])
def api_motor_servo():
    """控制雲台角度。"""
    data = request.get_json(silent=True) or {}
    x = data.get("x")
    y = data.get("y")
    return jsonify({"ok": True, "motor": motor.servo_to(x, y)})


@app.route("/api/motor/stop", methods=["POST"])
def api_motor_stop():
    """停止。"""
    return jsonify({"ok": True, "motor": motor.stop()})


@app.route("/api/motor/emergency", methods=["POST"])
def api_motor_emergency():
    """緊急停止。"""
    return jsonify({"ok": True, "motor": motor.emergency_stop()})


# ============================================================
# 啟動
# ============================================================

if __name__ == "__main__":
    print("=" * 62)
    print("  遙控車 Web 介面（純馬達控制）")
    print("=" * 62)

    # ── 嘗試切換到真實馬達模式 ──
    try:
        motor = MotorControl(mode=MODE_RPI)
        print(f"[motor] 已切換至真實馬達模式（MODE_RPI），"
              f"I2C bus：{motor.state.i2c_bus}")
    except Exception as e:
        print(f"[motor] 真實馬達初始化失敗：{e}，"
              f"將以模擬模式（MODE_SIMULATION）運行")
        motor = MotorControl(mode=MODE_SIMULATION)

    print(f"[motor] 最終模式：{motor.mode}")
    print("=" * 62)

    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n  關閉中…")
    finally:
        motor.shutdown()
