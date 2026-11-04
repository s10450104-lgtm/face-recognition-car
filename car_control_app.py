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

from pathlib import Path

from flask import Flask, jsonify, render_template, request

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
