"""
app.py — Flask Web 主程式
===========================
把「人臉辨識 + 遙控」整合成網頁介面。

架構
----
    背景執行緒：攝影機擷取 → HAAR 偵測 → LBPH 辨識 → 畫面標註
                              ↓
                  最新幀存到共用變數（用 Lock 保護）
                              ↓
    Flask 路由：  /video_feed  以 MJPEG 串流輸出
                 /api/*       JSON 狀態與控制指令

為什麼用背景執行緒
------------------
如果把攝影機迴圈直接寫在請求處理裡，每個頁面請求都會重新開一次
攝影機，而且會阻塞其他請求。改成單一背景執行緒持續擷取，
HTTP 請求只負責「取最新的那一幀」，回應速度才會穩定。

效能考量（對應課程 TROUBLE SHOOTING）
-------------------------------------
- 辨識用 imgsz 降解析度 + 跳幀，避免 CPU 負載過高
- 影像串流 (/video_feed) 與辨識解耦，辨識慢不會卡住串流
- 所有共用狀態都用 threading.Lock 保護
"""

from __future__ import annotations

import io
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, render_template,
                   request, stream_with_context)

# face_core 必須先於 cv2 匯入（設定 OPENCV_LOG_LEVEL 與主控台編碼）
import face_core as fc
from camera_source import open_camera
from motor_control import (ACTIONS, MODE_RPI, MODE_SIMULATION,
                           MotorControl)
from yolo_detector import YoloDetector, is_available as yolo_available

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# Flask 3.x 起 JSON_AS_ASCII 設定已移除，改用 app.json 物件。
# 不關掉的話回傳的中文會變成 \uXXXX，前端顯示會很難閱讀。
app.json.ensure_ascii = False


# ============================================================
# 全域狀態
# ============================================================

class SystemState:
    """
    系統共用狀態，所有欄位都以 lock 保護。

    多執行緒安全的重要性：Flask 預設多執行緒，背景擷取執行緒與
    HTTP 請求執行緒會同時存取這些欄位。
    """

    def __init__(self):
        self.lock = threading.Lock()

        # 影像
        self.frame: np.ndarray | None = None      # 已標註的彩色畫面
        self.raw_frame: np.ndarray | None = None  # 未標註原畫面（註冊用）
        self.frame_seq = 0
        self.fps = 0.0

        # 辨識
        self.face_results: list[dict] = []
        self.recognizer = None
        self.labels: list[str] = []
        self.model_loaded = False

        # 註冊任務
        self.register_task: dict | None = None

        # YOLO 物件偵測（由獨立背景執行緒更新，這裡只讀）
        self.ai_results: list[dict] = []
        self.ai_enabled = False
        self.ai_error: str | None = None
        self.ai_thread: threading.Thread | None = None

        # 執行緒
        self.running = False
        self.thread: threading.Thread | None = None

    def set_model(self, recognizer, labels: list[str]):
        with self.lock:
            self.recognizer = recognizer
            self.labels = labels
            self.model_loaded = True

    def clear_model(self):
        with self.lock:
            self.recognizer = None
            self.labels = []
            self.model_loaded = False


state = SystemState()
motor = MotorControl(mode=MODE_SIMULATION)

# 設定參數
CAMERA_INDEX = 0
STREAM_WIDTH = 640
STREAM_HEIGHT = 480
DETECT_EVERY_N_FRAMES = 1      # 每 N 幀辨識一次；車上建議 2~3

# ── YOLO 物件偵測 ──────────────────────────────────────────
# 樹莓派上 YOLOv8n 推論要 90ms 以上，若直接塞進串流迴圈會把
# 影像串流拖到 5 FPS 以下。所以改成獨立背景執行緒推論，
# 串流迴圈只負責把結果畫上去，兩者用 state 交換資料。
#
# 想在沒有安裝 ultralytics 的機器上執行，可以設成 False。
YOLO_ENABLED = True
YOLO_IMGSZ = 320              # 實測 640→5 FPS、320→11 FPS
YOLO_CONF = 0.30
YOLO_SKIP_FRAMES = 3          # 每 3 幀推論一次，中間沿用上次的結果

# YOLO 偵測器（單一實例，供背景執行緒與繪圖共用）
yolo = YoloDetector(imgsz=YOLO_IMGSZ,
                    conf=YOLO_CONF,
                    skip_frames=YOLO_SKIP_FRAMES)

# 串流畫面用的框線配色（BGR）
_AI_COLORS = {
    "person": (0, 255, 255),     # 黃
    "bicycle": (255, 0, 255),    # 紫紅
    "car": (255, 180, 0),        # 藍
    "motorcycle": (0, 255, 100), # 綠
    "bus": (0, 200, 255),        # 橘
}


# ============================================================
# 背景擷取執行緒
# ============================================================

def capture_loop():
    """
    背景執行緒：持續擷取 → 辨識 → 標註 → 更新 state.frame

    這是整個系統的心跳。任何例外都不能讓這個執行緒死掉，
    否則網頁會一直顯示最後一幀而沒有錯誤提示。
    """
    print("[capture] 啟動擷取執行緒…")

    cam = None
    detector = fc.build_face_detector()
    frame_times: list[float] = []

    while state.running:
        try:
            # ── 確保攝影機連線 ──
            if cam is None:
                try:
                    cam = open_camera(index=CAMERA_INDEX,
                                      frame_size=(STREAM_WIDTH, STREAM_HEIGHT))
                    print(f"[capture] 攝影機就緒：{cam.name}")
                except RuntimeError as e:
                    print(f"[capture] 攝影機開啟失敗：{e}")
                    time.sleep(2)
                    continue

            ok, frame = cam.read()
            if not ok:
                print("[capture] 讀取失敗，重新連線…")
                cam.release()
                cam = None
                time.sleep(1)
                continue

            # 方向校正已由 camera_source.apply_orientation() 統一處理，
            # 這裡不要再 flip，否則會翻兩次
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── 註冊模式：只擷取不辨識 ──
            with state.lock:
                task = state.register_task

            if task is not None:
                faces = fc.detect_faces(gray, detector)

                if len(faces):
                    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

                    if w < fc.MIN_FACE_SIZE:
                        cv2.putText(vis_tmp := frame.copy(),
                                    f"FACE TOO SMALL ({w}px)",
                                    (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 165, 255), 2)
                        _publish(vis_tmp, frame, [], frame_times)
                        time.sleep(0.05)
                        continue

                    roi = fc.normalize_face(
                        fc.crop_face(gray, (x, y, w, h), out_size=200))

                    # 檔名必須依「目前已擷取數量」即時計算。
                    # 若只在開始時算一次，之後每張都會覆寫同一個檔案。
                    n = task["count"] + 1
                    path = Path(task["dir"]) / f"{task['label']}_{n:03d}.jpg"

                    if fc.imwrite_safe(path, roi):
                        with state.lock:
                            # 再次確認任務仍是同一個，避免寫到已被取代的任務
                            if state.register_task is task:
                                task["count"] = n
                                if n >= task["_need"]:
                                    finished = task["label"]
                                    state.register_task = None
                                else:
                                    finished = None
                            else:
                                finished = None
                                n = task["count"]

                        if finished:
                            print(f"[register] 「{finished}」擷取完成 "
                                  f"({task['_need']} 張)", flush=True)
                        else:
                            print(f"[register] 已擷取 {n}/{task['_need']}", flush=True)
                    else:
                        print("[register] 寫檔失敗", flush=True)

                vis = frame.copy()
                cv2.putText(vis, "REGISTERING", (12, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                _publish(vis, frame, [], frame_times)
                time.sleep(0.05)
                continue

            # ── 一般模式：辨識 ──
            if state.frame_seq % DETECT_EVERY_N_FRAMES == 0:
                with state.lock:
                    rec = state.recognizer
                    labels = state.labels

                if rec is not None:
                    results = fc.identify_faces(gray, detector, rec, labels)
                else:
                    # 沒模型時只偵測不辨識，讓使用者看得出偵測有沒有運作
                    results = [
                        {"rect": r, "name": "NoModel", "distance": 0.0,
                         "is_known": False}
                        for r in fc.detect_faces(gray, detector)
                    ]

                vis = frame.copy()
                vis = fc.draw_result(vis, results)
                vis = _draw_ai(vis)
                vis = fc.draw_hud(
                    vis, "FACE RECOGNITION",
                    f"{len(labels)} registered" if labels else "no model")
                _publish(vis, frame, results, frame_times)
            else:
                with state.lock:
                    last = state.frame
                _publish(last if last is not None else frame, frame, state.face_results,
                         frame_times)

        except Exception as e:
            # 單幀出錯不應該讓整個擷取執行緒掛掉
            print(f"[capture] 擷取例外：{type(e).__name__}: {e}")
            time.sleep(0.5)

    if cam is not None:
        cam.release()
    print("[capture] 擷取執行緒已停止")


def _draw_ai(vis: np.ndarray) -> np.ndarray:
    """
    把 YOLO 的偵測結果畫到串流畫面上。

    結果是從 state.ai_results 讀的（由 yolo_loop 背景執行緒更新），
    所以這裡只是單純畫線，不會拖慢串流。
    """
    if not YOLO_ENABLED:
        return vis

    with state.lock:
        detections = list(state.ai_results)

    if not detections:
        return vis

    # 直接餵給 YoloDetector.draw()，但它會用 detector 內部的
    # results 而不是我們手上的 detections，所以改成手動畫。
    for d in detections:
        x1, y1, x2, y2 = d["rect"]
        color = _AI_COLORS.get(d["class"], (0, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"{d['cn']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.45, 2)
        ty = y1 - 6 if y1 - th - 10 > 0 else y1 + th + 8
        cv2.rectangle(vis, (x1, ty - th - 6), (x1 + tw + 8, ty + 4),
                      color, -1)
        cv2.putText(vis, label, (x1 + 4, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 2,
                    cv2.LINE_AA)

    return vis


def yolo_loop():
    """
    背景執行緒：讀取共用畫面 → YOLO 推論 → 寫回 state.ai_results

    為什麼要獨立一個執行緒？
    ----------------------
    樹莓派上 YOLOv8n 單次推論要 90ms 以上。如果在 capture_loop
    裡同步推論，影像串流會直接掉到 5 FPS 以下（實測）。
    拆成獨立執行緒後，串流維持 20 FPS，推論在旁邊慢慢跑，
    前端透過 /api/ai_status 另外抓結果（簡報建議的
    「非同步狀態更新」）。

    推論結果不是即時的：最多落後 skip_frames 幀，
    對「畫面上有偵測到什麼」這種需求完全夠用。
    """
    print("[yolo-loop] 啟動 YOLO 背景執行緒…")

    if not YOLO_ENABLED:
        state.ai_error = "已停用（YOLO_ENABLED = False）"
        print("[yolo-loop] YOLO 已停用")
        return

    if not yolo_available():
        state.ai_error = yolo.load_error or "尚未安裝 ultralytics"
        print(f"[yolo-loop] 環境不具備：{state.ai_error}")
        return

    if not yolo.load():
        state.ai_error = yolo.load_error
        print("[yolo-loop] 模型載入失敗，改為停用")
        return

    with state.lock:
        state.ai_enabled = True
        state.ai_error = None

    n = 0
    while state.running:
        try:
            with state.lock:
                frame = state.raw_frame

            if frame is None:
                time.sleep(0.1)
                continue

            n += 1
            # tick() 內部會依 skip_frames 判斷要不要真的推論
            detections = yolo.tick(frame)

            with state.lock:
                state.ai_results = detections

        except Exception as e:
            print(f"[yolo-loop] 例外：{type(e).__name__}: {e}")
            time.sleep(0.5)

    with state.lock:
        state.ai_enabled = False
    print("[yolo-loop] 已停止")


def _publish(vis: np.ndarray, raw: np.ndarray, results: list[dict],
             frame_times: list[float]):
    """把這一幀寫入共用狀態（含 FPS 計算）。"""
    now = time.time()
    frame_times.append(now)
    if len(frame_times) > 15:
        frame_times.pop(0)

    with state.lock:
        state.frame = vis
        state.raw_frame = raw
        state.face_results = results
        state.frame_seq += 1
        if len(frame_times) > 5:
            span = frame_times[-1] - frame_times[0]
            state.fps = (len(frame_times) - 1) / span if span > 0 else 0.0


def start_capture():
    """啟動背景擷取執行緒與 YOLO 執行緒（冪等）。"""
    if state.thread and state.thread.is_alive():
        return
    state.running = True

    state.thread = threading.Thread(target=capture_loop, daemon=True,
                                    name="capture")
    state.thread.start()

    # YOLO 背景執行緒獨立啟動。模型載入要一點時間，
    # 不能擋住串流，所以這邊不 join。
    if state.ai_thread is None or not state.ai_thread.is_alive():
        state.ai_thread = threading.Thread(target=yolo_loop, daemon=True,
                                           name="yolo")
        state.ai_thread.start()


def stop_capture():
    """停止背景執行緒。"""
    state.running = False
    if state.thread:
        state.thread.join(timeout=3)
    if state.ai_thread:
        state.ai_thread.join(timeout=3)


# ============================================================
# 影像串流
# ============================================================

def mjpeg_frame() -> bytes:
    """把最新一幀編碼成 JPEG。"""
    with state.lock:
        frame = state.frame

    if frame is None:
        # 還沒準備好，先回一張提示圖
        blank = np.full((STREAM_HEIGHT, STREAM_WIDTH, 3), 25, np.uint8)
        cv2.putText(blank, "INITIALIZING CAMERA...", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        frame = blank

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return b""
    return buf.tobytes()


@app.route("/video_feed")
def video_feed():
    """MJPEG 影像串流。瀏覽器把這個當成 <img src> 即可。"""
    def generate():
        while True:
            if not state.running:
                break
            frame = mjpeg_frame()
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                       + frame + b"\r\n")
            time.sleep(0.033)      # 約 30 FPS

    return Response(stream_with_context(generate()), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_snapshot")
def video_snapshot():
    """取得單張 JPEG（用於截圖或離線分析）。"""
    with state.lock:
        frame = state.frame if state.frame is not None else state.raw_frame
    if frame is None:
        return jsonify({"error": "camera not ready"}), 503
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return jsonify({"error": "encode failed"}), 500
    return Response(buf.tobytes(), mimetype="image/jpeg")


# ============================================================
# 頁面
# ============================================================

@app.route("/")
def index():
    """主頁面。"""
    return render_template("index.html",
                           actions=ACTIONS,
                           mode=motor.mode)


# ============================================================
# API：狀態查詢
# ============================================================

@app.route("/api/status")
def api_status():
    """系統總狀態（前端輪詢）。"""
    with state.lock:
        faces = [
            {"name": r["name"],
             "distance": round(float(r["distance"]), 1) if np.isfinite(r["distance"]) else None,
             "is_known": r["is_known"],
             "rect": [int(v) for v in r["rect"]]}
            for r in state.face_results
        ]
        payload = {
            "running": state.running,
            "fps": round(state.fps, 1),
            "frame_seq": state.frame_seq,
            "model_loaded": state.model_loaded,
            "labels": list(state.labels),
            "faces": faces,
            "face_count": len(faces),
            "known_count": sum(1 for f in faces if f["is_known"]),
            "threshold_strong": fc.THRESHOLD_STRONG,
            "threshold_weak": fc.THRESHOLD_WEAK,
            "register_task": (
                {"label": state.register_task["label"],
                 "count": state.register_task["count"],
                 "target": state.register_task.get("_need", 30)}
                if state.register_task else None
            ),
        }
    payload["motor"] = motor.state.to_dict()

    # YOLO 物件偵測狀態（前端單獨顯示）
    payload["ai"] = {
        "enabled": state.ai_enabled,
        "error": state.ai_error,
        "count": len(state.ai_results),
        "imgsz": YOLO_IMGSZ,
        "skip_frames": YOLO_SKIP_FRAMES,
    }
    return jsonify(payload)


@app.route("/api/ai_status")
def api_ai_status():
    """
    YOLO 物件偵測的詳細狀態。

    刻意跟 /api/status 分開：推論結果變動較慢，
    前端不需要每秒重抓一次全部資料。
    """
    data = yolo.status()
    with state.lock:
        data["thread_running"] = bool(state.ai_thread
                                       and state.ai_thread.is_alive())
        data["detections"] = list(state.ai_results)
        data["count"] = len(state.ai_results)
    return jsonify(data)


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
# API：人臉註冊與訓練
# ============================================================

@app.route("/api/register/start", methods=["POST"])
def api_register_start():
    """
    開始線上註冊：從攝影機自動擷取 N 張人臉照片。

    使用者不必開終端機跑 collect_faces.py，直接在網頁輸入名字即可。
    """
    data = request.get_json(silent=True) or {}
    label = (data.get("label") or "").strip()
    count = int(data.get("count", 30))

    if not label:
        return jsonify({"error": "請輸入名稱"}), 400

    bad = set('<>:"/\\|?*')
    if any(c in bad for c in label):
        return jsonify({"error": "名稱不能包含 < > : \" / \\ | ? *"}), 400

    count = max(5, min(count, 200))

    with state.lock:
        if state.register_task is not None:
            return jsonify({"error": "已經有註冊任務在進行中"}), 409

        out_dir = fc.DATA_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)

        # 從現有數量繼續編號（增量註冊，不用清掉舊資料）
        existing = len(list(out_dir.glob("*.jpg")))

        state.register_task = {
            "label": label,
            "count": existing,          # 直接從既有張數接續
            "_need": existing + count,  # 目標總數 = 既有 + 本次要拍
            "dir": str(out_dir),
        }

    print(f"[register] 開始註冊「{label}」，目標 {count} 張"
          f"（目前已有 {existing} 張）", flush=True)

    return jsonify({"ok": True, "label": label, "target": count,
                    "existing": existing,
                    "message": "請看著鏡頭，保持不動"})


@app.route("/api/register/cancel", methods=["POST"])
def api_register_cancel():
    """取消註冊任務。"""
    with state.lock:
        cancelled = state.register_task is not None
        state.register_task = None
    return jsonify({"ok": True, "cancelled": cancelled})


@app.route("/api/train", methods=["POST"])
def api_train():
    """
    用 data/ 目前所有照片重新訓練模型。

    註冊完之後呼叫這個，模型就會包含新的人。
    """
    samples = fc.collect_samples(fc.DATA_DIR)
    if not samples:
        return jsonify({"error": "data/ 沒有任何照片"}), 400

    try:
        # 訓練是耗時操作，放到背景執行緒避免阻塞請求
        def do_train():
            try:
                print("[train] 開始重新訓練…")
                rec, _, label_map = fc.train_lbph_model(samples, fc.MODEL_DIR)
                state.set_model(rec, label_map)
                print(f"[train] 完成：{label_map}")
            except Exception as e:
                print(f"[train] 失敗：{e}")

        threading.Thread(target=do_train, daemon=True).start()
        return jsonify({"ok": True,
                        "message": "開始重新訓練，請稍候…",
                        "people": {k: len(v) for k, v in samples.items()}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    """從 models/ 載入既有模型。"""
    try:
        rec, labels = fc.load_trained_model()
        state.set_model(rec, labels)
        return jsonify({"ok": True, "labels": labels})
    except FileNotFoundError as e:
        state.clear_model()
        return jsonify({"error": str(e)}), 404


@app.route("/api/people")
def api_people():
    """列出已註冊的人與照片數量。"""
    samples = fc.collect_samples(fc.DATA_DIR)
    return jsonify({
        "people": [
            {"label": k, "count": len(v)}
            for k, v in sorted(samples.items())
        ]
    })


@app.route("/api/delete/<label>", methods=["POST"])
def api_delete(label):
    """刪除某個人的所有照片（並清掉模型）。"""
    import shutil
    d = fc.DATA_DIR / label
    if not d.exists():
        return jsonify({"error": f"找不到 {label}"}), 404
    shutil.rmtree(d)
    state.clear_model()
    return jsonify({"ok": True, "deleted": label,
                    "message": "已刪除，請重新訓練模型"})


# ============================================================
# 啟動
# ============================================================

def load_model_on_startup():
    """啟動時嘗試載入既有模型。"""
    try:
        rec, labels = fc.load_trained_model()
        state.set_model(rec, labels)
        print(f"[startup] 已載入模型：{len(labels)} 人 → {labels}")
    except FileNotFoundError:
        print("[startup] 尚無模型，請先註冊人臉後訓練")


if __name__ == "__main__":
    print("=" * 62)
    print("  AI 自走車遙控 + 人臉辨識 Web 介面")
    print(f"  模式：{motor.mode}（{'模擬，指令只記錄不驅動硬體' if motor.mode == MODE_SIMULATION else '真實馬達'}）")
    print("=" * 62)

    load_model_on_startup()
    start_capture()

    try:
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n  關閉中…")
    finally:
        stop_capture()
        motor.shutdown()