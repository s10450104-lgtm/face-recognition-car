"""
motor_control.py — 馬達控制抽象層
==================================
同一份介面，兩種執行模式：

    模擬模式 (SIMULATION) : 目前在電腦上測試。只記錄指令、不接硬體
    真實模式 (RPI)        : 上車後用 I2C + PCA9685 驅動實際馬達

為什麼要這樣分
--------------
現在還不能用车，但 Web 介面、執行緒結構、API 路由都要先做好並測試。
把「硬體相依」限制在這一個檔案裡，其他模組完全不碰 I2C，
上車時只要換 backend，不必重寫 app.py。

對應課程進度
------------
- I2C / PCA9685：16 通道 12-bit PWM，PRESCALE 公式與寄存器操作
- threading.Lock：Flask 是多執行緒環境，多個請求同時下指令會讓 I2C 當機
- 後退加 time.sleep() 緩啟動，避免 RPi 5 瞬時電流尖峰
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

# 模式常數
MODE_SIMULATION = "simulation"
MODE_RPI = "rpi"

# 合法動作
ACTIONS = ("forward", "backward", "left", "right", "forward_left",
           "forward_right", "backward_left", "backward_right", "stop")

# 動作 → 中文（給 Web UI 顯示）
ACTION_LABELS = {
    "forward": "前進",
    "backward": "後退",
    "left": "左轉",
    "right": "右轉",
    "forward_left": "前左轉",
    "forward_right": "前右轉",
    "backward_left": "後左轉",
    "backward_right": "後右轉",
    "stop": "停止",
}


@dataclass
class MotorState:
    """馬達目前狀態，供 Web UI 輪詢。"""
    action: str = "stop"
    speed: int = 0
    left_speed: int = 0
    right_speed: int = 0
    servo_x: int = 90          # 伺服馬達水平角度 (0~180)
    servo_y: int = 90          # 伺服馬達垂直角度 (0~180)
    mode: str = MODE_SIMULATION
    command_count: int = 0
    last_command: str = ""
    last_command_time: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "label": ACTION_LABELS.get(self.action, self.action),
            "speed": self.speed,
            "left_speed": self.left_speed,
            "right_speed": self.right_speed,
            "servo_x": self.servo_x,
            "servo_y": self.servo_y,
            "mode": self.mode,
            "command_count": self.command_count,
            "last_command": self.last_command,
            "last_command_time": self.last_command_time,
        }


class MotorControl:
    """
    馬達控制主類別。

    用法
    ----
        motor = MotorControl(mode=MODE_SIMULATION)
        motor.move("forward")
        motor.set_speed(60)
        motor.servo_to(45, 120)
        motor.stop()

    執行緒安全
    ----------
    所有對硬體的操作都在 self.lock 保護下進行。
    Flask 多執行緒同時送指令時，若沒有鎖，會有多個執行緒同時
    寫 I2C，導致 PCA9685 當機（課堂 TROUBLE SHOOTING 提到的原因之一）。
    """

    # PCA9685 I2C 位址（課程簡報：預設 0x40）
    PCA9685_ADDR = 0x40
    PCA9685_FREQ = 50          # 伺服馬達用 50Hz
    PRESCALE = 121             # 25000000 / (4096 * 50 - 1) ≈ 121

    # 馬達通道（左/右/攝影頭雲台 俯仰/方位）
    CH_LEFT = 0
    CH_RIGHT = 1
    CH_SERVO_Y = 2
    CH_SERVO_X = 3

    # 速度範圍（0~100，對應 PCA9685 的 0~4095 duty）
    MIN_PWM = 0
    MAX_PWM = 4095

    def __init__(self, mode: str = MODE_SIMULATION,
                 default_speed: int = 50,
                 soft_start_delay: float = 0.01):
        """
        參數
        ----
        mode : str
            MODE_SIMULATION（預設，PC 測試用）或 MODE_RPI（上車用）
        default_speed : int
            前進/後退時的預設速度 0~100
        soft_start_delay : float
            後退緩啟動延遲（秒）。避免 RPi 5 馬達啟動的瞬時電流尖峰
        """
        if mode not in (MODE_SIMULATION, MODE_RPI):
            raise ValueError(f"未知的模式：{mode}")

        self.mode = mode
        self.default_speed = max(0, min(100, default_speed))
        self.soft_start_delay = soft_start_delay

        # 執行緒鎖：保護所有 I2C 操作
        self.lock = threading.Lock()

        # 硬體句柄（只有 RPI 模式才有）
        self._bus = None
        self._pca = None
        self._warned = False

        self.state = MotorState(mode=mode)

        if mode == MODE_RPI:
            self._init_hardware()

    # ── 硬體初始化 ──────────────────────────────────────
    def _init_hardware(self) -> None:
        """初始化 I2C 與 PCA9685（僅真實模式）。"""
        try:
            import board
            import adafruit_pca9685
            from adafruit_blinka.typing.wait import wait_for_blocks
        except ImportError as e:
            raise RuntimeError(
                "真實模式需要安裝：\n"
                "  sudo apt install python3-smbus i2c-tools\n"
                "  pip install adafruit-circuitpython-pca9685 gpiozero\n"
                f"（錯誤：{e}）"
            ) from e

        try:
            self._bus = board.I2C()
            self._pca = adafruit_pca9685.PCA9685(self._bus, self.PCA9685_ADDR)
            self._pca.frequency = self.PCA9685_FREQ
            wait_for_blocks()      # 等待 OSC 穩定
            # PCA9685 開機預設是睡眠模式，必須喚醒才會輸出 PWM
            time.sleep(0.01)
            print(f"[motor] PCA9685 已初始化 @ 0x{self.PCA9685_ADDR:02X}, "
                  f"{self.PCA9685_FREQ}Hz")
        except Exception as e:
            raise RuntimeError(
                f"PCA9685 初始化失敗：{e}\n"
                "請檢查：\n"
                "  1. I2C 是否啟用：sudo raspi-config → Interface Options → I2C\n"
                "  2. 模組接線是否正確（SDA/SCL/VCC/GND）\n"
                "  3. sudo i2cdetect -y 1 看是否看到 0x40"
            ) from e

    # ── PWM 底層輸出 ────────────────────────────────────
    def _set_channel(self, channel: int, duty: float) -> None:
        """
        設定單一 PWM 通道的 duty cycle。

        參數
        ----
        channel : int，0~15
        duty    : float，0.0~1.0（1.0 = 全速）
        """
        if not 0.0 <= duty <= 1.0:
            raise ValueError(f"duty 必須在 0~1 之間：{duty}")

        value = int(self.MAX_PWM * duty)

        # PCA9685 每個通道 4 個暫存器：
        #   LEDx_ON_L  = 0x06 + 4*x   LEDx_ON_H  = 0x07 + 4*x
        #   LEDx_OFF_L = 0x08 + 4*x   LEDx_OFF_H = 0x09 + 4*x
        # ON_L = 0（全亮起始）、OFF_L/H = duty（全亮結束）
        on_l = 0x06 + 4 * channel
        off_l = on_l + 2

        try:
            self._bus.write_byte_data(self.PCA9685_ADDR, on_l, 0)
            self._bus.write_byte_data(self.PCA9685_ADDR, off_l, value & 0xFF)
            self._bus.write_byte_data(
                self.PCA9685_ADDR, off_l + 1, (value >> 8) & 0x0F)
        except Exception as e:
            self._warn_once(f"寫入 PWM 失敗：{e}")

    def _warn_once(self, msg: str) -> None:
        """錯誤只提示一次，避免刷版。"""
        if not self._warned:
            print(f"[motor][WARN] {msg}")
            self._warned = True

    # ── 對外動作介面 ────────────────────────────────────
    def move(self, action: str, speed: Optional[int] = None) -> dict:
        """
        執行一個動作。

        參數
        ----
        action : str
            forward / backward / left / right /
            forward_left / forward_right /
            backward_left / backward_right / stop
        speed : int | None
            0~100，None 代表用 default_speed

        回傳
        ----
        dict : 更新後的狀態
        """
        if action not in ACTIONS:
            raise ValueError(f"不支援的動作：{action}")

        spd = self.default_speed if speed is None else max(0, min(100, speed))

        with self.lock:
            if self.mode == MODE_SIMULATION:
                self._simulate(action, spd)
            else:
                self._drive(action, spd)

            self.state.action = action
            self.state.speed = 0 if action == "stop" else spd
            self.state.command_count += 1
            self.state.last_command = ACTION_LABELS.get(action, action)
            self.state.last_command_time = time.time()

        return self.state.to_dict()

    def _simulate(self, action: str, speed: int) -> None:
        """
        模擬模式：只計算左右輪速度，不碰硬體。

        之所以還要算左右輪速度，是為了讓 Web UI 能顯示
        「這顆指令實際上會怎麼轉」，驗證 PID / 差速控制的邏輯。

        正負號的意義：正值 = 該輪正轉，負值 = 該輪反轉。
        """
        base, half = speed, speed // 2
        left, right = self._wheel_targets(action, base, half)

        self.state.left_speed = left
        self.state.right_speed = right

    @staticmethod
    def _wheel_targets(action: str, base: int, half: int) -> tuple[int, int]:
        """
        動作 → (左輪速度, 右輪速度)，正負號代表轉向。

        這是模擬與真實模式共用的唯一對照表，
        兩邊共用才能確保 Web UI 顯示的與實際車子行為一致。

        差速原理：讓左右輪不同速就能轉彎，兩輪反向就能原地旋轉。
        """
        table = {
            "stop":           (0, 0),
            "forward":        (base, base),
            "backward":       (-base, -base),
            "left":           (-base, base),       # 原地左旋
            "right":          (base, -base),       # 原地右旋
            "forward_left":   (half, base),        # 左輪減速 → 左彎
            "forward_right":  (base, half),        # 右輪減速 → 右彎
            "backward_left":  (-base, -half),
            "backward_right": (-half, -base),
        }
        return table[action]

    def _drive(self, action: str, speed: int) -> None:
        """真實模式：實際驅動馬達（僅樹莓派）。"""
        if action == "stop":
            self._set_channel(self.CH_LEFT, 0.0)
            self._set_channel(self.CH_RIGHT, 0.0)
            return

        base, half = speed, speed // 2
        left, right = self._wheel_targets(action, base, half)

        self.state.left_speed = left
        self.state.right_speed = right

        # 後退緩啟動：避免 RPi 5 瞬時電流尖峰
        # （課堂 TROUBLE SHOOTING 提到的做法）
        if action.startswith("backward") and self.soft_start_delay > 0:
            time.sleep(self.soft_start_delay)

        # PCA9685 只認絕對值；方向由另外的 GPIO 方向腳位決定
        self._set_channel(self.CH_LEFT, abs(left) / 100.0)
        self._set_channel(self.CH_RIGHT, abs(right) / 100.0)

    def stop(self) -> dict:
        """停止所有馬達。"""
        return self.move("stop", 0)

    def set_speed(self, speed: int) -> dict:
        """調整速度，並用當前動作重新輸出。"""
        speed = max(0, min(100, speed))
        self.default_speed = speed
        if self.state.action != "stop":
            return self.move(self.state.action, speed)
        self.state.speed = speed
        return self.state.to_dict()

    def servo_to(self, x: Optional[int] = None, y: Optional[int] = None) -> dict:
        """
        控制攝影頭雲台角度。

        參數
        ----
        x : int | None，水平角度 0~180（90 = 正中間）
        y : int | None，垂直角度 0~180

        伺服馬達用 50Hz PWM，duty = 角度 / 180
        """
        with self.lock:
            if x is not None:
                x = max(0, min(180, int(x)))
                self.state.servo_x = x
            if y is not None:
                y = max(0, min(180, int(y)))
                self.state.servo_y = y

            if self.mode == MODE_RPI:
                self._set_channel(self.CH_SERVO_X, self.state.servo_x / 180.0)
                self._set_channel(self.CH_SERVO_Y, self.state.servo_y / 180.0)

        return self.state.to_dict()

    def emergency_stop(self) -> dict:
        """
        緊急停止。

        刻意不使用 self.lock：緊急狀況下寧可搶佔也要立刻停，
        不能因為另一個執行緒正在操作 I2C 而卡住。
        """
        self.state.action = "stop"
        self.state.speed = 0
        self.state.left_speed = 0
        self.state.right_speed = 0
        self.state.command_count += 1
        self.state.last_command = "緊急停止"
        self.state.last_command_time = time.time()

        if self.mode == MODE_RPI:
            try:
                self._set_channel(self.CH_LEFT, 0.0)
                self._set_channel(self.CH_RIGHT, 0.0)
            except Exception:
                pass

        return self.state.to_dict()

    def shutdown(self) -> None:
        """程式結束前呼叫：確保馬達停下。"""
        try:
            self.stop()
            if self.mode == MODE_RPI and self._bus is not None:
                time.sleep(0.05)
                self._set_channel(self.CH_LEFT, 0.0)
                self._set_channel(self.CH_RIGHT, 0.0)
                self._set_channel(self.CH_SERVO_Y, 0.5)   # 雲台回中
                self._set_channel(self.CH_SERVO_X, 0.5)
        except Exception:
            pass


if __name__ == "__main__":
    """自我測試：模擬模式下所有動作都能跑。"""
    print("=" * 58)
    print("  馬達控制模擬測試 (SIMULATION)")
    print("=" * 58)

    m = MotorControl(mode=MODE_SIMULATION, default_speed=60)

    print(f"\n模式：{m.mode}\n")
    print(f"{'動作':<16s}{'左輪':>8s}{'右輪':>8s}{'速度':>8s}")
    print("-" * 58)

    for act in ACTIONS:
        s = m.move(act)
        print(f"{s['label']:<16s}{s['left_speed']:>8d}{s['right_speed']:>8d}{s['speed']:>8d}")

    print()
    s = m.set_speed(30)
    print(f"調整速度後：{s['label']}  速度={s['speed']}  "
          f"左={s['left_speed']} 右={s['right_speed']}")

    s = m.servo_to(45, 135)
    print(f"雲台角度：X={s['servo_x']}  Y={s['servo_y']}")

    s = m.emergency_stop()
    print(f"緊急停止：{s['label']}  速度={s['speed']}")

    print(f"\n累計指令數：{s['command_count']}")
    print("\n[OK] 模擬模式全部動作測試通過")
    print("     上車後改用 MotorControl(mode=MODE_RPI) 即接真實馬達")