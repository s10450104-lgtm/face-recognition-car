"""
motor_control.py — 馬達控制抽象層
==================================
包裝課程提供的 LOBOROBOT2.py，把 4 輪馬達 + 2 軸雲台統一成
與 app.py / Web 介面一致的動作介面。

兩種執行模式
------------
    模擬模式 (SIMULATION) : 在電腦上測試，只計算左右輪速度、不碰硬體
    真實模式 (RPI)        : 車上，實際驅動 LOBOROBOT 的 4 個馬達與 2 個舵機

硬體對應（來自 LOBOROBOT2.py，不可憑猜測）
------------------------------------------
    馬達 0  左前輪   PWM 通道 0    方向 IN1=2, IN2=1     ← 方向邏輯是反的
    馬達 1  右前輪   PWM 通道 5    方向 IN1=3, IN2=4
    馬達 2  左後輪   PWM 通道 6    方向 IN1=8, IN2=7
    馬達 3  右後輪   PWM 通道 11   方向走 GPIO 25 / 24（不是 PCA9685）

    舵機    通道 9  頂部（左右搖擺）
    舵機    通道 10 底座（上下俯仰）

注意
----
    馬達 0 在 LOBOROBOT2.py 裡用的是 `not forward`，代表它的接線極性
    與其他三顆相反。所有動作都必須呼叫 LOBOROBOT 提供的方法，
    不要自行拼 PWM，否則車子的轉向會整個反掉。

對應課程內容
------------
- PCA9685：16 通道 12-bit PWM，PRESCALE = 25000000/(4096*freq - 1)
- threading.Lock：Flask 多執行緒同時下指令會讓 I2C 當機
- 後退加 time.sleep() 緩啟動，避免 RPi 5 瞬時電流尖峰
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

# 模式常數
MODE_SIMULATION = "simulation"
MODE_RPI = "rpi"

# 動作名稱
ACTIONS = ("forward", "backward", "left", "right", "forward_left",
           "forward_right", "backward_left", "backward_right", "stop")

# 動作 → 中文（給 Web UI 顯示）
ACTION_LABELS = {
    "forward": "前進",
    "backward": "後退",
    "left": "原地左轉",
    "right": "原地右轉",
    "forward_left": "左平移",
    "forward_right": "右平移",
    "backward_left": "後左退",
    "backward_right": "後右退",
    "stop": "停止",
}


# ────────────────────────────────────────────────────────────────
#  I2C bus 自動偵測
# ────────────────────────────────────────────────────────────────
#  課程提供的 LOBOROBOT2.py 第 22 行寫死了：
#
#       self.bus = smbus.SMBus(1)
#
#  但 Raspberry Pi 5 上 I2C 裝置的編號不一定是 1。
#  實測這台車的 PCA9685 掛在 /dev/i2c-13，/dev/i2c-1 根本不存在，
#  直接跑會出現 FileNotFoundError。
#
#  底下提供兩個工具：
#    1. find_i2c_bus()          掃描所有 bus，找出 0x40 有回應的那條
#    2. redirected_smbus()      建構 LOBOROBOT 的當下，把 SMBus(1) 換掉
#
#  這樣完全不用改動課程提供的 LOBOROBOT2.py，
#  換一台車、bus 編號不同也一樣能跑。

PCA9685_ADDRESS = 0x40


def _available_i2c_buses() -> list[int]:
    """列出系統上存在的 I2C bus 編號。"""
    import glob
    import re

    nums = []
    for path in glob.glob("/dev/i2c-*"):
        m = re.search(r"/dev/i2c-(\d+)$", path)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def find_i2c_bus(address: int = PCA9685_ADDRESS,
                 probe: Optional[int] = None) -> tuple[Optional[int], str]:
    """
    找出接有 PCA9685 的 I2C bus。

    參數
    ----
    address : int，PCA9685 的 I2C 位址（預設 0x40）
    probe   : int | None，若指定則只試這一條 bus（不做掃描）

    回傳
    ----
    (bus 編號 | None, 說明文字)
    """
    # 環境變數優先，讓使用者可以手動指定
    import os
    env = os.environ.get("I2C_BUS", "").strip()
    if env.isdigit():
        n = int(env)
        return n, f"使用環境變數 I2C_BUS={n}"

    if probe is not None:
        buses = [probe]
    else:
        buses = _available_i2c_buses()
        if not buses:
            return None, ("系統上找不到任何 /dev/i2c-* 裝置。"
                          "請用 sudo raspi-config 開啟 I2C 後重開機。")

    try:
        import smbus
    except ImportError:
        return None, ("缺少 smbus 模組，請執行 "
                      "sudo apt install python3-smbus i2c-tools")

    tried = []
    for n in buses:
        try:
            bus = smbus.SMBus(n)
        except Exception as e:
            tried.append(f"bus {n}: 無法開啟（{type(e).__name__}）")
            continue

        try:
            # PCA9685 的 MODE1 暫存器讀得到值 = 裝置有回應
            bus.read_byte_data(address, 0x00)
            bus.close()
            return n, f"在 bus {n} 找到 0x{address:02X}"
        except Exception:
            tried.append(f"bus {n}: 0x{address:02X} 無回應")
        finally:
            try:
                bus.close()
            except Exception:
                pass

    return None, ("所有 bus 都找不到 0x%02X。\n  %s\n"
                  "  可能是馬達板沒通電，或 SDA/SCL 接線鬆脫。"
                  % (address, "\n  ".join(tried)))


@contextmanager
def redirected_smbus(bus_number: int, original_bus: int = 1):
    """
    在這段程式碼執行的期間，把 smbus.SMBus(1) 導到 bus_number。

    LOBOROBOT2.py 用的是 `import smbus` + `smbus.SMBus(1)`，
    是在呼叫时才取屬性，所以暫時換掉模組上的 SMBus 就有效，
    不需要改動課程檔案本身。
    """
    import smbus

    real_cls = smbus.SMBus

    def _factory(num=None, *args, **kwargs):
        # 只有課程檔案寫死的那個編號會被替換
        if num == original_bus:
            return real_cls(bus_number, *args, **kwargs)
        return real_cls(num, *args, **kwargs)

    smbus.SMBus = _factory
    try:
        yield
    finally:
        smbus.SMBus = real_cls


@dataclass
class MotorState:
    """馬達目前狀態，供 Web UI 輪詢。"""
    action: str = "stop"
    speed: int = 0
    left_speed: int = 0
    right_speed: int = 0
    servo_x: int = 90          # 頂部舵機（左右）0~180
    servo_y: int = 90          # 底座舵機（上下）0~180
    mode: str = MODE_SIMULATION
    command_count: int = 0
    last_command: str = ""
    last_command_time: float = field(default_factory=time.time)
    hardware_ok: bool = False  # 真實模式是否成功連上硬體
    i2c_bus: int | None = None  # 實際使用的 I2C bus 編號
    init_error: str = ""        # 初始化失敗的原因

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
            "hardware_ok": self.hardware_ok,
            "i2c_bus": self.i2c_bus,
            "init_error": self.init_error,
        }


class MotorControl:
    """
    馬達控制主類別。

    用法
    ----
        motor = MotorControl(mode=MODE_SIMULATION)   # 電腦測試
        motor = MotorControl(mode=MODE_RPI)          # 車上

        motor.move("forward", speed=50)
        motor.set_speed(70)
        motor.servo_to(x=45, y=120)
        motor.stop()
        motor.shutdown()

    執行緒安全
    ----------
    所有對硬體的操作都在 self.lock 保護下進行。
    Flask 是多執行緒環境，若沒有鎖，多個請求同時寫 I2C 會讓 PCA9685 當機
    （課堂 TROUBLE SHOOTING 提到的原因之一）。
    """

    def __init__(self, mode: str = MODE_SIMULATION,
                 default_speed: int = 50,
                 soft_start_delay: float = 0.01):
        """
        參數
        ----
        mode : str
            MODE_SIMULATION（預設，PC 測試用）或 MODE_RPI（車上用）
        default_speed : int
            前進/後退的預設速度 0~100
        soft_start_delay : float
            後退緩啟動延遲（秒）。避免 RPi 5 馬達啟動的瞬時電流尖峰
        """
        if mode not in (MODE_SIMULATION, MODE_RPI):
            raise ValueError(f"未知的模式：{mode}")

        self.mode = mode
        self.default_speed = max(0, min(100, default_speed))
        self.soft_start_delay = soft_start_delay

        # 執行緒鎖：保護所有 I2C / GPIO 操作
        self.lock = threading.Lock()

        # LOBOROBOT 實例（只有 RPI 模式才有）
        self._robot = None
        self._init_error: str = ""

        self.state = MotorState(mode=mode)

        if mode == MODE_RPI:
            self._init_hardware()

    # ── 硬體初始化 ──────────────────────────────────────
    def _init_hardware(self) -> None:
        """
        建立 LOBOROBOT 實例。

        注意：LOBOROBOT2.py 需要 smbus 與 gpiozero，且會直接對 I2C 下手。
        如果初始化失敗，記住錯誤訊息但不要讓整個程式崩潰 ——
        辨識功能應該要能獨立運作。

        I2C bus 的處理
        --------------
        LOBOROBOT2.py 裡寫死了 smbus.SMBus(1)，但樹莓派上不一定是
        bus 1（實測這台車是 bus 13）。這裡先掃描找出真正接著
        PCA9685 的 bus，再在建構當下把 SMBus(1) 導過去，
        完全不用改動課程提供的檔案。
        """
        try:
            from LOBOROBOT2 import LOBOROBOT
        except ImportError as e:
            self._init_error = (
                f"找不到 LOBOROBOT2.py（{e}）。"
                "請確認檔案與 motor_control.py 在同一個目錄。"
            )
            self.state.init_error = self._init_error
            print(f"[motor] {self._init_error}")
            return

        # ── 1. 找出正確的 I2C bus ──
        bus, why = find_i2c_bus()
        if bus is None:
            self._init_error = (
                f"找不到 PCA9685（0x{PCA9685_ADDRESS:02X}）：{why}"
            )
            self.state.init_error = self._init_error
            self.state.hardware_ok = False
            print(f"[motor] {self._init_error}")
            return

        self.state.i2c_bus = bus
        print(f"[motor] {why}（課程檔案寫死 bus 1，已導向 bus {bus}）")

        # ── 2. 在 bus 導向的情況下建構 LOBOROBOT ──
        try:
            with redirected_smbus(bus):
                self._robot = LOBOROBOT()
            self.state.hardware_ok = True
            self.state.init_error = ""
            print(f"[motor] LOBOROBOT 硬體初始化成功"
                  f"（PCA9685 @ 0x{PCA9685_ADDRESS:02X} on bus {bus}, 50Hz）")
        except Exception as e:
            self.state.hardware_ok = False
            self._init_error = (
                f"LOBOROBOT 初始化失敗：{type(e).__name__}: {e}\n"
                f"I2C bus = {bus}\n"
                "請檢查：\n"
                f"  1. bus {bus} 上看得到 0x{PCA9685_ADDRESS:02X} 嗎"
                f"（sudo i2cdetect -y {bus}）\n"
                "  2. gpiozero 是否可用：sudo apt install python3-gpiozero\n"
                "  3. GPIO 24/25 是否已被其他程式佔用"
            )
            self.state.init_error = self._init_error
            print(f"[motor] {self._init_error}")

    @property
    def hardware_available(self) -> bool:
        """真實模式下硬體是否可用。"""
        return self._robot is not None

    # ── 動作分派表 ──────────────────────────────────────
    # 對應 LOBOROBOT2.py 的方法簽名 (speed, t_time)
    def _simulated_wheels(self, action: str, speed: int) -> tuple[int, int]:
        """
        模擬模式計算左右輪速度。

        正負號：正值 = 該輪前進，負值 = 該輪後退。
        用來在沒有車子時驗證「這個動作實際上會怎麼轉」。
        """
        base, half = speed, speed // 2
        table = {
            "stop":           (0, 0),
            "forward":        (base, base),
            "backward":       (-base, -base),
            "left":           (base, -base),      # 原地左旋：左輪前、右輪後
            "right":          (-base, base),      # 原地右旋：右輪前、左輪後
            "forward_left":   (half, base),       # 左平移：右輪快 → 往左走
            "forward_right":  (base, half),       # 右平移
            "backward_left":  (-half, -base),
            "backward_right": (-base, -half),
        }
        return table[action]

    # ── 對外動作介面 ────────────────────────────────────
    def move(self, action: str, speed: Optional[int] = None,
             duration: float = 0.0) -> dict:
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
        duration : float
            持續幾秒後自動停止。0 = 持續轉動（需另外呼叫 stop()）

        回傳
        ----
        dict : 更新後的狀態
        """
        if action not in ACTIONS:
            raise ValueError(f"不支援的動作：{action}")

        spd = self.default_speed if speed is None else max(0, min(100, speed))

        with self.lock:
            if self.mode == MODE_SIMULATION:
                left, right = self._simulated_wheels(action, spd)
                self.state.left_speed = left
                self.state.right_speed = right
            else:
                self._drive(action, spd, duration)

            self.state.action = action
            self.state.speed = 0 if action == "stop" else spd
            self.state.command_count += 1
            self.state.last_command = ACTION_LABELS.get(action, action)
            self.state.last_command_time = time.time()

        return self.state.to_dict()

    def _drive(self, action: str, speed: int, duration: float) -> None:
        """
        真實模式：呼叫 LOBOROBOT2.py 的方法驅動馬達。

        所有 PWM / 方向控制都交給 LOBOROBOT，
        不要在這裡自行組 PWM 值 —— 馬達 0 的方向極性是反的，
        自己寫會讓車子往反方向跑。
        """
        if self._robot is None:
            # 硬體不可用時退化成模擬，不讓 Web 介面整個壞掉
            left, right = self._simulated_wheels(action, speed)
            self.state.left_speed = left
            self.state.right_speed = right
            return

        robot = self._robot
        t = duration if duration > 0 else 0

        # 後退緩啟動：避免 RPi 5 瞬時電流尖峰
        # （課堂 TROUBLE SHOOTING 提到的做法）
        if action == "backward" and self.soft_start_delay > 0:
            time.sleep(self.soft_start_delay)

        try:
            if action == "stop":
                robot.t_stop(t)
            elif action == "forward":
                robot.moveforward(speed, t)
            elif action == "backward":
                robot.movebackward(speed, t)
            elif action == "left":
                robot.turnLeft(speed, t)
            elif action == "right":
                robot.turnRight(speed, t)
            elif action == "forward_left":
                robot.moveLeft(speed, t)
            elif action == "forward_right":
                robot.moveRight(speed, t)
            elif action == "backward_left":
                robot.backward_Left(speed, t)
            elif action == "backward_right":
                robot.backward_Right(speed, t)
        except Exception as e:
            print(f"[motor] 驅動失敗（{action}）：{type(e).__name__}: {e}")

        # 更新狀態顯示（真實模式也要讓 Web UI 知道左右輪狀態）
        left, right = self._simulated_wheels(action, speed)
        self.state.left_speed = left
        self.state.right_speed = right

    def stop(self) -> dict:
        """停止所有馬達與舵機。"""
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
        控制雲台角度。

        參數
        ----
        x : int | None，頂部舵機（左右）角度 0~180
        y : int | None，底座舵機（上下）角度 0~180

        LOBOROBOT2.set_servo_angle() 的角度換算公式是
            pulse_width_us = angle * 11 + 500
        也就是 0° = 500µs、180° = 2480µs。
        """
        with self.lock:
            if x is not None:
                x = max(0, min(180, int(x)))
                self.state.servo_x = x
            if y is not None:
                y = max(0, min(180, int(y)))
                self.state.servo_y = y

            if self._robot is not None:
                try:
                    if x is not None:
                        self._robot.set_servo_angle(9, x, 0)
                    if y is not None:
                        self._robot.set_servo_angle(10, y, 0)
                except Exception as e:
                    print(f"[motor] 舵機設定失敗：{type(e).__name__}: {e}")

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

        if self._robot is not None:
            try:
                self._robot.t_stop(0)
            except Exception as e:
                print(f"[motor] 緊急停止失敗：{e}")

        return self.state.to_dict()

    def shutdown(self) -> None:
        """程式結束前呼叫：確保馬達停下、雲台回中。"""
        try:
            self.stop()
            if self._robot is not None:
                time.sleep(0.05)
                self._robot.set_servo_angle(9, 90, 0)
                self._robot.set_servo_angle(10, 90, 0)
        except Exception:
            pass


if __name__ == "__main__":
    """自我測試。"""
    import sys

    print("=" * 66)
    print("  馬達控制測試")
    print("=" * 66)

    # 從命令列決定模式：python motor_control.py rpi
    mode = MODE_RPI if (len(sys.argv) > 1 and sys.argv[1] == "rpi") else MODE_SIMULATION
    m = MotorControl(mode=mode, default_speed=60)

    print(f"\n模式：{m.mode}")
    if mode == MODE_RPI:
        print(f"硬體可用：{m.hardware_available}")
        if not m.hardware_available:
            print(f"錯誤：{m._init_error}")
            print("\n硬體未就緒，改用模擬模式測試。")
            m = MotorControl(mode=MODE_SIMULATION, default_speed=60)

    print()
    print(f"{'動作':<14s}{'左輪':>8s}{'右輪':>8s}{'速度':>8s}")
    print("-" * 66)

    for act in ACTIONS:
        s = m.move(act)
        print(f"{s['label']:<14s}{s['left_speed']:>8d}{s['right_speed']:>8d}{s['speed']:>8d}")

    print()
    s = m.set_speed(30)
    print(f"速度調整後：{s['label']}  速度={s['speed']}  "
          f"左={s['left_speed']} 右={s['right_speed']}")

    s = m.servo_to(45, 135)
    print(f"雲台角度：頂部(左右) X={s['servo_x']}°  底座(上下) Y={s['servo_y']}°")

    s = m.emergency_stop()
    print(f"緊急停止：{s['last_command']}  速度={s['speed']}")

    m.shutdown()
    print(f"\n累計指令數：{s['command_count']}")
    print()
    if mode == MODE_RPI:
        print("[OK] 真實模式測試結束（請確認車子已停止）")
    else:
        print("[OK] 模擬模式全部動作測試通過")
        print("     車上執行：python motor_control.py rpi")