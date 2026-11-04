"""
check_motor.py — 馬達硬體自我檢查
=================================
在車上執行，用來確認「PCA9685 有沒有通電」這件事。

為什麼需要這個？
--------------
LOBOROBOT2.py 裡寫死了 smbus.SMBus(1)，但樹莓派的 I2C 編號不一定是 1。
而且如果馬達板沒上電，錯誤訊息會是令人費解的 FileNotFoundError
或 Remote I/O error，看不出到底哪裡出問題。

這個腳本會做三件事：
    1. 列出系統上所有 I2C bus
    2. 逐條 bus 掃描，找出有回應的裝置
    3. 驗證 gpiozero 能不能用（LOBOROBOT2.py 需要）

執行
----
    python3 check_motor.py

正常輸出應該會看到類似：
    ✓ bus 13 有回應的裝置：0x40
    ✓ PCA9685 在 bus 13
    ✓ gpiozero 可用
    ✓ LOBOROBOT 硬體初始化成功
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from motor_control import (PCA9685_ADDRESS, MotorControl,  # noqa: E402
                           MODE_RPI, find_i2c_bus, _available_i2c_buses,
                           redirected_smbus)

OK = "\033[92m✓\033[0m"
NG = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"


def section(title: str):
    print("\n" + "=" * 58)
    print("  " + title)
    print("=" * 58)


def main() -> int:
    problems = 0

    # ── 1. I2C bus 清單 ──
    section("1. 系統上的 I2C bus")
    buses = _available_i2c_buses()
    if not buses:
        print(f"  {NG} 找不到任何 /dev/i2c-*")
        print()
        print("  請執行：sudo raspi-config")
        print("        Interfacing Options → I2C → Yes → Finish → reboot")
        return 1
    print(f"  {OK} 找到 {len(buses)} 條：{buses}")

    # ── 2. 掃描每條 bus ──
    section("2. 逐條掃描 I2C 裝置")
    try:
        import smbus
    except ImportError:
        print(f"  {NG} 缺少 smbus：sudo apt install python3-smbus i2c-tools")
        return 1

    found_any = False
    for n in buses:
        devices = []
        try:
            bus = smbus.SMBus(n)
        except Exception as e:
            print(f"  bus {n:<3d} 無法開啟（{type(e).__name__}）")
            continue

        for addr in range(0x03, 0x78):
            try:
                bus.read_byte(addr)
                devices.append(addr)
            except Exception:
                pass
        bus.close()

        mark = OK if PCA9685_ADDRESS in devices else WARN
        if PCA9685_ADDRESS in devices:
            found_any = True
        others = [f"0x{a:02X}" for a in devices
                  if a != PCA9685_ADDRESS][:6]
        print(f"  {mark} bus {n:<3d} 裝置："
              f"{[f'0x{a:02X}' for a in devices] or '（無）'}"
              + (f"   其他：{others}" if others else ""))

    # ── 3. 找 PCA9685 ──
    section("3. 尋找 PCA9685 (0x40)")
    bus_no, why = find_i2c_bus()
    print(f"  {'✓' if bus_no is not None else '✗'} {why}")

    if bus_no is None:
        problems += 1
        print()
        print("  常見原因：")
        print("   1. 馬達板沒通電（電池 / 電源線 / 板上的指示燈）")
        print("   2. SDA / SCL / GND 接線鬆脫")
        print("   3. PCA9685 板故障")
        print("   4. 馬達板跟樹莓派共用的電源被切斷（相機也一起消失就是這種）")

    # ── 4. gpiozero ──
    section("4. gpiozero（LOBOROBOT2.py 需要）")
    try:
        from gpiozero import LED
    except ImportError as e:
        print(f"  {NG} 缺少 gpiozero：sudo apt install python3-gpiozero（{e}）")
        problems += 1
    else:
        print(f"  {OK} gpiozero 可 import")
        # 只檢查後端有沒有裝，不要去動馬達用的 GPIO 24/25
        backends = []
        for name, mod in (("RPi.GPIO", "RPi"),
                          ("lgpio", "lgpio")):
            try:
                __import__(mod)
                backends.append(name)
            except ImportError:
                pass
        print(f"     GPIO 後端：{backends or '（沒裝任何一個）'}")

        if backends:
            # 用一顆無關的腳位做功能測試（避開馬達用的 24/25）
            try:
                probe = LED(17)
                probe.close()
                print(f"  {OK} GPIO 實際操作測試通過（GPIO 17）")
            except Exception as e:
                print(f"  {NG} GPIO 操作失敗：{type(e).__name__}: {e}")
                print("     若看到「無法取得 GPIO 權限」，把 pi 加入 gpio 群組：")
                print("       sudo usermod -aG gpio pi && 重新登入")
                problems += 1
        else:
            print(f"  {NG} 沒有可用的 GPIO 後端："
                  "sudo apt install python3-rpi.gpio")
            problems += 1

    # ── 5. 實際建構 LOBOROBOT ──
    section("5. LOBOROBOT 硬體初始化")
    if bus_no is None:
        print(f"  {WARN} 跳過（PCA9685 沒找到）")
    else:
        motor = MotorControl(mode=MODE_RPI)
        print(f"  {'✓' if motor.hardware_available else '✗'} "
              f"hardware_available = {motor.hardware_available}")
        print(f"    i2c_bus    = {motor.state.i2c_bus}")
        if motor.state.init_error:
            print(f"    init_error = {motor.state.init_error}")
            problems += 1

        if motor.hardware_available:
            print()
            print("  ⚠ 接下來會實際驅動馬達，請把車拿在手上！")
            print()
            try:
                motor.move("stop")
                time.sleep(0.3)
                print(f"  {OK} 停止指令執行成功")
            except Exception as e:
                print(f"  {NG} 停止指令失敗：{type(e).__name__}: {e}")
                problems += 1
            finally:
                motor.shutdown()

    # ── 總結 ──
    section("總結")
    if problems == 0:
        print("  {OK} 硬體檢查通過，可以進行馬達測試".replace("{OK}", OK))
        print()
        print("  記得：車要拿在手上，不要放地上。")
        return 0

    print(f"  {NG} 發現 {problems} 個問題，請先排除上面的問題。")
    return 1


if __name__ == "__main__":
    sys.exit(main())