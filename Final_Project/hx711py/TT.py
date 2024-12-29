import time
import sys
import RPi.GPIO as GPIO
from hx711 import HX711

# 清理 GPIO 並退出程式
def clean_and_exit():
    print("Cleaning...")
    GPIO.cleanup()
    print("Bye!")
    sys.exit()


# 初始化 HX711，並指定通訊腳位
hx = HX711(5, 6)

# 設置讀取格式：可以調整 "MSB", "LSB" 的組合以測試穩定性
hx.set_reading_format("MSB", "MSB")

# 設置參考單位（校正參數）
reference_unit = 114  # 根據測試調整此參數
hx.set_reference_unit(reference_unit)

# 重置並去皮
hx.reset()
hx.tare()
print("Tare done! Add weight now...")


# 平滑處理：計算多次測量的平均值
def get_stable_weight(readings=10):
    weights = []
    for _ in range(readings):
        weight = hx.get_weight(5)
        weights.append(weight)
        time.sleep(0.1)  # 每次測量間隔 0.1 秒
    return sum(weights) / len(weights)


# 主程式循環
try:
    while True:
        # 調試模式：檢查位元組順序和格式
        np_arr8_string = hx.get_np_arr8_string()
        binary_string = hx.get_binary_string()
        print(f"Binary: {binary_string} | NP Array: {np_arr8_string}")

        # 獲取穩定重量
        weight = get_stable_weight(10)  # 10 次取樣平均值
        print(f"Weight: {weight:.2f} g")

        # 節能模式：關閉電源並重新上電
        hx.power_down()
        hx.power_up()
        time.sleep(0.5)  # 主循環間隔 0.5 秒

except (KeyboardInterrupt, SystemExit):
    clean_and_exit()
