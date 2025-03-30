import pyautogui
import os
from detect import process_image_and_extract_nutrients

# Tentukan path penyimpanan sementara
screenshot_path = os.path.join("./output", "screenshot.jpg")

# Ambil screenshot dan simpan sebagai file
screenshot = pyautogui.screenshot()
screenshot.save(screenshot_path)

# Panggil fungsi dengan screenshot sebagai input
detected_data, output_image_path = process_image_and_extract_nutrients(
    model_path="./model/model.tflite",
    image_path=screenshot_path,
    xlsx_path="./data-nutri.xlsx",
    min_conf=0.1,
)

# Cetak hasil
print(detected_data)
print(f"Output image saved at: {output_image_path}")
