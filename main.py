import cv2
import numpy as np
import pyautogui
import os
import threading
import speech_recognition as sr
from detect import process_image

screen_width, screen_height = pyautogui.size()
screenshot_path = os.path.join("./output", "screenshot.jpg")

cv2.namedWindow("Halal Scroller", cv2.WINDOW_NORMAL)


def voice_command_listener():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("[INFO] Voice recognition started...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        while True:
            try:
                print("[INFO] Listening for 'next' command...")
                audio = recognizer.listen(source, timeout=None)
                command = recognizer.recognize_google(audio).lower()
                print(f"[INFO] Detected voice: {command}")
                if "next" in command:
                    print("[INFO] 'Next' detected! Pressing Down.")
                    pyautogui.press("down")
                if "up" in command:
                    print("[INFO] 'Next' detected! Pressing Down.")
                    pyautogui.press("up")
            except sr.UnknownValueError:
                print("[INFO] No recognizable voice detected.")
            except sr.RequestError:
                print("[ERROR] Could not connect to voice recognition service.")
            except sr.WaitTimeoutError:
                print("[INFO] Listening timeout, retrying...")


voice_thread = threading.Thread(target=voice_command_listener, daemon=True)
voice_thread.start()

while True:
    screen = pyautogui.screenshot(region=(0, 0, screen_width, screen_height))
    screen.save(screenshot_path)

    detected_data, output_image_path = process_image(
        model_path="./model/model.tflite",
        image_path=screenshot_path,
        xlsx_path="./data-nutri.xlsx",
        min_conf=0.3,
    )
    print(detected_data)

    if detected_data:
        print("[INFO] Detected food! Pressing Down.")
        pyautogui.press("down")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("[INFO] Exiting program...")
        break

cv2.destroyAllWindows()
