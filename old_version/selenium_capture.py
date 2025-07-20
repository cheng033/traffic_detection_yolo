# selenium_capture.py
import threading
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

class SeleniumCapture:
    def __init__(self, url):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1280,720")
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.get(url)
        time.sleep(5)
        self.stopped = False
        # 啟動自動refresh thread
        self.thread = threading.Thread(target=self.auto_refresh, daemon=True)
        self.thread.start()
    
    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return 5  # 可以根據實際網頁刷新速度調整
        return 0
    
    def auto_refresh(self):
        while not self.stopped:
            time.sleep(10)
            print("[自動重新整理] F5 中...")
            self.driver.refresh()
            time.sleep(5)

    def read(self):
        # 這裡模擬成 cv2.VideoCapture.read() 介面，回傳 (ret, frame)
        try:
            png = self.driver.get_screenshot_as_png()
            image = Image.open(BytesIO(png)).convert('RGB')
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 這邊可以視需要裁切
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            frame = frame[cy-170:cy+250, cx-350:cx+350]
            return True, frame
        except Exception as e:
            print("[selenium read error]", e)
            return False, None

    def release(self):
        self.stopped = True
        self.driver.quit()
