from flask import Flask, Response, request
import cv2
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import numpy as np
import time
from selenium.webdriver.common.by import By

# ====== 設定 ======
TARGET_URL = "https://tw.live/cam/?id=NWT0052"
WINDOW_SIZE = "1400,900"
INTERVAL = 0.1  # 每幾秒抓一張圖

# ====== 初始化 Selenium ======
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument(f"--window-size={WINDOW_SIZE}")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
driver = webdriver.Chrome(options=chrome_options)
driver.get(TARGET_URL)
time.sleep(5)  # 等網頁完全載入

# ====== 初始化 Flask ======
current_url = TARGET_URL

app = Flask(__name__)

@app.route('/set_url', methods=['GET'])
def set_url():
    global current_url
    url = request.args.get('target', '').strip()
    if not url:
        return "No url!", 400
    try:
        driver.get(url)
        time.sleep(5)
        current_url = url
        print(f"已切換爬蟲來源到: {url}")
        return "ok"
    except Exception as e:
        print(f"切換失敗: {e}")
        return f"fail: {e}", 500

# ====== MJPEG Streaming ======


def gen_frames():
    last_refresh = time.time()
    REFRESH_INTERVAL = 30 # 單位：秒，這裡設你要的秒

    while True:
        try:
            # 檢查是否該刷新
            if time.time() - last_refresh > REFRESH_INTERVAL:
                print("[自動重新整理] Selenium ")
                driver.refresh()
                time.sleep(3)    # 等網頁重載完成
                last_refresh = time.time()

            # 抓畫面
            element = driver.find_element(By.CSS_SELECTOR, "div.container.google-anno-skip.my-3")
            png = element.screenshot_as_png
            nparr = np.frombuffer(png, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                img = img[0:540, :]  # 裁切下緣
                ret, buffer = cv2.imencode('.jpg', img)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(INTERVAL)
        except Exception as e:
            print("擷取影像失敗:", e)
            time.sleep(1)


@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)