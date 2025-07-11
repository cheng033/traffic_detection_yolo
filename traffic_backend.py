import asyncio
import threading
import websockets
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# 允許前端跨域存取
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 接收的網址請求
class StreamRequest(BaseModel):
    url: str

# 存放方向統計資料
direction_counts = {"left": 0, "straight": 0, "right": 0}
clients = set()

@app.post("/start")
async def start_stream(req: StreamRequest):
    print("[後端啟動分析] 來自網址:", req.url)
    threading.Thread(target=run_tracking_model, args=(req.url,), daemon=True).start()
    return {"status": "started"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await websocket.send_text(json.dumps(direction_counts))
            await asyncio.sleep(1)
    except:
        clients.remove(websocket)

# 真實分析模型整合（使用 YOLOv8 + OpenCV）
def run_tracking_model(url):
    import cv2
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from PIL import Image
    import numpy as np
    from io import BytesIO
    from ultralytics import YOLO
    import time
    import re

    # 初始化 webdriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1280,720")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    time.sleep(5)

    model = YOLO("modelv3.pt")

    # 初始化 ROI
    entry_poly = np.array([[20, 400], [200, 400], [200, 500], [20, 500]], np.int32)
    left_poly = np.array([[0, 0], [150, 0], [150, 150], [0, 150]], np.int32)
    straight_poly = np.array([[160, 0], [320, 0], [320, 150], [160, 150]], np.int32)
    right_poly = np.array([[330, 0], [480, 0], [480, 150], [330, 150]], np.int32)

    entered_set = set()
    direction_decided = dict()

    def point_in_poly(point, poly):
        return cv2.pointPolygonTest(poly, point, False) >= 0

    while True:
        try:
            png = driver.get_screenshot_as_png()
            image = Image.open(BytesIO(png)).convert('RGB')
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            crop = frame[cy-170:cy+250, cx-350:cx+350]

            results = model.track(crop, persist=True, tracker="bytetrack.yaml")
            boxes = results[0].boxes

            if boxes is not None:
                ids = boxes.id.cpu().numpy() if boxes.id is not None else itertools.count()
                for box, tid in zip(boxes.xyxy.cpu().numpy(), ids):
                    tid = int(tid)
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    center = (cx, cy)

                    if tid not in entered_set and point_in_poly(center, entry_poly):
                        entered_set.add(tid)

                    if tid in entered_set and tid not in direction_decided:
                        if point_in_poly(center, left_poly):
                            direction = "left"
                        elif point_in_poly(center, straight_poly):
                            direction = "straight"
                        elif point_in_poly(center, right_poly):
                            direction = "right"
                        else:
                            direction = None

                        if direction:
                            direction_counts[direction] += 1
                            direction_decided[tid] = direction

            time.sleep(0.3)
        except Exception as e:
            print("[分析錯誤]", e)
            time.sleep(1)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
