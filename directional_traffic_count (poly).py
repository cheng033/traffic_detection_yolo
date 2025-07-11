import cv2
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
import numpy as np
from io import BytesIO
from ultralytics import YOLO
from datetime import datetime
import csv
import itertools
import time
import threading
import re

# ---------- 使用者輸入網址 ----------
url = input("請輸入即時影像網址（例如：https://tw.live/cam/?id=NWT0052）：\n")

# ---------- 初始化 ----------
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1280,720")
driver = webdriver.Chrome(options=chrome_options)
driver.get(url)
time.sleep(5)

page_title = driver.title.strip()
sanitized_title = re.sub(r'[\\/:*?"<>|\s]+', '_', page_title)
csv_filename = f"{sanitized_title}_direction.csv"

# 每10秒刷新防止停播
def auto_refresh():
    while True:
        time.sleep(10)
        print("[自動重新整理] F5 中...")
        driver.refresh()
        time.sleep(5)

threading.Thread(target=auto_refresh, daemon=True).start()

model = YOLO("modelv3.pt")
log_records = []

# 畫多邊形區域
NUM_AREAS = 4
areas = []
drawing = False
current_poly = []
colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0)]

def draw_polygon(event, x, y, flags, param):
    global drawing, current_poly, areas
    if event == cv2.EVENT_LBUTTONDOWN:
        current_poly.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_poly) >= 3:
            areas.append(np.array(current_poly, np.int32))
            current_poly = []

# 擷取畫面並裁切區域
png = driver.get_screenshot_as_png()
image = Image.open(BytesIO(png)).convert('RGB')
frame = np.array(image)
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
h, w = frame.shape[:2]
cx, cy = w // 2, h // 2
crop = frame[cy-170:cy+250, cx-350:cx+350]

cv2.namedWindow("Draw Areas")
cv2.setMouseCallback("Draw Areas", draw_polygon)

while True:
    disp = crop.copy()
    for idx, poly in enumerate(areas):
        cv2.polylines(disp, [poly], isClosed=True, color=colors[idx], thickness=2)
        cv2.putText(disp, f"Area {idx+1}", tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], 2)
    for point in current_poly:
        cv2.circle(disp, point, 4, (255, 255, 255), -1)
    cv2.putText(disp, "左鍵點多邊形，右鍵完成一區 | Enter: confirm, C: clear", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imshow("Draw Areas", disp)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        driver.quit()
        cv2.destroyAllWindows()
        exit()
    if key == ord('c'):
        areas.clear()
        current_poly.clear()
    if key == 13 and len(areas) == NUM_AREAS:
        break

cv2.destroyWindow("Draw Areas")

entry_poly = areas[0]
left_poly, straight_poly, right_poly = areas[1:]
entered_set = set()
direction_decided = dict()
direction_counts = {"left": 0, "straight": 0, "right": 0}

def point_in_poly(point, poly):
    return cv2.pointPolygonTest(poly, point, False) >= 0

# ---------- 主迴圈 ----------
while True:
    png = driver.get_screenshot_as_png()
    image = Image.open(BytesIO(png)).convert('RGB')
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    crop = frame[cy-170:cy+250, cx-350:cx+350]

    results = model.track(crop, persist=True, tracker="bytetrack.yaml")
    boxes = results[0].boxes

    if boxes is not None:
        ids = boxes.id.cpu().numpy() if boxes.id is not None else itertools.count()
        classes = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for box, tid, cls, conf in zip(boxes.xyxy.cpu().numpy(), ids, classes, confs):
            if conf < 0.4:
                continue
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
                    log_records.append([tid, direction, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

            color = (0, 0, 255) if tid in entered_set else (128, 128, 128)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(crop, (x1, y1), (x2, y2), color, 2)
            cv2.putText(crop, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(crop, f"LEFT: {direction_counts['left']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(crop, f"STRAIGHT: {direction_counts['straight']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(crop, f"RIGHT: {direction_counts['right']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Detection + Direction", crop)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# 儲存 CSV
with open(csv_filename, "w", newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["TrackID", "Direction", "Timestamp"])
    writer.writerows(log_records)

driver.quit()
cv2.destroyAllWindows()
print("🎯 方向記錄完成並儲存為 CSV！")
