# YOLOv8 光流方向追蹤 + 自動刷新 + 統計視覺化

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
from ultralytics import YOLO
from datetime import datetime
import itertools
import time
import threading
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 初始化 ----------
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1280,720")
driver = webdriver.Chrome(options=chrome_options)
driver.get("https://tw.live/cam/?id=NWT0052")
time.sleep(5)

page_title = driver.title.strip()
sanitized_title = re.sub(r'[\\/:*?"<>|\s]+', '_', page_title)
csv_filename = f"{sanitized_title}_flow.csv"

model = YOLO("modelv9.pt")
allowed_classes = [i for i, name in model.names.items() if 'car' in name.lower() or 'motor' in name.lower() or 'bike' in name.lower()]
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

active_flow = set()
flow_tracks = dict()
direction_dict = dict()
counters = {"left": 0, "right": 0, "straight": 0, "U-turn": 0}
records = []

# 擷取畫面並裁切中間區域
png = driver.get_screenshot_as_png()
image = Image.open(BytesIO(png)).convert('RGB')
frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
h, w = frame.shape[:2]
cx, cy = w // 2, h // 2
crop = frame[cy-170:cy+250, cx-350:cx+350]

# ROI 區域設定
NUM_AREAS = 1
areas, temp = [], []
drawing = False

def draw_rect(event, x, y, flags, param):
    global drawing, temp, areas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        temp = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        temp.append((x, y))
        if len(temp) == 2:
            areas.append(tuple(temp))
            temp = []

cv2.namedWindow("Draw Area")
cv2.setMouseCallback("Draw Area", draw_rect)
while True:
    disp = crop.copy()
    for rect in areas:
        cv2.rectangle(disp, rect[0], rect[1], (0, 0, 255), 2)
    if len(temp) == 2:
        cv2.rectangle(disp, temp[0], temp[1], (0, 255, 255), 2)
    cv2.putText(disp, "Enter: confirm, C: clear", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow("Draw Area", disp)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        driver.quit()
        cv2.destroyAllWindows()
        exit()
    if key == ord('c'):
        areas.clear()
        temp.clear()
    if key == 13 and len(areas) == NUM_AREAS:
        break
cv2.destroyWindow("Draw Area")

prev_crop = None
last_refresh = time.time()

while True:
    # 自動 F5 防斷流
    if time.time() - last_refresh > 10:
        driver.refresh()
        time.sleep(3)
        last_refresh = time.time()

    png = driver.get_screenshot_as_png()
    image = Image.open(BytesIO(png)).convert('RGB')
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    crop = frame[cy-170:cy+250, cx-350:cx+350]

    results = model.track(crop, persist=True, tracker="bytetrack.yaml")
    boxes = results[0].boxes
    if boxes is not None:
        ids = boxes.id.cpu().numpy() if boxes.id is not None else itertools.count()
        classes = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for box, tid, cls, conf in zip(boxes.xyxy.cpu().numpy(), ids, classes, confs):
            if conf < 0.4 or int(cls) not in allowed_classes:
                continue
            tid = int(tid)
            x1, y1, x2, y2 = map(int, box)
            cx_obj, cy_obj = int((x1 + x2)/2), int((y1 + y2)/2)

            area_box = (
                min(areas[0][0][0], areas[0][1][0]),
                min(areas[0][0][1], areas[0][1][1]),
                max(areas[0][0][0], areas[0][1][0]),
                max(areas[0][0][1], areas[0][1][1])
            )
            if tid not in active_flow and area_box[0] < cx_obj < area_box[2] and area_box[1] < cy_obj < area_box[3]:
                active_flow.add(tid)
                flow_tracks[tid] = [(cx_obj, cy_obj)]

            if tid in active_flow and tid in flow_tracks and prev_crop is not None:
                prev_pt = np.float32([flow_tracks[tid][-1]]).reshape(-1, 1, 2)
                new_pt, status, err = cv2.calcOpticalFlowPyrLK(prev_crop, crop, prev_pt, None, **lk_params)
                if status[0][0] == 1:
                    vx = new_pt[0][0][0] - prev_pt[0][0][0]
                    vy = new_pt[0][0][1] - prev_pt[0][0][1]
                    flow_tracks[tid].append((cx_obj, cy_obj))
                    flow_tracks[tid] = flow_tracks[tid][-5:]
                    for i in range(1, len(flow_tracks[tid])):
                        cv2.arrowedLine(crop, flow_tracks[tid][i-1], flow_tracks[tid][i], (255,255,255), 2)

                    if tid not in direction_dict:
                        if abs(vx) > abs(vy):
                            direction = "right" if vx > 2 else "left" if vx < -2 else "straight"
                        elif abs(vy) > 2:
                            direction = "straight"
                        else:
                            direction = "U-turn"
                        direction_dict[tid] = direction
                        counters[direction] += 1
                        records.append([tid, model.names[int(cls)], direction, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

            label = f"{model.names[int(cls)]}"
            cv2.rectangle(crop, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(crop, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    stat_text = f"left: {counters['left']} right: {counters['right']} straight: {counters['straight']} U-turn: {counters['U-turn']}"
    cv2.putText(crop, stat_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Detection + Stats", crop)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    prev_crop = crop.copy()

driver.quit()
cv2.destroyAllWindows()

# 儲存 CSV
with open(csv_filename, "w", newline="", encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["TrackID", "Class", "Direction", "Timestamp"])
    writer.writerows(records)

# 畫出 donut 圖
df = pd.DataFrame(records, columns=["TrackID", "Class", "Direction", "Timestamp"])
count_by_dir = df["Direction"].value_counts()

fig, ax = plt.subplots(figsize=(5, 5))
wedges, texts, autotexts = ax.pie(
    count_by_dir.values,
    labels=count_by_dir.index,
    autopct='%1.0f%%',
    startangle=90,
    wedgeprops=dict(width=0.4)
)
ax.set_title("車輛方向分布")
plt.tight_layout()
plt.savefig(f"{sanitized_title}_direction_donut.png", dpi=200)
plt.close()

print("✅ 統計完成，CSV 與視覺化圖表已儲存！")
