from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
from ultralytics import YOLO
from datetime import datetime
import csv
import itertools
import time
import threading
import re
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

# 抓取 <title> 並轉成合法檔名
page_title = driver.title.strip()
sanitized_title = re.sub(r'[\\/:*?"<>|\s]+', '_', page_title)
csv_filename = f"{sanitized_title}.csv"

# 每10秒刷新防止停播
def auto_refresh():
    while True:
        time.sleep(10)
        print("[自動重新整理] F5 中...")
        driver.refresh()
        time.sleep(5)

threading.Thread(target=auto_refresh, daemon=True).start()

model = YOLO("modelv3.pt")
recording = True
log_records = []
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_crop_manual_roi_refresh.mp4", fourcc, 10, (700, 500))

NUM_AREAS = 4
drawing = False
areas = []
temp = []
area_palette = [(255, 0, 0), (0, 165, 255), (0, 255, 255), (128, 0, 128)]
yolo_palette = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
counted = [set() for _ in range(NUM_AREAS)]
cross = [0 for _ in range(NUM_AREAS)]
vehicle_counts = [dict() for _ in range(NUM_AREAS)]
tracks = {}

def is_overlap(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return inter_x1 < inter_x2 and inter_y1 < inter_y2

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

png = driver.get_screenshot_as_png()
image = Image.open(BytesIO(png)).convert('RGB')
frame = np.array(image)
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
h, w = frame.shape[:2]
cx, cy = w // 2, h // 2
crop = frame[cy-170:cy+250, cx-350:cx+350]

cv2.namedWindow("Draw Areas")
cv2.setMouseCallback("Draw Areas", draw_rect)

while True:
    disp = crop.copy()
    for rect in areas:
        cv2.rectangle(disp, rect[0], rect[1], (0, 255, 0), 2)
    if len(temp) == 2:
        cv2.rectangle(disp, temp[0], temp[1], (0, 0, 255), 2)
    cv2.putText(disp, "Enter: confirm, C: clear", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Draw Areas", disp)
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

cv2.destroyWindow("Draw Areas")

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
            label = f"{model.names[int(cls)]} {conf:.2f}"
            color = yolo_palette[int(cls) % len(yolo_palette)]
            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.rectangle(crop, (x1, y1), (x2, y2), color, 2)
            cv2.putText(crop, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if tid not in tracks:
                tracks[tid] = []
            tracks[tid].append((cx, cy))
            if len(tracks[tid]) > 10:
                tracks[tid] = tracks[tid][-10:]

            det_box = (x1, y1, x2, y2)
            for i in range(NUM_AREAS):
                area_box = (
                    min(areas[i][0][0], areas[i][1][0]),
                    min(areas[i][0][1], areas[i][1][1]),
                    max(areas[i][0][0], areas[i][1][0]),
                    max(areas[i][0][1], areas[i][1][1])
                )
                if tid not in counted[i] and is_overlap(det_box, area_box):
                    direction = "unknown"
                    if len(tracks[tid]) >= 2:
                        dy = tracks[tid][-1][1] - tracks[tid][0][1]
                        if dy > 20:
                            direction = "進入"
                        elif dy < -20:
                            direction = "離開"
                        else:
                            direction = "靜止"

                    cross[i] += 1
                    counted[i].add(tid)
                    cname = model.names[int(cls)]
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_records.append([i + 1, tid, cname, direction, timestamp])
                    if cname in vehicle_counts[i]:
                        vehicle_counts[i][cname] += 1
                    else:
                        vehicle_counts[i][cname] = 1
                    print(f"[{timestamp}] {direction} Area {i+1} -> ID:{tid} 類別:{cname}")

    for i in range(NUM_AREAS):
        color = area_palette[i]
        cv2.rectangle(crop, areas[i][0], areas[i][1], color, 2)
        cv2.putText(crop, f"A{i+1}:{cross[i]}", (areas[i][0][0], areas[i][0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(crop, now_str, (crop.shape[1]-250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Detection + Stats", crop)
    if recording:
        out.write(crop)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord('r'):
        for s in counted: s.clear()
        for i in range(NUM_AREAS): cross[i] = 0
        log_records.clear()
        for dic in vehicle_counts: dic.clear()
        print("已重置計數")

driver.quit()
out.release()
cv2.destroyAllWindows()

# 儲存主記錄（含方向）
with open(csv_filename, "w", newline="", encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["Area", "TrackID", "Class", "Direction", "Timestamp"])
    writer.writerows(log_records)

# 儲存統計表
stats_filename = csv_filename.replace(".csv", "_summary.csv")
with open(stats_filename, "w", newline="", encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["Area", "Class", "Count"])
    for i, area_dict in enumerate(vehicle_counts):
        for cname, count in area_dict.items():
            writer.writerow([f"Area {i+1}", cname, count])

# 畫出 donut 圖
df = pd.read_csv(stats_filename)
for area in df['Area'].unique():
    subdf = df[df['Area'] == area]
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        subdf['Count'],
        labels=subdf['Class'],
        autopct='%1.0f%%',
        startangle=90,
        wedgeprops=dict(width=0.4)
    )
    ax.set_title(f"{area} 車種分佈")
    plt.tight_layout()
    plt.savefig(f"{area}_donut_chart.png", dpi=200)
    plt.close()

print("\U0001f389 所有記錄與統計圖表皆已完成！")
