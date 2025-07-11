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

model = YOLO("modelv1.pt")
log_records = []

# 畫框
NUM_AREAS = 4
areas = []
temp = []
drawing = False
colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0)]

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

# 擷取畫面並裁切區域
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
    for idx, rect in enumerate(areas):
        cv2.rectangle(disp, rect[0], rect[1], colors[idx], 2)
        cv2.putText(disp, f"Area {idx+1}", rect[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], 2)
    if len(temp) == 2:
        cv2.rectangle(disp, temp[0], temp[1], (255, 255, 255), 2)
    cv2.putText(disp, "Enter: confirm, C: clear", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
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

entry_box = areas[0]
left_box, straight_box, right_box = areas[1:]
entered_set = set()
direction_decided = dict()
direction_counts = {"left": 0, "straight": 0, "right": 0}

def is_overlap(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return inter_x1 < inter_x2 and inter_y1 < inter_y2

def box_from_area(area):
    return (
        min(area[0][0], area[1][0]),
        min(area[0][1], area[1][1]),
        max(area[0][0], area[1][0]),
        max(area[0][1], area[1][1])
    )

entry_box = box_from_area(entry_box)
left_box = box_from_area(left_box)
straight_box = box_from_area(straight_box)
right_box = box_from_area(right_box)

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
            det_box = (x1, y1, x2, y2)

            if tid not in entered_set and is_overlap(det_box, entry_box):
                entered_set.add(tid)

            if tid in entered_set and tid not in direction_decided:
                if is_overlap(det_box, left_box):
                    direction = "left"
                elif is_overlap(det_box, straight_box):
                    direction = "straight"
                elif is_overlap(det_box, right_box):
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
