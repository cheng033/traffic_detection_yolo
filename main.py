# main.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO
import numpy as np
import itertools
import sys
import argparse

# ---------- 解析 GUI 傳入的參數 ----------
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, help='影片路徑')
parser.add_argument('--camera', action='store_true', help='使用攝影機')
args = parser.parse_args()

MODEL_PATH = "modelv1.pt"

if args.video:
    cap = cv2.VideoCapture(args.video)
elif args.camera:
    cap = cv2.VideoCapture(0)
else:
    print("請指定影片或攝影機作為來源")
    sys.exit()

ret, first_frame = cap.read()
if not ret:
    print("無法讀取影像來源")
    sys.exit()

# ---------- 讓使用者畫兩個矩形區域 ----------
drawing = False
areas = []
temp = []

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

cv2.namedWindow("Draw 2 Areas")
cv2.setMouseCallback("Draw 2 Areas", draw_rect)

while True:
    disp = first_frame.copy()
    for rect in areas:
        cv2.rectangle(disp, rect[0], rect[1], (0, 255, 0), 2)
    if len(temp) == 2:
        cv2.rectangle(disp, temp[0], temp[1], (0, 0, 255), 2)

    cv2.putText(disp, "Press Enter: confirm, C: clear", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Draw 2 Areas", disp)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()
    if key == ord('c'):
        areas.clear()
        temp.clear()
        print("已清除畫面，請重新繪製兩個區域")
    if key == 13 and len(areas) == 2:
        break

cv2.destroyWindow("Draw 2 Areas")
print("兩個區域座標：", areas)

# ---------- 初始化模型 ----------
model = YOLO(MODEL_PATH)
names = model.names
palette = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
           (0,255,255),(128,128,0),(128,0,128),(0,128,128),(255,255,255)]

# ---------- 計數資料 ----------
tracks = {}
counted_left = set()
counted_right = set()
cross_left = 0
cross_right = 0

def is_overlap(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return inter_x1 < inter_x2 and inter_y1 < inter_y2

# ---------- 主迴圈 ----------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes = results[0].boxes

    if boxes is not None:
        ids = boxes.id.cpu().numpy() if boxes.id is not None else itertools.count()
        classes = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for box, tid, cls, conf in zip(boxes.xyxy.cpu().numpy(), ids, classes, confs):
            if conf < 0.4:
                continue

            tid = int(tid)
            label = f"{names[int(cls)]} {conf:.2f}"
            color = palette[int(cls) % len(palette)]

            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            det_box = (x1, y1, x2, y2)
            area0_box = (
                min(areas[0][0][0], areas[0][1][0]),
                min(areas[0][0][1], areas[0][1][1]),
                max(areas[0][0][0], areas[0][1][0]),
                max(areas[0][0][1], areas[0][1][1])
            )
            area1_box = (
                min(areas[1][0][0], areas[1][1][0]),
                min(areas[1][0][1], areas[1][1][1]),
                max(areas[1][0][0], areas[1][1][0]),
                max(areas[1][0][1], areas[1][1][1])
            )

            if tid not in counted_left and is_overlap(det_box, area0_box):
                cross_left += 1
                counted_left.add(tid)
            if tid not in counted_right and is_overlap(det_box, area1_box):
                cross_right += 1
                counted_right.add(tid)

    cv2.rectangle(frame, areas[0][0], areas[0][1], (0,255,0), 2)
    cv2.rectangle(frame, areas[1][0], areas[1][1], (255,0,0), 2)

    cv2.putText(frame, f"Left crossing:  {cross_left}",  (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    cv2.putText(frame, f"Right crossing: {cross_right}", (30,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)

    cv2.imshow("YOLOv8 Crossing Counter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord('r'):
        tracks.clear()
        counted_left.clear()
        counted_right.clear()
        cross_left = cross_right = 0
        print("已重置計數")

cap.release()
cv2.destroyAllWindows()

