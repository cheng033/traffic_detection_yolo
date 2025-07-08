# main.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO
import numpy as np
import itertools
import sys
import argparse

import torch

# 確認設備是否支援 CUDA（GPU）
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU 已啟用：", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("未偵測到 GPU，使用 CPU 模式")


# ---------- 解析 GUI 傳入的參數 ----------
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, help='影片路徑')
parser.add_argument('--camera', action='store_true', help='使用攝影機')
args = parser.parse_args()

MODEL_PATH = "modelv1.pt"

if args.video:
    cap = cv2.VideoCapture(args.video)
elif args.camera:

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("攝影機開啟失敗，請確認裝置是否存在、權限允許，或是否被其他程式佔用")
        sys.exit()

else:
    print("請指定影片或攝影機作為來源")
    sys.exit()

ret, first_frame = cap.read()

if not ret or first_frame is None:
    print("無法讀取第一幀影像，可能是來源無畫面或裝置未準備好")
    cap.release()
    sys.exit()



# ---------- 讓使用者畫矩形區域 ----------
NUM_AREAS = 4
drawing = False
areas = []
temp = []
area_palette = [
    (255, 0, 0),    # 藍（上）
    (0, 165, 255),  # 橙（下）
    (0, 255, 255),  # 黃（左）
    (128, 0, 128),  # 紫（右）
]

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

cv2.namedWindow("Draw Areas")
cv2.setMouseCallback("Draw Areas", draw_rect)

while True:
    disp = first_frame.copy()
    for rect in areas:
        cv2.rectangle(disp, rect[0], rect[1], (0, 255, 0), 2)
    if len(temp) == 2:
        cv2.rectangle(disp, temp[0], temp[1], (0, 0, 255), 2)

    cv2.putText(disp, "Press Enter: confirm, C: clear", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Draw Areas", disp)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()
    if key == ord('c'):
        areas.clear()
        temp.clear()
        print("已清除畫面，請重新繪製區域")
    if key == 13 and len(areas) == NUM_AREAS:
        break

cv2.destroyWindow("Draw Areas")
print("區域座標：", areas)

# ---------- 初始化模型 ----------

model = YOLO(MODEL_PATH).to(device)

names = model.names
yolo_palette = [
    (0, 255, 0),      # 綠（car）
    (0, 0, 255),      # 紅（bus）
    (255, 255, 0),    # 青藍（motor）
    (255, 0, 255),    # 粉紅（truck）
]
# ---------- 計數資料 ----------
tracks = {}

counted = [set() for _ in range(NUM_AREAS)]
cross = [0 for _ in range(NUM_AREAS)]

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
            color = yolo_palette[int(cls) % len(yolo_palette)]

            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            det_box = (x1, y1, x2, y2)

            for i in range(NUM_AREAS):
                area_box = (
                    min(areas[i][0][0], areas[i][1][0]),
                    min(areas[i][0][1], areas[i][1][1]),
                    max(areas[i][0][0], areas[i][1][0]),
                    max(areas[i][0][1], areas[i][1][1])
                )

                if tid not in counted[i] and is_overlap(det_box, area_box):
                    cross[i] += 1
                    counted[i].add(tid)
    for i in range(NUM_AREAS):
        color = area_palette[i]
        cv2.rectangle(frame, areas[i][0], areas[i][1], color, 2)
        cv2.putText(frame, f"Area {i+1} crossing: {cross[i]}", (30, 40 + i * 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        

    cv2.imshow("YOLOv8 Crossing Counter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord('r'):
        tracks.clear()
        for s in counted:
            s.clear()
        for i in range(NUM_AREAS):
            cross[i] = 0
        print("已重置計數")

cap.release()
cv2.destroyAllWindows()

