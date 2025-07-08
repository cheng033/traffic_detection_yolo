# main.py（支援使用者點選影片檔案 + 畫 4 區計數）

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO
import numpy as np
import itertools
import sys
import torch
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# 確認設備是否支援 CUDA（GPU）
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ GPU 已啟用：", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("⚠️ 未偵測到 GPU，使用 CPU 模式")

# ---------- 選影片 ----------
Tk().withdraw()
VIDEO_PATH = askopenfilename(title="請選擇影片檔案", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
if not VIDEO_PATH:
    print("未選擇影片，程式結束")
    sys.exit()
cap = cv2.VideoCapture(VIDEO_PATH)

ret, first_frame = cap.read()
if not ret or first_frame is None:
    print("無法讀取影片影像，請確認檔案是否損毀")
    cap.release()
    sys.exit()

# ---------- 使用者畫 4 區域 ----------
NUM_AREAS = 4
drawing = False
areas = []
temp = []
area_palette = [(255, 0, 0), (0, 165, 255), (0, 255, 255), (128, 0, 128)]

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

    cv2.putText(disp, "Enter: 確認, C: 清除, ESC: 離開", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
print("✔️ 區域座標：", areas)

# ---------- 初始化模型 ----------
MODEL_PATH = "modelv1.pt"
model = YOLO(MODEL_PATH).to(device)
names = model.names
yolo_palette = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

# ---------- 初始化計數 ----------
tracks = {}
counted = [set() for _ in range(NUM_AREAS)]
cross = [0 for _ in range(NUM_AREAS)]

# ---------- 判斷重疊 ----------
def is_overlap(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)

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
            class_name = names[int(cls)]
            label = f"{class_name} {conf:.2f}"
            color = yolo_palette[int(cls) % len(yolo_palette)]
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
        cv2.putText(frame, f"Area {i+1}: {cross[i]}", (30, 40 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("YOLOv8 Crossing Counter", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord('r'):
        tracks.clear()
        for s in counted:
            s.clear()
        cross = [0 for _ in range(NUM_AREAS)]
        print("🔁 已重置計數")

cap.release()
cv2.destroyAllWindows()
