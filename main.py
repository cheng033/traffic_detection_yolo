# main.py
import cv2
from ultralytics import YOLO
import numpy as np
import itertools


MODEL_PATH  = "best2.pt"          # YOLOv8 權重
VIDEO_PATH  = "test_video.mp4"    # 要分析的影片


# ---------- 1. 先讓使用者畫兩個矩形區域 ----------
drawing = False
areas   = []       # 每個元素 [(x1,y1),(x2,y2)]
temp    = []

def draw_rect(event, x, y, flags, param):
    global drawing, temp, areas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        temp = [(x, y)]          # 記錄第一點
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        temp.append((x, y))      # 第二點
        if len(temp) == 2:
            areas.append(tuple(temp))
            temp = []

cap = cv2.VideoCapture(VIDEO_PATH)
ret, first_frame = cap.read()
if not ret:
    print("無法讀取影片，請確認 VIDEO_PATH")
    exit()

cv2.namedWindow("Draw 2 Areas (Click-LT & RB twice)")
cv2.setMouseCallback("Draw 2 Areas (Click-LT & RB twice)", draw_rect)

while True:
    disp = first_frame.copy()
    for rect in areas:
        cv2.rectangle(disp, rect[0], rect[1], (0,255,0), 2)
    if len(temp) == 2:
        cv2.rectangle(disp, temp[0], temp[1], (0,0,255), 2)
    cv2.imshow("Draw 2 Areas (Click-LT & RB twice)", disp)
    if len(areas) >= 2:
        break
    if cv2.waitKey(1) & 0xFF == 27:   # ESC 可退出
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Draw 2 Areas (Click-LT & RB twice)")
print("兩個區域座標：", areas)

# ---------- 2. 初始化模型 ----------
model = YOLO(MODEL_PATH)
names = model.names           # 類別對照表
# 為每個類別自動產生顏色（循環使用 10 種顏色）
palette = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
           (0,255,255),(128,128,0),(128,0,128),(0,128,128),(255,255,255)]

# ---------- 3. 統計與資料結構 ----------
tracks          = {}      # id -> [(cx,cy)...]
counted_left    = set()
counted_right   = set()
cross_left  = 0
cross_right = 0

def in_area(pt, rect):
    (x1,y1),(x2,y2)=rect
    x_min,x_max = sorted([x1,x2])
    y_min,y_max = sorted([y1,y2])
    return x_min<=pt[0]<=x_max and y_min<=pt[1]<=y_max

# ---------- 4. 影片迴圈 ----------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes   = results[0].boxes
    if boxes is not None:
        ids     = boxes.id.cpu().numpy() if boxes.id is not None else itertools.count()
        classes = boxes.cls.cpu().numpy()
        confs   = boxes.conf.cpu().numpy()

        for box, tid, cls, conf in zip(boxes.xyxy.cpu().numpy(), ids, classes, confs):
            tid = int(tid)
            label = f"{names[int(cls)]} {conf:.2f}"
            color = palette[int(cls)%len(palette)]

            x1,y1,x2,y2 = map(int, box)
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ----- crossing 計算 -----
            base_pt = (cx, y2)        
            if tid not in counted_left and in_area(base_pt, areas[0]):   # ❷
                cross_left += 1
                counted_left.add(tid)                                    # ❸
            if tid not in counted_right and in_area(base_pt, areas[1]):  # ❹
                cross_right += 1
                counted_right.add(tid)

    # 畫區域框
    cv2.rectangle(frame, areas[0][0], areas[0][1], (0,255,0), 2)
    cv2.rectangle(frame, areas[1][0], areas[1][1], (255,0,0), 2)

    # 顯示計數
    cv2.putText(frame, f"Left crossing:  {cross_left}",  (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    cv2.putText(frame, f"Right crossing: {cross_right}", (30,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)

    cv2.imshow("YOLOv8 Crossing Counter", frame)
    key=cv2.waitKey(1)&0xFF
    if key==27: break           # ESC 離開
    if key==ord('r'):           # R 重置
        tracks.clear(); counted_left.clear(); counted_right.clear()
        cross_left=cross_right=0
        print("已重置計數")

cap.release()
cv2.destroyAllWindows()
