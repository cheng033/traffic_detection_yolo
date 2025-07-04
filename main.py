# main.py

# 避免 OpenMP 錯誤（libiomp5md.dll 載入重複），設定環境變數
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 匯入必要套件
import cv2                      # OpenCV 做影像處理
from ultralytics import YOLO   # YOLOv8 模型
import numpy as np
import itertools
from tkinter import Tk
from tkinter.filedialog import askopenfilename  # 檔案選擇器
import sys

# ---------- 使用者選擇影片 ----------
MODEL_PATH = "best2.pt"  # 模型路徑寫死，若需改為選擇模型，也可用 askopenfilename()

Tk().withdraw()  # 關閉 tkinter 主視窗，只使用檔案選取對話框
VIDEO_PATH = askopenfilename(title="請選擇影片檔案", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])

# 如果沒選檔案，結束程式
if not VIDEO_PATH:
    print("未選擇影片，程式結束。")
    sys.exit()


# ---------- 1. 讓使用者畫兩個矩形區域 ----------

drawing = False
areas = []     # 儲存已畫完的區域，每個元素是 [(x1,y1), (x2,y2)]
temp = []      # 暫存目前正在畫的矩形

# 滑鼠回呼函式，讓使用者框出兩個區域
def draw_rect(event, x, y, flags, param):
    global drawing, temp, areas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        temp = [(x, y)]  # 紀錄起點
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        temp.append((x, y))  # 紀錄終點
        if len(temp) == 2:
            areas.append(tuple(temp))  # 儲存區域
            temp = []  # 清空暫存

# 開啟影片
cap = cv2.VideoCapture(VIDEO_PATH)
ret, first_frame = cap.read()  # 讀取第一幀，供使用者框選

# 若無法讀取影片，跳出錯誤
if not ret:
    print("無法讀取影片，請確認 VIDEO_PATH")
    sys.exit()

# 顯示畫面，讓使用者畫出兩個區域
cv2.namedWindow("Draw 2 Areas (Click-LT & RB twice)")
cv2.setMouseCallback("Draw 2 Areas (Click-LT & RB twice)", draw_rect)

while True:
    disp = first_frame.copy()

    # 畫出已完成的區域
    for rect in areas:
        cv2.rectangle(disp, rect[0], rect[1], (0, 255, 0), 2)
    
    # 畫暫時區域（正在畫）
    if len(temp) == 2:
        cv2.rectangle(disp, temp[0], temp[1], (0, 0, 255), 2)

    cv2.imshow("Draw 2 Areas (Click-LT & RB twice)", disp)

    key = cv2.waitKey(1) & 0xFF  # 取得按鍵

    if key == 27:  # ESC 鍵離開
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()

    if key == ord('c'):
        if temp:  # 如果正在畫但還沒放開滑鼠
            temp.clear()
            print("已取消當前正在繪製的矩形")
        elif areas:  # 如果已完成的框中還有
            removed = areas.pop()  # 移除最後一個完成的區域
            print(f"已移除最後一個區域：{removed}")
        else:
            print("目前沒有可以取消的區域")



    if key == 13 and len(areas) >= 2:  # Enter 鍵（keycode 13）＋兩個區域畫完
        break


cv2.destroyWindow("Draw 2 Areas (Click-LT & RB twice)")
print("兩個區域座標：", areas)


# ---------- 2. 初始化 YOLO 模型 ----------
model = YOLO(MODEL_PATH)
names = model.names  # 類別名稱對照表

# 為不同類別指定不同顏色（最多支援 10 類）
palette = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
           (0,255,255),(128,128,0),(128,0,128),(0,128,128),(255,255,255)]


# ---------- 3. 資料結構：計數與追蹤 ----------

tracks = {}           # 儲存各 ID 的軌跡（未使用，可拓展）
counted_left = set()  # 已計數過的 ID（左側區域）
counted_right = set() # 已計數過的 ID（右側區域）
cross_left = 0        # 左區域進入計數器
cross_right = 0       # 右區域進入計數器

# 判斷某點是否在指定矩形區域中
def is_overlap(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    return inter_x1 < inter_x2 and inter_y1 < inter_y2


# ---------- 4. 主迴圈：逐幀偵測 + 統計 ----------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 播放結束

    # YOLOv8 進行追蹤推論，搭配 ByteTrack
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes = results[0].boxes  # 取得偵測結果

    if boxes is not None:
        # 若 boxes.id 為 None 則建立自動計數器
        ids = boxes.id.cpu().numpy() if boxes.id is not None else itertools.count()
        classes = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        # 處理每個偵測框
        for box, tid, cls, conf in zip(boxes.xyxy.cpu().numpy(), ids, classes, confs):
            tid = int(tid)
            label = f"{names[int(cls)]} {conf:.2f}"  # 類別 + 信心值
            color = palette[int(cls) % len(palette)]  # 對應顏色

            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)

            # 繪製偵測框與標籤
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # crossing 判斷點：底邊中心
            # 將 YOLO 偵測框轉為 (x1, y1, x2, y2)
            det_box = (x1, y1, x2, y2)

            # 將使用者畫的區域轉為標準 box 格式
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

            # 改用重疊判斷
            if tid not in counted_left and is_overlap(det_box, area0_box):
                cross_left += 1
                counted_left.add(tid)

            if tid not in counted_right and is_overlap(det_box, area1_box):
                cross_right += 1
                counted_right.add(tid)


    # 畫出兩個統計區域的框
    cv2.rectangle(frame, areas[0][0], areas[0][1], (0,255,0), 2)
    cv2.rectangle(frame, areas[1][0], areas[1][1], (255,0,0), 2)

    # 顯示計數資訊
    cv2.putText(frame, f"Left crossing:  {cross_left}",  (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    cv2.putText(frame, f"Right crossing: {cross_right}", (30,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)

    # 顯示影像
    cv2.imshow("YOLOv8 Crossing Counter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 離開
        break
    if key == ord('r'):  # 按 R 重置統計
        tracks.clear()
        counted_left.clear()
        counted_right.clear()
        cross_left = cross_right = 0
        print("已重置計數")

# ---------- 收尾 ----------
cap.release()
cv2.destroyAllWindows()