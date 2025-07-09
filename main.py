
# main.py
# 主要流程：初始化設備與參數 -> 取得來源畫面 -> 手動畫選區域 -> 初始化模型 -> 逐幀推論與計數顯示

import os
import cv2                              # OpenCV：處理影像與視窗顯示
from ultralytics import YOLO            # Ultralytics YOLOv8：物件偵測+追蹤
import numpy as np                      # Numpy：數值運算（可用於影像資料處理）
import itertools                        # itertools：產生自動遞增ID
import sys                              # sys：系統級操作，如結束程式
import argparse                         # argparse：命令列參數解析
import torch                            # torch：PyTorch深度學習框架

# 解決部分環境下OpenMP重複載入的錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --------------------------
# 裝置與執行環境初始化
# --------------------------
def get_device():
    """
    檢查本機是否有CUDA（GPU）可用。
    若有則回傳GPU，否則用CPU。
    並印出目前運算裝置。
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU 已啟用：", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("未偵測到 GPU，使用 CPU 模式")
    return device

# --------------------------
# 解析命令列參數（由GUI或Terminal呼叫時用）
# --------------------------
def parse_args():
    """
    支援兩種輸入模式：
      --video [影片路徑]
      --camera (攝影機，預設裝置1)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='影片路徑')
    parser.add_argument('--camera', action='store_true', help='使用攝影機')
    return parser.parse_args()

# --------------------------
# 開啟來源（影片或攝影機）
# --------------------------
def open_capture(args):
    """
    根據用戶選擇開啟影片檔案或攝影機。
    失敗時結束程式。
    """
    if args.video:
        cap = cv2.VideoCapture(args.video)
    elif args.camera:
        cap = cv2.VideoCapture(1)  # 攝影機通常外接為1，內建為0（可依需求修改）
        if not cap.isOpened():
            print("攝影機開啟失敗，請確認裝置是否存在、權限允許，或是否被其他程式佔用")
            sys.exit()
    else:
        print("請指定影片或攝影機作為來源")
        sys.exit()
    return cap

# --------------------------
# 取得影片/攝影機的第一幀畫面（用於手動畫區域）
# --------------------------
def get_first_frame(cap):
    """
    讀取來源的第一幀畫面（單張影像）。
    若失敗（來源無畫面），釋放資源並結束程式。
    """
    ret, frame = cap.read()
    if not ret or frame is None:
        print("無法讀取第一幀影像，可能是來源無畫面或裝置未準備好")
        cap.release()
        sys.exit()
    return frame

# --------------------------
# 互動畫框區域，回傳多個區域座標與每個區域對應的顏色
# --------------------------
def draw_rect_setup(NUM_AREAS, first_frame):
    """
    讓使用者用滑鼠在第一幀畫出NUM_AREAS個矩形區域。
    按Enter確認，C清除，ESC離開。
    回傳：
      - areas: List of [(左上, 右下), ...]
      - area_palette: 顏色配色表
    """
    drawing = False                      # 是否正在畫
    areas = []                           # 存所有選取區域的座標
    temp = []                            # 暫存當前畫的區域座標
    area_palette = [
        (255, 0, 0),    # 藍（上）
        (0, 165, 255),  # 橙（下）
        (0, 255, 255),  # 黃（左）
        (128, 0, 128),  # 紫（右）
    ]

    # OpenCV 滑鼠Callback
    def draw_rect(event, x, y, flags, param):
        nonlocal drawing, temp, areas
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
        # 畫出所有已選區域
        for rect in areas:
            cv2.rectangle(disp, rect[0], rect[1], (0, 255, 0), 2)
        # 畫出正在選的區域（紅色框）
        if len(temp) == 2:
            cv2.rectangle(disp, temp[0], temp[1], (0, 0, 255), 2)
        # 指示文字
        cv2.putText(disp, "Press Enter: confirm, C: clear", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Draw Areas", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC鍵：退出程式
            cv2.destroyAllWindows()
            sys.exit()
        if key == ord('c'):
            areas.clear()
            temp.clear()
            print("已清除畫面，請重新繪製區域")
        if key == 13 and len(areas) == NUM_AREAS:  # Enter：完成
            break

    cv2.destroyWindow("Draw Areas")
    print("區域座標：", areas)
    return areas, area_palette

# --------------------------
# 判斷兩個矩形（偵測框與區域）有無重疊
# --------------------------
def is_overlap(box1, box2):
    """
    判斷兩個矩形是否有重疊（單純做框與框重疊）。
    box格式：(x1, y1, x2, y2)
    """
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return inter_x1 < inter_x2 and inter_y1 < inter_y2

# --------------------------
# 主流程
# --------------------------
def main():
    MODEL_PATH = "modelv1.pt"               # 訓練好的YOLOv8模型路徑
    NUM_AREAS = 4                           # 區域數

    device = get_device()                   # 取得GPU或CPU
    args = parse_args()                     # 解析參數
    cap = open_capture(args)                # 開啟來源（影片或攝影機）
    first_frame = get_first_frame(cap)      # 取得第一幀（畫框用）

    # 互動選擇區域
    areas, area_palette = draw_rect_setup(NUM_AREAS, first_frame)

    # 載入YOLO模型
    model = YOLO(MODEL_PATH).to(device)
    names = model.names
    yolo_palette = [
        (0, 255, 0),      # 綠（car）
        (0, 0, 255),      # 紅（bus）
        (255, 255, 0),    # 青藍（motor）
        (255, 0, 255),    # 粉紅（truck）
    ]

    # 初始化計數結構
    tracks = {}  # (暫未用，預留進階功能)
    counted = [set() for _ in range(NUM_AREAS)]  # 各區域已計數track_id集合
    cross = [0 for _ in range(NUM_AREAS)]        # 各區域累計跨越數

    # --------------------------
    # 進入逐幀推論/計數主迴圈
    # --------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 結束

        # 物件偵測+追蹤
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        boxes = results[0].boxes

        if boxes is not None:
            # 取得各追蹤物件的ID、類別、信心分數
            ids = boxes.id.cpu().numpy() if boxes.id is not None else itertools.count()
            classes = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            # 對所有物件進行處理
            for box, tid, cls, conf in zip(boxes.xyxy.cpu().numpy(), ids, classes, confs):
                if conf < 0.4:
                    continue  # 跳過低信心

                tid = int(tid)
                label = f"{names[int(cls)]} {conf:.2f}"
                color = yolo_palette[int(cls) % len(yolo_palette)]

                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # 中心點（進階可用於判斷方向）

                # 畫出物件偵測框與標籤
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                det_box = (x1, y1, x2, y2)

                # 檢查此track_id有沒有進過各區域（只計一次）
                for i in range(NUM_AREAS):
                    # 計算當前區域的四邊界
                    area_box = (
                        min(areas[i][0][0], areas[i][1][0]),
                        min(areas[i][0][1], areas[i][1][1]),
                        max(areas[i][0][0], areas[i][1][0]),
                        max(areas[i][0][1], areas[i][1][1])
                    )

                    # 若這個物件（track id）未被本區域計數過，且有重疊，則+1
                    if tid not in counted[i] and is_overlap(det_box, area_box):
                        cross[i] += 1
                        counted[i].add(tid)

        # 畫出每個區域框線和計數文字
        for i in range(NUM_AREAS):
            color = area_palette[i]
            cv2.rectangle(frame, areas[i][0], areas[i][1], color, 2)
            cv2.putText(frame, f"Area {i + 1} crossing: {cross[i]}", (30, 40 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # 顯示推論畫面
        cv2.imshow("YOLOv8 Crossing Counter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc結束
            break
        if key == ord('r'):  # r重置所有計數與紀錄
            tracks.clear()
            for s in counted:
                s.clear()
            for i in range(NUM_AREAS):
                cross[i] = 0
            print("已重置計數")

    # --------------------------
    # 結束時釋放所有資源
    # --------------------------
    cap.release()
    cv2.destroyAllWindows()

# 程式進入點
if __name__ == '__main__':
    main()

