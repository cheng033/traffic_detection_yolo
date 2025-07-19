import os
import cv2
from ultralytics import YOLO
import numpy as np
import itertools
import sys
import argparse
import torch
from selenium_capture import SeleniumCapture

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_device():
    """檢查CUDA/GPU狀態，回傳device物件"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU 已啟用：", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("未偵測到 GPU，使用 CPU 模式")
    return device

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='影片路徑')
    parser.add_argument('--camera', action='store_true', help='使用攝影機')
    parser.add_argument('--url', type=str, help='即時網頁攝影機網址')
    return parser.parse_args()

def open_capture(args):
    """根據參數開啟影片或攝影機，回傳 cap 物件"""
    if args.video:
        cap = cv2.VideoCapture(args.video)
    elif args.camera:
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("攝影機開啟失敗，請確認裝置是否存在、權限允許，或是否被其他程式佔用")
            sys.exit()
    elif args.url:
        cap = SeleniumCapture(args.url)
        cap._is_selenium = True
        sys.exit()
    elif args.url:
        cap = SeleniumCapture(args.url)
        cap._is_selenium = True
        return cap
    else:
        print("請指定影片或攝影機作為來源")
        sys.exit()
    return cap

def open_captur(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        cap._is_selenium = False
        return cap
    elif args.camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("攝影機開啟失敗")
            exit()
        cap._is_selenium = False
        return cap
    elif args.url:
        cap = SeleniumCapture(args.url)
        cap._is_selenium = True
        return cap
    else:
        print("請指定影片或攝影機作為來源")
        exit()

def get_first_frame(cap):
    """取得第一幀影像，失敗則結束"""
    ret, frame = cap.read()
    if not ret or frame is None:
        print("無法讀取第一幀影像，可能是來源無畫面或裝置未準備好")
        cap.release()
        sys.exit()
    return frame

def draw_rect_setup(NUM_AREAS, first_frame):
    """用戶畫偵測區域，回傳區域清單與顏色配置"""
    drawing = False
    areas = []
    temp = []
    area_palette = [
        (255, 0, 0),    # 藍（左）
        (0, 165, 255),  # 橙（上）
        (0, 255, 255),  # 黃（右）
        (128, 0, 128),  # 紫（下）
    ]

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
        for idx, rect in enumerate(areas):
            cv2.rectangle(disp, rect[0], rect[1], area_palette[idx % len(area_palette)], 2)
            cv2.putText(disp, f"Area {idx}", rect[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, area_palette[idx % len(area_palette)], 2)
        if len(temp) == 2:
            cv2.rectangle(disp, temp[0], temp[1], (0, 0, 255), 2)
        cv2.putText(disp, "Press Enter: confirm, C: clear", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Draw Areas", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
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
    return areas, area_palette

def is_overlap(box1, box2):
    """判斷兩個矩形框是否重疊"""
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return inter_x1 < inter_x2 and inter_y1 < inter_y2

def get_direction(entry, exit):
    """
    根據入口和出口的區域編號，回傳方向（左轉、右轉、直行）
    這裡假設區域 0=左, 1=上, 2=右, 3=下，請依實際區域調整！
    """
    # 左邊進，上邊出 = 右轉
    if entry == 0 and exit == 1:
        return '右轉'
    # 左邊進，下邊出 = 左轉
    elif entry == 0 and exit == 3:
        return '左轉'
    # 左邊進，右邊出 = 直行
    elif entry == 0 and exit == 2:
        return '直行'
    # 上邊進，右邊出 = 右轉
    elif entry == 1 and exit == 2:
        return '右轉'
    # 上邊進，左邊出 = 左轉
    elif entry == 1 and exit == 0:
        return '左轉'
    # 上邊進，下邊出 = 直行
    elif entry == 1 and exit == 3:
        return '直行'
    # 右邊進，下邊出 = 右轉
    elif entry == 2 and exit == 3:
        return '右轉'
    # 右邊進，上邊出 = 左轉
    elif entry == 2 and exit == 1:
        return '左轉'
    # 右邊進，左邊出 = 直行
    elif entry == 2 and exit == 0:
        return '直行'
    # 下邊進，左邊出 = 右轉
    elif entry == 3 and exit == 0:
        return '右轉'
    # 下邊進，右邊出 = 左轉
    elif entry == 3 and exit == 2:
        return '左轉'
    # 下邊進，上邊出 = 直行
    elif entry == 3 and exit == 1:
        return '直行'
    else:
        return '其他'

# ...[前略，與上個版本相同，僅主流程與方向計數邏輯更新]...

def main():
    MODEL_PATH = "modelv1.pt"
    NUM_AREAS = 4

    device = get_device()
    args = parse_args()
    cap = open_capture(args)
    first_frame = get_first_frame(cap)
    areas, area_palette = draw_rect_setup(NUM_AREAS, first_frame)

    model = YOLO(MODEL_PATH).to(device)
    names = model.names
    yolo_palette = [
        (0, 255, 0),      # 綠
        (0, 0, 255),      # 紅
        (255, 255, 0),    # 青藍
        (255, 0, 255),    # 粉紅
    ]

    # --- 每個出口的三方向統計 ---
    entry_area = {}      # {track_id: 入口}
    counted_turn = set() # {track_id}
    stats = [
        {"左轉": 0, "右轉": 0, "直行": 0} for _ in range(NUM_AREAS)
    ]

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
                det_box = (x1, y1, x2, y2)

                # 判斷各區域
                for i in range(NUM_AREAS):
                    area_box = (
                        min(areas[i][0][0], areas[i][1][0]),
                        min(areas[i][0][1], areas[i][1][1]),
                        max(areas[i][0][0], areas[i][1][0]),
                        max(areas[i][0][1], areas[i][1][1])
                    )
                    if is_overlap(det_box, area_box):
                        if tid not in entry_area:
                            entry_area[tid] = i
                        elif tid not in counted_turn and i != entry_area[tid]:
                            direction = get_direction(entry_area[tid], i)
                            if direction in stats[i]:
                                stats[i][direction] += 1
                            counted_turn.add(tid)
                            print(f"車輛 {tid} 從 {entry_area[tid]} 到 {i}，出口 {i} 方向: {direction}")

                # 畫偵測框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 顯示每個出口區域的三方向計數
        for i in range(NUM_AREAS):
            color = area_palette[i]
            cv2.rectangle(frame, areas[i][0], areas[i][1], color, 2)
            txt = f"出口{i} 左:{stats[i]['左轉']} 右:{stats[i]['右轉']} 直:{stats[i]['直行']}"
            cv2.putText(frame, txt, (areas[i][0][0], areas[i][0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("YOLOv8 Crossing Counter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('r'):
            entry_area.clear()
            counted_turn.clear()
            for s in stats:
                s["左轉"] = s["右轉"] = s["直行"] = 0
            print("已重置計數")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


