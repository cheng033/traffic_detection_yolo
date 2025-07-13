# main.py
# 主要流程：初始化設備與參數 -> 取得來源畫面 -> 手動畫選區域 -> 初始化模型 -> 逐幀推論與計數顯示

import os
import cv2
import torch
import argparse
from ultralytics import YOLO
from yolo_direction import draw_lines_interface, process_frame, draw_result_overlay
from selenium_capture import SeleniumCapture

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_device():
    if torch.cuda.is_available():
        print("GPU 已啟用：", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("使用 Apple MPS")
        return torch.device("mps")
    else:
        print("未偵測到 GPU/MPS，使用 CPU")
        return torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='影片路徑')
    parser.add_argument('--camera', action='store_true', help='使用攝影機')
    parser.add_argument('--url', type=str, help='即時網頁攝影機網址')
    return parser.parse_args()

def open_capture(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        cap._is_selenium = False
        return cap
    elif args.camera:
        cap = cv2.VideoCapture(1)
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
    

def main():
    device = get_device()
    args = parse_args()
    cap = open_capture(args)
    if hasattr(cap, '_is_selenium') and cap._is_selenium:
        ret, first_frame = cap.read()
    else:
        ret, first_frame = cap.read()

    if not ret or first_frame is None:
        print("無法取得來源畫面")
        cap.release()
        exit()

    # 互動畫線
    draw_lines_interface(first_frame)

    # 載入YOLO
    model = YOLO("modelv8.pt").to(device)
    names = model.names
    yolo_palette = [
        (0, 255, 0),      # 綠（car）
        (0, 0, 255),      # 紅（bus）
        (255, 255, 0),    # 青藍（motor）
        (255, 0, 255),    # 粉紅（truck）
    ]

    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / fps) if fps and fps > 0 else 33
    while True:
        if hasattr(cap, '_is_selenium') and cap._is_selenium:
            ret, frame = cap.read()
        else:
            ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame, model, names)
        draw_result_overlay(frame)
        cv2.imshow("YOLO Crossing Direction", frame)
        if cv2.waitKey(wait_time) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
