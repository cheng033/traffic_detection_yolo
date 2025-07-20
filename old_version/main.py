import os
import sys
import cv2
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
from old_version.yolo_direction import draw_lines_interface, colors
from old_version.selenium_capture import SeleniumCapture
import datetime


# 全域紀錄
vehicle_tracks = {}
vehicle_direction_map = {}

# ------------------ 工具函數 ------------------
def point_to_line_distance(p, a, b):
    import numpy as np
    a, b, p = np.array(a), np.array(b), np.array(p)
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def get_line_index_bbox_and_center(x1, y1, x2, y2, lines, threshold=30):
    points = [
        (x1, y1), (x1, y2), (x2, y1), (x2, y2),
        ((x1 + x2) // 2, (y1 + y2) // 2)
    ]
    for i, (p1, p2) in enumerate(lines):
        for pt in points:
            if point_to_line_distance(pt, p1, p2) < threshold:
                return i
    return None

def cleanup_lost_vehicles(names):
    global vehicle_tracks, vehicle_direction_map
    to_delete = []
    for tid, track in vehicle_tracks.items():
        if track.get('left_zone', False):
            start = track['start_line']
            end = track['last_line']
            if start is not None and end is not None and start != end:
                key = (start, end)
                if key not in vehicle_direction_map:
                    # 動態建立計數字典：使用模型 names 中的 0,1,2,3
                    vehicle_direction_map[key] = {names[i]: 0 for i in [0, 1, 2, 3]}
                if track['class'] in vehicle_direction_map[key]:
                    vehicle_direction_map[key][track['class']] += 1
                else:
                    print(f"⚠ 未知類別 {track['class']}, 請確認 model.names")
                print(f"[+1] {track['class']} id={tid} 從線 {start} 到線 {end}")
            to_delete.append(tid)
    for tid in to_delete:
        del vehicle_tracks[tid]
# ------------------ GUI 選擇來源 ------------------
def choose_video():
    path = filedialog.askopenfilename(
        title="選擇影片",
        filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")]
    )
    if path:
        video_path.set(path)
        status_label.config(text=f"已選擇影片：{os.path.basename(path)}")
    else:
        video_path.set("")
        status_label.config(text="尚未選擇影片")

def start_detection():
    mode = input_mode.get()
    video = video_path.get()
    url = url_input.get()

    # 關閉 GUI 視窗，開始跑主程式
    root.destroy()

    # 決定來源
    if mode == "video":
        cap = cv2.VideoCapture(video)
    elif mode == "camera":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("錯誤", "無法開啟攝影機")
            return
    elif mode == "url":
        cap = SeleniumCapture(url)
    else:
        messagebox.showerror("錯誤", "未選擇來源")
        return

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        print("無法取得來源畫面")
        cap.release()
        return

    # 畫線
    lines, rect_zone = draw_lines_interface(first_frame)

    # 載入 YOLO 模型
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")
    model = YOLO("modelv14.pt").to(device)
    names = model.names

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        boxes = results[0].boxes

        ids = boxes.id.to("cpu").numpy() if boxes.id is not None else []
        classes = boxes.cls.to("cpu").numpy()
        confs = boxes.conf.to("cpu").numpy() if boxes.conf is not None else [0.0]*len(ids)
        allowed_ids = [0, 1, 2, 3]
        for box, tid, cls, conf in zip(boxes.xyxy.to("cpu").numpy(), ids, classes, confs):
            cls_id = int(cls)
            if cls_id not in allowed_ids:
                continue

            class_name = names[cls_id]  # 從模型裡抓名稱
            tid = int(tid)

            x1, y1, x2, y2 = map(int, box)
            line_idx = get_line_index_bbox_and_center(x1, y1, x2, y2, lines)

            if tid not in vehicle_tracks:
                vehicle_tracks[tid] = {
                    'class': class_name,
                    'start_line': None,
                    'last_line': None,
                    'last_seen': frame_count
                }

            track = vehicle_tracks[tid]
            track['last_seen'] = frame_count

            cx, cy = (x1+x2)//2, (y1+y2)//2
            rx, ry, rw, rh = rect_zone
            if not (rx <= cx <= rx+rw and ry <= cy <= ry+rh):
                track['left_zone'] = True

            if line_idx is not None:
                if track['start_line'] is None:
                    track['start_line'] = line_idx
                    print(f"ID={tid} 起點設定為線 {line_idx}")
                track['last_line'] = line_idx

            color_map = {'car':(255,0,0), 'motorbike':(0,255,0), 'bus':(128,128,128), 'truck':(0,0,255)}
            label_color = color_map.get(class_name,(255,255,255))
            center = ((x1+x2)//2,(y1+y2)//2)
            label = f"{class_name} {tid} ({conf:.2f})"
            cv2.circle(frame, center, 4, label_color, -1)
            cv2.putText(frame, label, (center[0]+5, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

        cleanup_lost_vehicles(names)

        # 畫線和統計
        for i,(p1,p2) in enumerate(lines):
            cv2.line(frame,p1,p2,colors[i%4],2)
            mid=((p1[0]+p2[0])//2,(p1[1]+p2[1])//2)
            cv2.putText(frame,f"{i}",mid,cv2.FONT_HERSHEY_SIMPLEX,0.7,colors[i%4],2)

        y0 = 30
        for idx,((start,end),data) in enumerate(vehicle_direction_map.items()):
            text = f"{start}->{end}: "
            for cls in ['motorbike','car','truck','bus']:
                text += f"{cls[0]}:{data.get(cls,0)} "
            cv2.putText(frame,text,(30,y0+idx*25),cv2.FONT_HERSHEY_SIMPLEX,0.6,colors[start%len(colors)],2)

        cv2.imshow("YOLO Direction Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

#紀錄資訊


# ------------------ GUI 界面 ------------------
root = tk.Tk()
root.title("YOLOv8 計數系統 GUI")
root.geometry("450x330")
root.resizable(False, False)

input_mode = tk.StringVar(value="video")
video_path = tk.StringVar()
url_input = tk.StringVar()

tk.Label(root, text="請選擇輸入來源：", font=("Arial", 14)).pack(pady=10)

radio_frame = tk.Frame(root)
radio_frame.pack()

tk.Radiobutton(radio_frame, text="影片檔案", variable=input_mode, value="video").grid(row=0, column=0, sticky='w')
tk.Button(radio_frame, text="選擇影片", command=choose_video).grid(row=0, column=1, padx=10)

tk.Radiobutton(radio_frame, text="攝影機（即時）", variable=input_mode, value="camera").grid(row=1, column=0, sticky='w')

tk.Radiobutton(radio_frame, text="即時網頁影像", variable=input_mode, value="url").grid(row=2, column=0, sticky='w')
tk.Entry(radio_frame, textvariable=url_input, width=40).grid(row=2, column=1, padx=10)
tk.Label(radio_frame, text="範例:https://tw.live/cam/?id=NWT0052", fg="gray", font=("Arial", 8)).grid(row=3, column=1, sticky='w')

tk.Button(root, text="開始偵測", command=start_detection, bg="green", fg="white", font=("Arial", 12)).pack(pady=18)

status_label = tk.Label(root, text="尚未選擇影片或來源", fg="blue")
status_label.pack()

root.mainloop()
