import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import torch

# ---------- Apple M1/M2 GPU 支援 ----------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ 使用裝置：{device}")

# ---------- 全域變數 ----------
colors = [(0,0,255), (0,128,255), (0,255,255), (255,0,255)]  # 紅、橙、黃、紫
line_colors = ['Red', 'Orange', 'Yellow', 'Purple']
lines = []
temp_line = []
drawing = False
vehicle_tracks = {}
vehicle_direction_map = {}
FRAME_COUNT = 0 
line_pass_count = []

# ---------- 工具函數 ----------
def point_to_line_distance(p, a, b):
    a, b, p = np.array(a), np.array(b), np.array(p)
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def get_line_index_bbox_and_center(x1, y1, x2, y2, threshold=30):
    points = [
        (x1, y1), (x1, y2), (x2, y1), (x2, y2),
        ((x1 + x2) // 2, (y1 + y2) // 2)
    ]
    for i, (p1, p2) in enumerate(lines):
        for pt in points:
            if point_to_line_distance(pt, p1, p2) < threshold:
                return i
    return None

# ---------- 畫線介面 ----------
def draw_line_event(event, x, y, flags, param):
    global drawing, temp_line, lines
    if event == cv2.EVENT_LBUTTONDOWN and len(lines) < 4:
        drawing = True
        temp_line = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        temp_line.append((x, y))
        if len(temp_line) == 2:
            lines.append(tuple(temp_line))
            temp_line = []

def draw_lines_interface(first_frame):
    global line_pass_count
    print("\n📌 提醒：請順時針方向畫線，例如：上→右→下→左，對應編號為 0, 1, 2, 3")
    cv2.namedWindow("Draw Lines")
    cv2.setMouseCallback("Draw Lines", draw_line_event)
    while True:
        disp = first_frame.copy()
        for i, line in enumerate(lines):
            cv2.line(disp, line[0], line[1], colors[i % 4], 2)
            mid = ((line[0][0]+line[1][0])//2, (line[0][1]+line[1][1])//2)
            cv2.putText(disp, f"{i}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i % 4], 2)
        if len(temp_line) == 2:
            cv2.line(disp, temp_line[0], temp_line[1], (255, 255, 255), 2)

        cv2.imshow("Draw Lines", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            sys.exit()
        if key == ord('c'):
            if temp_line:
                temp_line.clear()
            elif lines:
                lines.pop()
        if key == 13 and len(lines) >= 2:
            line_pass_count = [0] * len(lines)
            break
    cv2.destroyWindow("Draw Lines")

    # ---------- 建立偵測矩形區域 ----------
    global rect_zone
    all_points = [pt for line in lines for pt in line]
    xs, ys = zip(*all_points)
    rx, ry = min(xs), min(ys)
    rw, rh = max(xs) - rx, max(ys) - ry
    rect_zone = (rx, ry, rw, rh)

# ---------- 車輛離開矩形區域就統計 ----------
# 🚩 僅記錄車輛離開矩形區域後的「第一條線」與「最後一條線」，其他中途線不統計

def cleanup_lost_vehicles(current_frame, timeout=30):
    global vehicle_tracks, vehicle_direction_map
    to_delete = []
    for tid, track in vehicle_tracks.items():
        if track.get('left_zone', False):
            start = track['start_line']
            end = track['last_line']
            if start in [0, 2] and end in [0, 1, 2, 3] and start != end:
                key = (start, end)
                if key not in vehicle_direction_map:
                    vehicle_direction_map[key] = {'motor':0, 'car':0, 'truck':0, 'bus':0}
                vehicle_direction_map[key][track['class']] += 1
                print(f"[+1] {track['class']} id={tid} 從線 {start} 到線 {end}")
            to_delete.append(tid)
    for tid in to_delete:
        del vehicle_tracks[tid]

# ---------- 主程式入口 ----------
def choose_video_file():
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="選擇影片檔案",
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )
    return video_path

if __name__ == "__main__":
    video_path = choose_video_file()
    if not video_path:
        print("❌ 未選擇影片，程式結束")
        sys.exit()

    model = YOLO("yolov8n.pt").to(device)
    names = model.names

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 無法打開影片")
        sys.exit()

    ret, first_frame = cap.read()
    if not ret:
        print("❌ 無法讀取影片第一幀")
        sys.exit()

    draw_lines_interface(first_frame)

    while True:
        ret, frame = cap.read()
        FRAME_COUNT += 1  # ✅ 每幀遞增，以便偵測是否離開畫面()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        boxes = results[0].boxes

        ids = boxes.id.to("cpu").numpy() if boxes.id is not None else []
        classes = boxes.cls.to("cpu").numpy()
        confs = boxes.conf.to("cpu").numpy() if boxes.conf is not None else [0.0]*len(ids)

        for box, tid, cls, conf in zip(boxes.xyxy.to("cpu").numpy(), ids, classes, confs):
            tid = int(tid)
            class_name = names[int(cls)]
            if class_name not in ['motor', 'car', 'truck', 'bus']:
                continue

            x1, y1, x2, y2 = map(int, box)
            line_idx = get_line_index_bbox_and_center(x1, y1, x2, y2, threshold=30)

            if tid not in vehicle_tracks:
                vehicle_tracks[tid] = {
                    'class': class_name,
                    'start_line': None,
                    'last_line': None,
                    'last_seen': FRAME_COUNT
                }

            track = vehicle_tracks[tid]
            track['last_seen'] = FRAME_COUNT

            # ---------- 判斷是否離開過矩形區域 ----------
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cx, cy = center
            rx, ry, rw, rh = rect_zone
            if not (rx <= cx <= rx + rw and ry <= cy <= ry + rh):
                track['left_zone'] = True

            if line_idx is not None:
                if track['start_line'] is None:
                    track['start_line'] = line_idx
                    print(f"🚗 ID={tid} 起點設定為線 {line_idx}")
                track['last_line'] = line_idx

            color_map = {
                'car': (255, 0, 0),
                'motor': (0, 255, 0),
                'bus': (128, 128, 128),
                'truck': (0, 0, 255)
            }
            label_color = color_map.get(class_name, (255, 255, 255))
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            label = f"{class_name} {tid} ({conf:.2f})"
            cv2.circle(frame, center, 4, label_color, -1)
            cv2.putText(frame, label, (center[0]+5, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

        cleanup_lost_vehicles(FRAME_COUNT)

        # ---------- 統計顯示 ----------
        for i, (p1, p2) in enumerate(lines):
            cv2.line(frame, p1, p2, colors[i % 4], 2)
            mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
            cv2.putText(frame, f"{i}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i % 4], 2)

        y0 = 30
        for idx, ((start, end), data) in enumerate(vehicle_direction_map.items()):
            text = f"{start}->{end}: "
            for cls in ['motor','car','truck','bus']:
                text += f"{cls[0]}:{data[cls]} "
            color = colors[start % len(colors)]
            cv2.putText(frame, text, (30, y0 + idx*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("YOLO Direction Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
