# yolo_direction.py
import cv2
import numpy as np
import sys
import datetime
# ----- 全域變數 -----
colors = [(0,0,255), (0,128,255), (0,255,255), (255,0,255)]  # 紅、橙、黃、紫
line_colors = ['Red', 'Orange', 'Yellow', 'Purple']
lines = []
temp_line = []
drawing = False
vehicle_tracks = {}
vehicle_direction_map = {}
FRAME_COUNT = 0 
line_pass_count = []

# ----- 工具 function -----
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

def infer_direction(start, end, total):
    diff = (end - start) % total
    if diff == 1:
        return 'L'
    elif diff == 2:
        return 'S'
    elif diff == 3:
        return 'R'
    return None

# ----- 互動畫線 -----
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
    cv2.namedWindow("Draw Lines")
    cv2.setMouseCallback("Draw Lines", draw_line_event)
    while True:
        disp = first_frame.copy()
        for i, line in enumerate(lines):
            cv2.line(disp, line[0], line[1], colors[i % 4], 2)
            mid = ((line[0][0]+line[1][0])//2, (line[0][1]+line[1][1])//2)
            cv2.putText(disp, f"{i}:{line_colors[i]}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i % 4], 2)
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

# ----- YOLO推論處理與方向計數 -----
def process_frame(frame, model, names):
    global vehicle_tracks, vehicle_direction_map, FRAME_COUNT
    FRAME_COUNT += 1

    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes = results[0].boxes

    info_to_record = []
    if boxes is None:
        return frame

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
            vehicle_tracks[tid] = {'class': class_name, 'lines': []}

        track = vehicle_tracks[tid]
        history = track['lines']

        if line_idx is not None and (not history or history[-1] != line_idx):
            history.append(line_idx)
            line_pass_count[line_idx] += 1

        if len(history) >= 2:
            start, end = history[0], history[-1]
            if start != end:
                direction = infer_direction(start, end, len(lines))
                key = (start, end)
                if direction:
                    if key not in vehicle_direction_map:
                        vehicle_direction_map[key] = {'motor':0, 'car':0, 'truck':0, 'bus':0, 'dir': direction}
                    vehicle_direction_map[key][class_name] += 1
                    print(f"[+1] {class_name} id={tid} 方向: {direction} ({start}->{end})")
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    info_to_record.append([tid, class_name, direction, timestamp])

                vehicle_tracks[tid]['lines'] = []

        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        label = f"{class_name} {tid} ({conf:.2f})"
        cv2.circle(frame, center, 4, (255, 255, 255), -1)
        cv2.putText(frame, label, (center[0]+5, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return frame, info_to_record

def draw_result_overlay(frame):
    for i, (p1, p2) in enumerate(lines):
        cv2.line(frame, p1, p2, colors[i % 4], 2)
        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        cv2.putText(frame, f"{i}:{line_colors[i]}  ({line_pass_count[i]})", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i % 4], 2)

    y0 = 30
    for idx, ((start, end), data) in enumerate(vehicle_direction_map.items()):
        total = data['motor'] + data['car'] + data['truck'] + data['bus']
        text = f"{start}->{end} ({data['dir']}): "
        for cls in ['motor','car','truck','bus']:
            text += f"{cls[0]}:{data[cls]} "
        color = colors[start % len(colors)]
        cv2.putText(frame, text, (30, y0 + idx*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
