import cv2                       # OpenCV，用於攝影機畫面
from selenium_capture import SeleniumCapture  # 你的 selenium 網頁截圖模組
import numpy as np
import time
from flask import Flask, render_template, request, Response, jsonify, send_file
import threading
import itertools
from ultralytics import YOLO
import torch
import io
import csv
import json



# 建立 Flask 應用實例
app = Flask(__name__)

# 首頁路由：渲染 index.html（要放在 templates 資料夾）
@app.route('/')
def index():
    return render_template('index.html')

# 攝影機預覽路由：回傳本地攝影機第一幀
@app.route('/preview_camera')
def preview_camera():
    cap = cv2.VideoCapture(0)            # 開啟攝影機
    time.sleep(0.5)                      # 等待攝影機預熱
    ret, img = cap.read()                # 讀取一張影像
    cap.release()                        # 關閉攝影機
    if not ret or img is None:
        return "無法取得攝影機畫面", 500
    _, buf = cv2.imencode('.jpg', img)   # 轉 JPEG 格式
    return Response(buf.tobytes(), content_type='image/jpeg')

# 網頁串流預覽路由：用 Selenium 擷取一張網頁畫面
@app.route('/preview_url')
def preview_url():
    url = request.args.get("url")            # 取得前端傳來的網址
    cap = SeleniumCapture(url)               # 用 Selenium 開網頁
    time.sleep(2)                            # 等網頁載入
    ret, img = cap.read()                    # 擷取一張影像
    cap.release()                            # 關閉 Selenium driver
    if not ret or img is None:
        return "無法取得串流畫面", 500
    _, buf = cv2.imencode('.jpg', img)       # 轉 JPEG 格式
    return Response(buf.tobytes(), content_type='image/jpeg')


# ========== 全域辨識狀態 ==========
yolo_thread = None        # YOLO 執行緒物件
yolo_running = False      # 控制執行緒開關
yolo_frame = None         # 儲存即時畫面（np array）
yolo_stats = None         # 儲存即時統計數據
yolo_lock = threading.Lock()   # 多執行緒安全鎖
yolo_model = None         # YOLOv8 模型全域物件（避免每次都重新加載）

#建立一個初始化YOLO模型的工具函數，加快啟動速度
def get_yolo_model():
    global yolo_model
    if yolo_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        yolo_model = YOLO("modelv1.pt").to(device)
    return yolo_model

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

def yolo_worker(capture_type, source, areas):
    """
    capture_type: "video", "camera", "url"
    source: 路徑 or url
    areas: list，前端給的偵測線
    """
    global yolo_running, yolo_frame, yolo_stats

    # --- 模型初始化 ---
    model = get_yolo_model()
    names = model.names
    yolo_palette = [(0,255,0),(0,0,255),(255,255,0),(255,0,255)]
    area_palette = [(255,0,0),(0,165,255),(0,255,255),(128,0,128)]

    # --- 判斷來源 ---
    if capture_type == "video":
        cap = cv2.VideoCapture(source)
    elif capture_type == "camera":
        cap = cv2.VideoCapture(0)
    elif capture_type == "url":
        cap = SeleniumCapture(source)
    else:
        print("未知來源類型")
        return

    NUM_AREAS = len(areas)
    entry_area = {}
    counted_turn = set()
    stats = [{"左轉":0, "右轉":0, "直行":0} for _ in range(NUM_AREAS)]

    while yolo_running:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # --- YOLO Tracking ---
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        boxes = results[0].boxes

        if boxes is not None:
            ids = boxes.id.cpu().numpy() if boxes.id is not None else itertools.count()
            classes = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for box, tid, cls, conf in zip(boxes.xyxy.cpu().numpy(), ids, classes, confs):
                if conf < 0.4: continue
                tid = int(tid)
                label = f"{names[int(cls)]} {conf:.2f}"
                color = yolo_palette[int(cls)%len(yolo_palette)]
                x1, y1, x2, y2 = map(int, box)
                det_box = (x1, y1, x2, y2)
                for i in range(NUM_AREAS):
                    line = areas[i]
                    # 這裡你原本是矩形，線的話要用線段-點距檢查（先略，假設仍用區塊）
                    area_box = (
                        min(line[0][0], line[1][0]),
                        min(line[0][1], line[1][1]),
                        max(line[0][0], line[1][0]),
                        max(line[0][1], line[1][1])
                    )
                    if is_overlap(det_box, area_box):
                        if tid not in entry_area:
                            entry_area[tid] = i
                        elif tid not in counted_turn and i != entry_area[tid]:
                            direction = get_direction(entry_area[tid], i)
                            if direction in stats[i]:
                                stats[i][direction] += 1
                            counted_turn.add(tid)
                # 畫偵測框
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label, (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 顯示統計
        for i in range(NUM_AREAS):
            color = area_palette[i]
            line = areas[i]
            cv2.line(frame, tuple(line[0]), tuple(line[1]), color, 3)
            txt = f"出口{i} 左:{stats[i]['左轉']} 右:{stats[i]['右轉']} 直:{stats[i]['直行']}"
            cv2.putText(frame, txt, (line[0][0], line[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- 更新全域狀態 ---
        with yolo_lock:
            yolo_frame = frame.copy()
            yolo_stats = [dict(s) for s in stats]

        time.sleep(0.03)   # 避免 CPU 過高

    cap.release()
    print("YOLO 執行緒結束")

@app.route('/start', methods=['POST'])
def start():
    print("收到 /start 請求")
    global yolo_thread, yolo_running
    # 若有執行中，先停掉
    if yolo_thread and yolo_thread.is_alive():
        yolo_running = False
        yolo_thread.join()

    # 取得前端傳來的設定
    input_mode = request.form['input_mode']
    areas = json.loads(request.form['areas'])
    source = None

    # 處理來源
    if input_mode == "video":
        file = request.files['video']
        video_path = "uploaded_video.mp4"
        file.save(video_path)
        source = video_path
    elif input_mode == "camera":
        source = 0
    elif input_mode == "url":
        source = request.form['video_url']
    else:
        return jsonify({"status": "ERR", "msg": "來源錯誤"})

    # 啟動 YOLO 執行緒
    yolo_running = True
    yolo_thread = threading.Thread(target=yolo_worker, args=(input_mode, source, areas), daemon=True)
    yolo_thread.start()

    return jsonify({"status": "OK"})

@app.route('/video_feed')
def video_feed():
    def gen():
        last_time = time.time()
        while True:
            with yolo_lock:
                frame = yolo_frame.copy() if yolo_frame is not None else None
            if frame is not None:
                _, buf = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            # 增加 sleep
            while time.time() - last_time < 1/12:  # 最多 12 FPS
                time.sleep(0.01)
            last_time = time.time()

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    with yolo_lock:
        stats = yolo_stats.copy() if yolo_stats is not None else []
    return jsonify(stats)

@app.route('/export_csv')
def export_csv():
    with yolo_lock:
        stats = yolo_stats.copy() if yolo_stats is not None else []
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["出口", "左轉", "右轉", "直行"])
    for i, s in enumerate(stats):
        writer.writerow([f"出口{i}", s.get("左轉",0), s.get("右轉",0), s.get("直行",0)])
    output.seek(0)
    return Response(output.getvalue(), mimetype='text/csv',
                    headers={"Content-Disposition": "attachment;filename=flow_stats.csv"})
print(app.url_map)

# 啟動 Flask 伺服器（要放在所有路由之後）
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)


