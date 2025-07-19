import cv2
from ultralytics import YOLO
import numpy as np
import torch
import itertools
import json

# 區域對應方向邏輯（可自定義）
def get_direction(entry, exit):
    """
    根據入口和出口的區域編號，回傳方向（左轉、右轉、直行）
    假設區域 0=左, 1=上, 2=右, 3=下
    """
    if entry == 0 and exit == 1:
        return '右轉'
    elif entry == 0 and exit == 3:
        return '左轉'
    elif entry == 0 and exit == 2:
        return '直行'
    elif entry == 1 and exit == 2:
        return '右轉'
    elif entry == 1 and exit == 0:
        return '左轉'
    elif entry == 1 and exit == 3:
        return '直行'
    elif entry == 2 and exit == 3:
        return '右轉'
    elif entry == 2 and exit == 1:
        return '左轉'
    elif entry == 2 and exit == 0:
        return '直行'
    elif entry == 3 and exit == 0:
        return '右轉'
    elif entry == 3 and exit == 2:
        return '左轉'
    elif entry == 3 and exit == 1:
        return '直行'
    else:
        return '其他'

def is_overlap(box1, box2):
    """判斷兩個矩形框是否重疊"""
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return inter_x1 < inter_x2 and inter_y1 < inter_y2

class YoloCrossCounter:
    def __init__(self, mode, video_path=None, url=None, areas=None, model_path="modelv1.pt", num_areas=4):
        # Device 設定
        print("初始化 YoloCrossCounter", mode, video_path, url)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 載入 YOLOv8 模型
        self.model = YOLO(model_path).to(self.device)
        self.names = self.model.names
        # 輸入來源初始化
        if mode == 'video':
            self.cap = cv2.VideoCapture(video_path)
            print("影片來源開啟狀態", self.cap.isOpened())
        elif mode == 'camera':
            self.cap = cv2.VideoCapture(0)
        elif mode == 'url':
            self.cap = cv2.VideoCapture(url)
        else:
            raise Exception("Invalid mode")
        # 偵測區域處理
        self.areas = json.loads(areas) if isinstance(areas, str) else areas
        self.num_areas = num_areas
        # 統計用變數
        self.entry_area = {}      # {track_id: entry_area_idx}
        self.counted_turn = set() # 已經計算過的 track_id
        self.stats = [
            {"左轉": 0, "右轉": 0, "直行": 0} for _ in range(self.num_areas)
        ]
        self.colors = [
            (255, 0, 0),    # 藍
            (0, 165, 255),  # 橙
            (0, 255, 255),  # 黃
            (128, 0, 128),  # 紫
        ]
        self.yolo_palette = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
        ]

    def run(self):
        """生成每一幀畫面，供 Flask video_feed 用"""
        print("進入run()迴圈")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")
            boxes = results[0].boxes

            if boxes is not None:
                ids = boxes.id.cpu().numpy() if boxes.id is not None else itertools.count()
                classes = boxes.cls.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                for box, tid, cls, conf in zip(boxes.xyxy.cpu().numpy(), ids, classes, confs):
                    if conf < 0.4:
                        continue
                    tid = int(tid)
                    label = f"{self.names[int(cls)]} {conf:.2f}"
                    color = self.yolo_palette[int(cls) % len(self.yolo_palette)]
                    x1, y1, x2, y2 = map(int, box)
                    det_box = (x1, y1, x2, y2)

                    # 判斷進入哪個區域
                    for i, area in enumerate(self.areas):
                        # area = [[x1, y1], [x2, y2]]
                        area_box = (
                            min(area[0][0], area[1][0]),
                            min(area[0][1], area[1][1]),
                            max(area[0][0], area[1][0]),
                            max(area[0][1], area[1][1])
                        )
                        if is_overlap(det_box, area_box):
                            # 記錄入口
                            if tid not in self.entry_area:
                                self.entry_area[tid] = i
                            # 進出口不同且尚未計算，才計數
                            elif tid not in self.counted_turn and i != self.entry_area[tid]:
                                direction = get_direction(self.entry_area[tid], i)
                                if direction in self.stats[i]:
                                    self.stats[i][direction] += 1
                                self.counted_turn.add(tid)
                                # print(f"車輛 {tid} 入口 {self.entry_area[tid]}，出口 {i}，方向: {direction}")

                    # 畫 YOLO 偵測框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 畫區域與即時計數
            for i, area in enumerate(self.areas):
                color = self.colors[i % len(self.colors)]
                cv2.rectangle(frame, tuple(area[0]), tuple(area[1]), color, 2)
                txt = f"出口{i} 左:{self.stats[i]['左轉']} 右:{self.stats[i]['右轉']} 直:{self.stats[i]['直行']}"
                cv2.putText(frame, txt, (area[0][0], area[0][1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            yield frame

    def get_stats(self):
        """回傳即時統計數據（dict 格式，方便 Flask 回傳 json）"""
        return {f"出口{i}": self.stats[i] for i in range(self.num_areas)}


