import tkinter as tk                           # 導入 tkinter，Python 標準 GUI 工具包，tk 是常見別名
from tkinter import filedialog, messagebox     # 導入檔案對話框、訊息視窗（例如開啟檔案、彈出警告用）
from PIL import Image, ImageTk                 # 導入 PIL 圖像處理（Image：開啟/處理圖片；ImageTk：Tkinter 顯示圖片轉換）
import cv2                                     # OpenCV，主流影像處理與即時攝影機/影片讀取、處理套件
import numpy as np                             # numpy，主要用於陣列數值運算（OpenCV 與 YOLO 輸出多為 numpy array）
from old_version.selenium_capture import SeleniumCapture   # 匯入自訂的 SeleniumCapture 類別，提供網頁畫面截取（仿 cv2.VideoCapture 介面）
from ultralytics import YOLO                   # Ultralytics YOLOv8，先進的目標偵測/追蹤模型
import csv                                     # Python 標準 csv 檔案處理（流量資料匯出用）

class TrafficDetectionGUI:
    def __init__(self, root):                                 # 建構子，root 是 Tkinter 主視窗
        self.root = root
        self.root.title("車流量辨識系統")                     # 設定視窗標題

        # === 主畫面結構 ===
        self.frame_top = tk.Frame(root)                        # 主上層 Frame，左右分區用
        self.frame_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 影像顯示區（左邊）
        self.frame_video = tk.Frame(self.frame_top)            # 左側 Frame，放影像
        self.frame_video.pack(side=tk.LEFT, padx=10, pady=10)
        self.display_w = 800                                   # 顯示區寬度（canvas用，與所有座標相關）
        self.display_h = 450                                   # 顯示區高度
        self.canvas = tk.Canvas(self.frame_video, width=self.display_w, height=self.display_h, bg="#eee")
        self.canvas.pack()                                     # Canvas：影像＋畫線區

        # 控制/操作區（右邊）
        self.frame_controls = tk.Frame(self.frame_top)         # 右側 Frame，放所有控制按鈕/欄位
        self.frame_controls.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        # 來源選擇（本地影片、攝影機）
        self.btn_choose_video = tk.Button(self.frame_controls, text="選擇影片", width=18, command=self.choose_video)
        self.btn_choose_video.pack(pady=3)
        self.btn_camera = tk.Button(self.frame_controls, text="攝影機模式", width=18, command=self.choose_camera)
        self.btn_camera.pack(pady=3)

        # URL爬蟲區（在攝影機按鈕下方）
        self.url_var = tk.StringVar()                          # 用於儲存輸入框內容
        self.entry_url = tk.Entry(self.frame_controls, textvariable=self.url_var, width=20)
        self.entry_url.pack(pady=(6,2))
        self.btn_crawl = tk.Button(self.frame_controls, text="啟動爬蟲", width=18, command=self.choose_url)
        self.btn_crawl.pack(pady=2)

        # 線段設定（設定1~4條線）
        self.min_lines = 2                                    # 最少畫幾條線
        self.max_lines = 4                                    # 最多幾條線（可調整）
        self.line_set_buttons = []                            # 儲存線設定按鈕
        for i in range(self.max_lines):
            btn = tk.Button(
                self.frame_controls,
                text=f"設定線{i+1}",
                width=18,
                command=lambda idx=i: self.edit_line(idx),     # 綁定每個設定線N
                state=tk.DISABLED
            )
            btn.pack(pady=1)
            self.line_set_buttons.append(btn)

        # 其餘主控按鈕（全部清除、開始辨識、暫停、匯出、退出）
        self.btn_redraw = tk.Button(self.frame_controls, text="全部清除", width=18, command=self.clear_all_lines, state=tk.DISABLED)
        self.btn_redraw.pack(pady=3)
        self.btn_start = tk.Button(self.frame_controls, text="開始辨識", width=18, state=tk.DISABLED, command=self.start_detection)
        self.btn_start.pack(pady=3)
        self.btn_pause = tk.Button(self.frame_controls, text="暫停", width=18, state=tk.DISABLED, command=self.pause_detection)
        self.btn_pause.pack(pady=3)
        self.btn_export_csv = tk.Button(self.frame_controls, text="匯出CSV", width=18, state=tk.DISABLED, command=self.export_csv)
        self.btn_export_csv.pack(pady=3)
        self.btn_exit = tk.Button(self.frame_controls, text="退出", width=18, command=root.quit)
        self.btn_exit.pack(pady=3)

        # 統計表格（右側下方，顯示即時統計）
        self.lbl_table_title = tk.Label(self.frame_controls, text="流量統計表", font=("Arial", 12, "bold"))
        self.lbl_table_title.pack(anchor="w", padx=4, pady=(20,0))
        self.text_table = tk.Text(self.frame_controls, height=7, width=24)  # 右側小表格
        self.text_table.pack(padx=4, pady=6)

        # === 狀態變數初始化 ===
        self.cap = None                          # 當前來源物件（cv2.VideoCapture 或 SeleniumCapture）
        self.cap_type = None                     # 來源型別 "video"、"camera"、"selenium"
        self.video_source = None                 # 檔案路徑或攝影機編號（備查）
        self.detection_running = False           # 是否正在進行辨識
        self.lines = [None for _ in range(self.max_lines)]      # 存每條線的座標（Canvas上 x,y）
        self.line_ids = [None for _ in range(self.max_lines)]   # Canvas 畫線物件id
        self.label_ids = [None for _ in range(self.max_lines)]  # Canvas 編號文字物件id
        self.editing_line_idx = None             # 正在設定哪一條線（索引）
        self.editing_point = 0                   # 目前編輯點數（0=等起點，1=等終點）
        self.imgtk = None                        # 影像暫存（Tkinter 圖片，防止被GC釋放）
        self.origin_w = None                     # 原始影像寬度（for座標轉換用）
        self.origin_h = None                     # 原始影像高度
        self.x_scale = 1                         # 原始寬度/顯示寬度（canvas→原圖轉換倍率）
        self.y_scale = 1                         # 原始高度/顯示高度


    # === 來源選擇 ===
    def choose_video(self):
        self.release_cap()    # 釋放前一個來源（防止同時打開多個 VideoCapture）
        # 彈出檔案選擇視窗，只允許選擇 mp4/avi 等影片檔
        path = filedialog.askopenfilename(title="選擇影片檔案", filetypes=[("Video files", "*.mp4;*.avi")])
        if path:
            self.cap = cv2.VideoCapture(path)  # 用 OpenCV 打開選中的影片檔
            self.cap_type = "video"            # 標記來源型態
            self.after_choose_source("影片")    # 共用的後續 UI/狀態更新
    
    def choose_camera(self):
        self.release_cap()    # 釋放舊來源
        self.cap = cv2.VideoCapture(0)         # 打開預設攝影機（編號0）
        self.cap_type = "camera"               # 標記來源型態
        self.after_choose_source("攝影機")
    
    def choose_url(self):
        url = self.url_var.get().strip()       # 取得輸入欄網址
        if not url:
            messagebox.showerror("錯誤", "請輸入網址！")    # 檢查是否有填
            return
        self.release_cap()
        try:
            self.cap = SeleniumCapture(url)    # 用 SeleniumCapture 以 headless Chrome 取得即時畫面
            self.cap_type = "selenium"
            self.after_choose_source("爬蟲畫面")
        except Exception as e:
            messagebox.showerror("錯誤", f"Selenium 初始化失敗：{e}")
    
    def after_choose_source(self, src_name):
        # 切換來源之後，先顯示第一幀
        self.show_first_frame()
        # 開放所有設定線按鈕（可畫線）
        for btn in self.line_set_buttons:
            btn.config(state=tk.NORMAL)
        self.btn_redraw.config(state=tk.NORMAL)         # 全部清除按鈕開啟
        self.btn_start.config(state=tk.DISABLED)        # 辨識尚未啟動
        # 彈出提示訊息
        messagebox.showinfo("請標註線", f"目前來源：{src_name}\n請按右側各條線的「設定」按鈕，依序指定起點與終點（至少2條線）")
    
    def release_cap(self):
        # 釋放/關閉現有來源資源，避免資源占用與多重開啟
        if self.cap is not None:
            try:
                if self.cap_type == "selenium":
                    self.cap.release()         # SeleniumCapture: 關閉 driver
                elif self.cap_type in ("video", "camera"):
                    self.cap.release()         # OpenCV VideoCapture: 釋放資源
            except Exception:
                pass
            self.cap = None
            self.cap_type = None

    # === 影像顯示與畫線 ===
    def show_first_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.origin_w = frame.shape[1]
            self.origin_h = frame.shape[0]
            self.x_scale = self.origin_w / self.display_w
            self.y_scale = self.origin_h / self.display_h
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2.resize(frame_rgb, (self.display_w, self.display_h)))
            self.imgtk = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self.imgtk)
            self.redraw_all_lines()
        else:
            messagebox.showerror("錯誤", "無法讀取影片/攝影機/爬蟲來源")

    def edit_line(self, idx):
        self.editing_line_idx = idx
        self.editing_point = 0
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.config(cursor="crosshair")
        messagebox.showinfo("設定線", f"請點選第{idx+1}條線的起點，再點終點")

    def on_canvas_click(self, event):
        idx = self.editing_line_idx
        if idx is None:
            return
        if self.editing_point == 0:
            self.lines[idx] = [(event.x, event.y), None]
            self.editing_point = 1
        elif self.editing_point == 1:
            if self.lines[idx] is None:
                return
            self.lines[idx][1] = (event.x, event.y)
            self.redraw_line(idx)
            self.editing_line_idx = None
            self.editing_point = 0
            self.canvas.unbind("<Button-1>")
            self.canvas.config(cursor="")
            # 判斷已完成線數
            line_count = sum(1 for l in self.lines if l is not None and l[0] and l[1])
            if line_count >= self.min_lines:
                self.btn_start.config(state=tk.NORMAL)
            else:
                self.btn_start.config(state=tk.DISABLED)

    def redraw_line(self, idx):
        if self.line_ids[idx]:
            self.canvas.delete(self.line_ids[idx])
            self.line_ids[idx] = None
        if self.label_ids[idx]:
            self.canvas.delete(self.label_ids[idx])
            self.label_ids[idx] = None
        data = self.lines[idx]
        if data is None or data[0] is None or data[1] is None:
            return
        x1, y1 = data[0]
        x2, y2 = data[1]
        line_id = self.canvas.create_line(x1, y1, x2, y2, fill="red", width=3)
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        label_id = self.canvas.create_text(mid_x, mid_y, text=str(idx+1), fill="white", font=("Arial", 14, "bold"))
        self.line_ids[idx] = line_id
        self.label_ids[idx] = label_id

    def redraw_all_lines(self):
        for idx in range(self.max_lines):
            self.redraw_line(idx)

    def clear_all_lines(self):
        for i in range(self.max_lines):
            if self.line_ids[i]:
                self.canvas.delete(self.line_ids[i])
                self.line_ids[i] = None
            if self.label_ids[i]:
                self.canvas.delete(self.label_ids[i])
                self.label_ids[i] = None
            self.lines[i] = None
        self.btn_start.config(state=tk.DISABLED)

    # === YOLO辨識 & 計數 ===
    def start_detection(self):
        if hasattr(self, 'detection_running') and self.detection_running:
            return
        self.model = YOLO('modelv14.pt')
        self.detection_running = True
        self.counts = [0 for _ in range(self.max_lines)]
        self.track_dict = {}
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_start.config(state=tk.DISABLED)
        self.btn_export_csv.config(state=tk.DISABLED)
        self.text_table.delete(1.0, tk.END)
        self.update_detection()

    def update_detection(self):
        if not self.detection_running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.detection_running = False
            self.btn_pause.config(state=tk.DISABLED)
            self.btn_export_csv.config(state=tk.NORMAL)
            messagebox.showinfo("完成", "影片/來源已結束，辨識停止！")
            return

        # YOLOv8追蹤辨識
        results = self.model.track(frame, persist=True, verbose=False, conf=0.3)[0]
        boxes = np.array(results.boxes.data.tolist(), dtype="float")
        for box in boxes:
            if len(box) >= 7:
                x1, y1, x2, y2, track_id, conf, class_id = box[:7]
            elif len(box) == 6:
                x1, y1, x2, y2, conf, class_id = box
                track_id = -1
            else:
                continue
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            # 判斷 crossing (要用原始frame座標)
            for idx, line in enumerate(self.lines):
                if line and line[0] and line[1]:
                    # canvas->原始座標
                    lx1, ly1 = int(line[0][0]*self.x_scale), int(line[0][1]*self.y_scale)
                    lx2, ly2 = int(line[1][0]*self.x_scale), int(line[1][1]*self.y_scale)
                    self._check_cross(track_id, (center_x, center_y), idx, lx1, ly1, lx2, ly2)
            # 畫框與中心點
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.circle(frame, (center_x, center_y), 4, (0,0,255), -1)
        # 畫每條線（用原始frame座標畫）
        for idx, line in enumerate(self.lines):
            if line and line[0] and line[1]:
                lx1, ly1 = int(line[0][0]*self.x_scale), int(line[0][1]*self.y_scale)
                lx2, ly2 = int(line[1][0]*self.x_scale), int(line[1][1]*self.y_scale)
                cv2.line(frame, (lx1,ly1), (lx2,ly2), (0,0,255), 3)
                mx = int((lx1 + lx2) / 2)
                my = int((ly1 + ly2) / 2)
                cv2.putText(frame, str(idx+1), (mx-10, my-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        # 更新Tkinter影像（縮回顯示尺寸）
        show_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        show_img = cv2.resize(show_img, (self.display_w, self.display_h))
        imgtk = ImageTk.PhotoImage(Image.fromarray(show_img))
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0,0,anchor="nw", image=imgtk)
        self.update_table()
        if self.detection_running:
            self.root.after(1, self.update_detection)

    def _check_cross(self, track_id, center, line_idx, x1, y1, x2, y2):
        if track_id == -1:
            return False
        key = (int(track_id), line_idx)
        px, py = center
        dist = self._point_to_line_dist(px, py, x1, y1, x2, y2)
        if dist < 10 and key not in self.track_dict:
            self.counts[line_idx] += 1
            self.track_dict[key] = True
            return True
        return False

    def _point_to_line_dist(self, px, py, x1, y1, x2, y2):
        a = np.array([px, py])
        b = np.array([x1, y1])
        c = np.array([x2, y2])
        ba = a - b
        bc = c - b
        if np.dot(ba, bc) < 0:
            return np.linalg.norm(ba)
        ca = a - c
        cb = b - c
        if np.dot(ca, cb) < 0:
            return np.linalg.norm(ca)
        return np.abs(np.cross(bc, ba)) / np.linalg.norm(bc)

    def update_table(self):
        self.text_table.delete(1.0, tk.END)
        for idx in range(self.max_lines):
            self.text_table.insert(tk.END, f"線{idx+1} 車流量：{self.counts[idx]}\n")

    def pause_detection(self):
        self.detection_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_export_csv.config(state=tk.NORMAL)
        messagebox.showinfo("暫停", "辨識已暫停，可繼續或匯出報表")

    def export_csv(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV 檔案", "*.csv")])
        if not file_path:
            return
        with open(file_path, "w", newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["線編號", "車流量"])
            for idx in range(self.max_lines):
                writer.writerow([idx+1, self.counts[idx]])
        messagebox.showinfo("匯出完成", "已儲存流量報表！")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficDetectionGUI(root)
    root.mainloop()