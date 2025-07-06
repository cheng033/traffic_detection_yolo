import tkinter as tk
from tkinter import filedialog
import subprocess
import sys

# 建立主介面
root = tk.Tk()
root.title("YOLOv8 計數系統 GUI")
root.geometry("400x250")

# 選擇來源變數
input_mode = tk.StringVar(value="video")
video_path = tk.StringVar()

# 動作：選擇影片

def choose_video():
    path = filedialog.askopenfilename(
        title="選擇影片",
        filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")]
    )
    video_path.set(path)
    status_label.config(text=f"已選擇影片：{path.split('/')[-1]}")

# 動作：啟動主程式

def start_detection():
    mode = input_mode.get()
    if mode == "video" and not video_path.get():
        status_label.config(text="請先選擇影片！")
        return

    cmd = [sys.executable, "main.py"]
    if mode == "video":
        cmd += ["--video", video_path.get()]
    else:
        cmd += ["--camera"]

    status_label.config(text="已啟動偵測... 請稍候")
    subprocess.Popen(cmd)
    root.destroy()  # 關閉 GUI

# GUI 元件設計
tk.Label(root, text="請選擇輸入來源：", font=("Arial", 14)).pack(pady=10)

# Radio Buttons
tk.Radiobutton(root, text="影片檔案", variable=input_mode, value="video").pack()
tk.Button(root, text="選擇影片", command=choose_video).pack(pady=5)
tk.Radiobutton(root, text="攝影機（即時）", variable=input_mode, value="camera").pack()

# 啟動按鈕
tk.Button(root, text="開始偵測", command=start_detection, bg="green", fg="white", font=("Arial", 12)).pack(pady=15)

# 狀態標籤
status_label = tk.Label(root, text="尚未選擇影片或來源", fg="blue")
status_label.pack()

root.mainloop()
