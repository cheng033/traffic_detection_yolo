import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import sys
import os

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
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
   

    # 必要檢查
    if not os.path.exists(script_path):
        messagebox.showerror("找不到main.py", f"main.py不存在於：\n{script_path}")
        return

    cmd = [sys.executable, script_path]
    if mode == "video":
        if not video:
            status_label.config(text="請先選擇影片！")
            return
        cmd += ["--video", video]
    elif mode == "camera":
        cmd += ["--camera"]
    elif mode == "url":
        if not url:
            status_label.config(text="請輸入即時影像網址！")
            return
        cmd += ["--url", url]

    status_label.config(text="已啟動偵測... 請稍候")
    try:
        subprocess.Popen(cmd)
    except Exception as e:
        messagebox.showerror("執行錯誤", str(e))
        return
    root.destroy()

# --- 介面設計 ---
root = tk.Tk()
root.title("YOLOv8 計數系統 GUI")
root.geometry("430x320")
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
tk.Label(radio_frame, text="範例：https://tw.live/cam/?id=NWT0052", fg="gray", font=("Arial", 8)).grid(row=3, column=1, sticky='w')

tk.Button(root, text="開始偵測", command=start_detection, bg="green", fg="white", font=("Arial", 12)).pack(pady=18)

status_label = tk.Label(root, text="尚未選擇影片或來源", fg="blue")
status_label.pack()

root.mainloop()