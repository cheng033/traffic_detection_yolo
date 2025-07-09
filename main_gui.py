import tkinter as tk                           # 匯入tkinter做GUI
from tkinter import filedialog, messagebox     # 檔案對話框、彈出訊息框
import subprocess                             # 執行外部程式用
import sys                                    # 取得python執行檔路徑
import os                                     # 處理檔案路徑

# 動作：選擇影片
def choose_video():
    path = filedialog.askopenfilename(        # 彈出檔案選擇視窗
        title="選擇影片",
        filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")]
    )
    if path:                                  # 如果有選到檔案
        video_path.set(path)                  # 設定變數
        status_label.config(text=f"已選擇影片：{os.path.basename(path)}") # 更新顯示檔名
    else:                                     # 如果沒選
        video_path.set("")                    # 清空影片路徑
        status_label.config(text="尚未選擇影片") # 狀態顯示

# 動作：啟動 main.py 執行偵測
def start_detection():
    mode = input_mode.get()                   # 取得來源選項（video/camera）
    video = video_path.get()                  # 取得影片路徑
    # 取得main.py路徑（確保即使工作目錄不同也能找到）
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")

    if mode == "video" and not video:         # 若是影片但沒選，提示
        status_label.config(text="請先選擇影片！")
        return

    # 檢查main.py是否存在
    if not os.path.exists(script_path):
        messagebox.showerror("找不到main.py", f"main.py不存在於：\n{script_path}")
        return

    cmd = [sys.executable, script_path]       # 組合命令：python main.py
    if mode == "video":
        cmd += ["--video", video]             # 加上影片路徑參數
    else:
        cmd += ["--camera"]                   # 或是選攝影機參數

    status_label.config(text="已啟動偵測... 請稍候") # 更新狀態
    try:
        subprocess.Popen(cmd)                 # 啟動main.py新程序
    except Exception as e:
        messagebox.showerror("執行錯誤", str(e)) # 啟動失敗彈窗顯示錯誤
        return
    root.destroy()                            # 關閉GUI

# --- 介面設計 ---
root = tk.Tk()                               # 建立主視窗
root.title("YOLOv8 計數系統 GUI")            # 設定標題
root.geometry("400x250")                     # 固定大小
root.resizable(False, False)                 # 不允許縮放

input_mode = tk.StringVar(value="video")     # 用來儲存選擇的輸入來源
video_path = tk.StringVar()                  # 儲存影片路徑

tk.Label(root, text="請選擇輸入來源：", font=("Arial", 14)).pack(pady=10) # 標題

tk.Radiobutton(root, text="影片檔案", variable=input_mode, value="video").pack()  # 選影片
tk.Button(root, text="選擇影片", command=choose_video).pack(pady=5)              # 影片選擇按鈕
tk.Radiobutton(root, text="攝影機（即時）", variable=input_mode, value="camera").pack() # 選攝影機

tk.Button(root, text="開始偵測", command=start_detection,                      # 啟動主程式按鈕
          bg="green", fg="white", font=("Arial", 12)).pack(pady=15)

status_label = tk.Label(root, text="尚未選擇影片或來源", fg="blue")             # 狀態顯示
status_label.pack()

root.mainloop()                               # 進入GUI事件迴圈，等待用戶操作
