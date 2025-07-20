# traffic_detection_yolo

model.pt可在hugging face下載<br>

https://huggingface.co/uchen3/yolov8s-intersection-model/tree/main

## 安裝環境
python 3.11.13

```bash
pip install -r requirements.txt
```
## 執行
影片開始前選擇輸入源(影片檔, 攝影機, 線上網址)
```bash
python main_gui.py
```
若要爬蟲即時影像先執行server
```bash
python server.py
```

