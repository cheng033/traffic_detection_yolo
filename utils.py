import csv
import pandas as pd
import matplotlib.pyplot as plt

def write_csv(records, csv_filename):
    with open(csv_filename, "w", newline="", encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["TrackID", "Class", "Direction", "Timestamp"])
        writer.writerows(records)

def plot_direction_donut(records, sanitized_title):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(records, columns=["TrackID", "Class", "Direction", "Timestamp"])
    if df.empty:
        print("沒有任何紀錄，略過甜甜圈圖繪製")
        return

    # --- ① 統計 L / S / R ---
    dir_counts = df['Direction'].value_counts().reindex(['L', 'S', 'R']).fillna(0).astype(int)
    if dir_counts.sum() == 0:
        print("Direction 欄位沒有 L/S/R，略過甜甜圈圖繪製")
        return

    # --- ② 配色與標籤 ---
    color_map = {'L': '#ff6384', 'S': '#36a2eb', 'R': '#ffcd56'}
    values = dir_counts.values
    labels = [f"{d}\n{cnt}" for d, cnt in dir_counts.items()]
    colors = [color_map[d] for d in dir_counts.index]

    # --- ③ 畫 donut ---
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.4)
    )
    ax.set_title("Traffic Direction Distribution (L / S / R)")

    # --- ④ 附加統計資訊 ---
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    span_sec = (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds()
    span_min = span_sec / 60 if span_sec > 0 else 1
    avg = len(df) / span_min
    plt.figtext(
        0.5, 0.01,
        f"Duration: {span_min:.2f} min, Avg traffic: {avg:.2f} veh/min",
        ha='center', fontsize=12
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{sanitized_title}_direction_donut.png", dpi=200)
    plt.close()


