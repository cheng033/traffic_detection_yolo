import cv2
import numpy as np
import sys

# 全域變數
colors = [(0,0,255), (0,128,255), (0,255,255), (255,0,255)]
line_colors = ['Red', 'Orange', 'Yellow', 'Purple']
lines = []
temp_line = []
drawing = False
line_pass_count = []
rect_zone = None

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
    global line_pass_count, rect_zone
    print("\n請順時針方向畫線,例如：上→右→下→左，對應編號為 0,1,2,3")
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

    # 建立矩形區域
    all_points = [pt for line in lines for pt in line]
    xs, ys = zip(*all_points)
    rx, ry = min(xs), min(ys)
    rw, rh = max(xs) - rx, max(ys) - ry
    rect_zone = (rx, ry, rw, rh)
    return lines, rect_zone
