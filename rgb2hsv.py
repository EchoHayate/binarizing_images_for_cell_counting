import cv2
import numpy as np

# 创建一个数组，包含所有RGB值
rgb_colors = np.array([
    [60, 50, 100],
    [90, 80, 130],
    [130, 100, 140],
    [170, 150, 170]
], dtype=np.uint8)

# 将RGB转换为HSV
hsv_colors = cv2.cvtColor(rgb_colors.reshape((4, 1, 3)), cv2.COLOR_RGB2HSV)

# 打印转换后的HSV值
print(hsv_colors.reshape((4, 3)))