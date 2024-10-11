import cv2
import numpy as np

def nothing(x):
    pass

# 读取图像
image = cv2.imread('/home/xianyun/wst/Automatic-Identification-and-Counting-of-Blood-Cells/clustered_image.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 创建一个窗口
cv2.namedWindow('HSV Range Finder')

# 创建用于调整HSV范围的滑块
cv2.createTrackbar('H Min', 'HSV Range Finder', 0, 179, nothing)
cv2.createTrackbar('H Max', 'HSV Range Finder', 179, 179, nothing)
cv2.createTrackbar('S Min', 'HSV Range Finder', 0, 255, nothing)
cv2.createTrackbar('S Max', 'HSV Range Finder', 255, 255, nothing)
cv2.createTrackbar('V Min', 'HSV Range Finder', 0, 255, nothing)
cv2.createTrackbar('V Max', 'HSV Range Finder', 255, 255, nothing)

while(1):
    # 获取当前滑块的位置
    h_min = cv2.getTrackbarPos('H Min', 'HSV Range Finder')
    h_max = cv2.getTrackbarPos('H Max', 'HSV Range Finder')
    s_min = cv2.getTrackbarPos('S Min', 'HSV Range Finder')
    s_max = cv2.getTrackbarPos('S Max', 'HSV Range Finder')
    v_min = cv2.getTrackbarPos('V Min', 'HSV Range Finder')
    v_max = cv2.getTrackbarPos('V Max', 'HSV Range Finder')
    
    # 定义HSV范围
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    
    # 根据范围创建掩码
    mask = cv2.inRange(hsv, lower, upper)
    
    # 应用掩码
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # 显示原图和结果
    cv2.imshow('Original', image)
    cv2.imshow('Result', result)
    
    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"HSV range: [{h_min}, {s_min}, {v_min}] - [{h_max}, {s_max}, {v_max}]")
cv2.destroyAllWindows()