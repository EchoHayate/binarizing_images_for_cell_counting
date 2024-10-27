
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure

def segment_cells(image, lower_threshold, upper_threshold):
    """根据颜色阈值分割细胞"""
    mask = cv2.inRange(image, lower_threshold, upper_threshold)
    return mask

    # mask = cv2.inRange(image,lower_threshold, upper_threshold)
    # cleaned_mask = morphology.remove_small_objects(mask > 0, min_size=10)
    # return cleaned_mask.astype(np.uint8) * 255

def count_cells(mask):
    """计数分割后的细胞"""
    labels = measure.label(mask)
    return labels.max()

# 读取图像
# image = cv2.imread('/home/xianyun/wst/binarizing_images_for_cell_counting/clustered_image.jpg')
image = cv2.imread('change/to/your/own/path')
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义深色和浅色细胞的HSV阈值
dark_lower = np.array([120, 50, 50])
dark_upper = np.array([150, 255, 150])
light_lower = np.array([125, 0,99])
light_upper = np.array([139, 59, 172])
# light_lower = np.array([0, 0, 0])
# light_upper = np.array([179,255, 255])
# 分割深色和浅色细胞
dark_mask = segment_cells(image_hsv, dark_lower, dark_upper)
light_mask = segment_cells(image_hsv, light_lower, light_upper)
def remove_thin_lines(mask, kernel_size=(3, 3)):
    # 创建形态学操作的核
    kernel = np.ones(kernel_size, np.uint8)

    # 开运算：先腐蚀后膨胀
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return opening

# 假设 light_mask 是你通过 segment_cells 函数得到的掩模
cleaned_mask = remove_thin_lines(light_mask)
# 计数细胞
dark_count = count_cells(dark_mask)
light_count = count_cells(cleaned_mask)

# 可视化结果
fig, ax = plt.subplots(2, 2, figsize=(12, 12))

ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0, 0].set_title('Original Image')
ax[0, 0].axis('off')

ax[0, 1].imshow(dark_mask, cmap='gray')
ax[0, 1].set_title(f'Dark Cells (Count: {dark_count})')
ax[0, 1].axis('off')

ax[1, 0].imshow(cleaned_mask, cmap='gray')
ax[1, 0].set_title(f'Light Cells (Count: {light_count})')
ax[1, 0].axis('off')

# 创建一个复合图像，显示深色和浅色细胞
composite = np.zeros_like(image)
composite[dark_mask > 0] = [0, 0, 255]  # 深色细胞为红色
composite[light_mask > 0] = [0, 255, 0]  # 浅色细胞为绿色
ax[1, 1].imshow(composite)
ax[1, 1].set_title('Composite (Red: Dark, Green: Light)')
ax[1, 1].axis('off')

plt.tight_layout()
plt.show()

print(f"Dark cell count: {dark_count}")
print(f"Light cell count: {light_count}")

