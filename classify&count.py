# # import cv2
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from skimage import morphology, measure

# # def load_image(path):
# #     """加载图像并转换为BGR到HSV."""
# #     image = cv2.imread(path)
# #     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# #     return hsv_image

# # def segment_cells_by_color(hsv_image, color_range):
# #     """根据颜色范围进行图像分割."""
# #     mask = cv2.inRange(hsv_image, color_range[0], color_range[1])
# #     cleaned_mask = morphology.remove_small_objects(mask > 0, min_size=100)
# #     return cleaned_mask

# # def count_cells(mask):
# #     """计数分割后的细胞."""
# #     labels = measure.label(mask)
# #     return labels.max()  # 返回最大的标签号，即细胞数

# # def create_color_range_image(color_range):
# #     # 创建一个线性渐变的颜色条
# #     color_low = color_range[0]
# #     color_high = color_range[1]
# #     gradient = np.linspace(color_low, color_high, 256).astype(np.uint8)
# #     gradient = np.tile(gradient, (100, 1, 1))  # 扩展到更大的尺寸以便于观察
    
# #     return gradient

# # transparent_purple_range = (np.array([100, 128, 126]), np.array([130, 98, 126]))
# # light_and_blue_purple_range = (np.array([140, 73, 143]), np.array([170, 30, 150]))
# # # transparent_purple_range = (np.array([126, 128, 100]), np.array([126 , 98 ,130]))
# # # light_and_blue_purple_range = (np.array([143 , 73 ,140]), np.array([150 , 30,170]))
# # # # 生成颜色条
# # # transparent_purple_img = create_color_range_image(transparent_purple_range)
# # # light_and_blue_purple_img = create_color_range_image(light_and_blue_purple_range)

# # # # 显示颜色条
# # # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# # # ax[0].imshow(transparent_purple_img)
# # # ax[0].set_title('Transparent Purple')
# # # ax[0].axis('off')  # 不显示坐标轴

# # # ax[1].imshow(light_and_blue_purple_img)
# # # ax[1].set_title('Light and Blue Purple')
# # # ax[1].axis('off')  # 不显示坐标轴

# # # plt.show()



# # # 使用示例
# # image_path = '/home/xianyun/wst/Automatic-Identification-and-Counting-of-Blood-Cells/clustered_image.jpg'
# # hsv_image = load_image(image_path)
# # transparent_purple_cells = segment_cells_by_color(hsv_image, transparent_purple_range)
# # light_and_blue_purple_cells = segment_cells_by_color(hsv_image, light_and_blue_purple_range)

# # print("Transparent Purple Cell Count:", count_cells(transparent_purple_cells))
# # print("Light and Blue Purple Cell Count:", count_cells(light_and_blue_purple_cells))

# # # 显示二值化后的图像
# # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# # ax[0].imshow(transparent_purple_cells, cmap='gray')
# # ax[0].set_title('Binary Image of Transparent Purple Cells')
# # ax[0].axis('off')

# # ax[1].imshow(light_and_blue_purple_cells, cmap='gray')
# # ax[1].set_title('Binary Image of Light and Blue Purple Cells')
# # ax[1].axis('off')

# # plt.show()


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
image = cv2.imread('/home/xianyun/wst/Automatic-Identification-and-Counting-of-Blood-Cells/clustered_image.jpg')
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


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import measure, filters

# def segment_cells(image, is_dark=True):
#     """使用Otsu's方法和额外的图像处理来分割细胞"""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # 使用Otsu's方法找到全局阈值
#     global_thresh = filters.threshold_otsu(gray)
    
#     if is_dark:
#         # 对于深色细胞，我们取小于阈值的部分
#         binary = gray < global_thresh
#     else:
#         # 对于浅色细胞，我们取大于阈值但小于一定值的部分
#         binary = (gray > global_thresh) & (gray < 200)  # 200是一个经验值，可能需要调整
    
#     # 应用形态学操作来改善分割结果
#     kernel = np.ones((2,2), np.uint8)
#     binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=2)
#     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
#     return binary

# def count_cells(mask):
#     """计数分割后的细胞"""
#     labels = measure.label(mask)
#     return labels.max()

# # 读取图像
# image = cv2.imread('/home/xianyun/wst/Automatic-Identification-and-Counting-of-Blood-Cells/clustered_image.jpg')

# # 分割深色和浅色细胞
# dark_mask = segment_cells(image, is_dark=True)
# light_mask = segment_cells(image, is_dark=False)

# # 确保两个mask不重叠
# overlap = dark_mask & light_mask
# dark_mask = dark_mask & ~overlap
# light_mask = light_mask & ~overlap

# # 计数细胞
# dark_count = count_cells(dark_mask)
# light_count = count_cells(light_mask)

# # 可视化结果
# fig, ax = plt.subplots(2, 2, figsize=(12, 12))

# ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# ax[0, 0].set_title('Original Image')
# ax[0, 0].axis('off')

# ax[0, 1].imshow(dark_mask, cmap='gray')
# ax[0, 1].set_title(f'Dark Cells (Count: {dark_count})')
# ax[0, 1].axis('off')

# ax[1, 0].imshow(light_mask, cmap='gray')
# ax[1, 0].set_title(f'Light Cells (Count: {light_count})')
# ax[1, 0].axis('off')

# # 创建一个复合图像，显示深色和浅色细胞
# composite = np.zeros_like(image)
# composite[dark_mask > 0] = [0, 0, 255]  # 深色细胞为红色
# composite[light_mask > 0] = [0, 255, 0]  # 浅色细胞为绿色
# ax[1, 1].imshow(composite)
# ax[1, 1].set_title('Composite (Red: Dark, Green: Light)')
# ax[1, 1].axis('off')

# plt.tight_layout()
# plt.show()

# print(f"Dark cell count: {dark_count}")
# print(f"Light cell count: {light_count}")

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import measure, morphology

# # def segment_cells(image, lower_threshold, upper_threshold):
# #     """根据颜色阈值分割细胞"""
# #     mask = cv2.inRange(image, lower_threshold, upper_threshold)
# #     # 应用形态学操作来改善分割结果
# #     kernel = np.ones((3,3), np.uint8)
# #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
# #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
# #     return mask
# def segment_cells(image, lower_threshold, upper_threshold):
#     """根据颜色阈值分割细胞"""
#     # mask = cv2.inRange(image, lower_threshold, upper_threshold)
#     # return mask

#     mask = cv2.inRange(image,lower_threshold, upper_threshold)
#     cleaned_mask = morphology.remove_small_objects(mask > 0, min_size=10)
#     return cleaned_mask.astype(np.uint8) * 255

# def count_cells(mask):
#     """计数分割后的细胞"""
#     labels = measure.label(mask)
#     return labels.max()

# # 读取图像
# image = cv2.imread('/home/xianyun/wst/Automatic-Identification-and-Counting-of-Blood-Cells/clustered_image.jpg')
# image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # 调整后的HSV阈值
# dark_lower = np.array([120, 50, 50])
# dark_upper = np.array([150, 255, 150])
# light_lower = np.array([120, 50,180])
# light_upper = np.array([150, 255, 255])

# # 分割深色和浅色细胞
# dark_mask = segment_cells(image_hsv, dark_lower, dark_upper)
# light_mask = segment_cells(image_hsv, light_lower, light_upper)

# # 确保浅色mask不包含深色部分
# light_mask = cv2.bitwise_and(light_mask, cv2.bitwise_not(dark_mask))

# # 计数细胞
# dark_count = count_cells(dark_mask)
# light_count = count_cells(light_mask)

# # 可视化结果
# fig, ax = plt.subplots(2, 2, figsize=(12, 12))

# ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# ax[0, 0].set_title('Original Image')
# ax[0, 0].axis('off')

# ax[0, 1].imshow(dark_mask, cmap='gray')
# ax[0, 1].set_title(f'Dark Cells (Count: {dark_count})')
# ax[0, 1].axis('off')

# ax[1, 0].imshow(light_mask, cmap='gray')
# ax[1, 0].set_title(f'Light Cells (Count: {light_count})')
# ax[1, 0].axis('off')

# # 创建一个复合图像，显示深色和浅色细胞
# composite = np.zeros_like(image)
# composite[dark_mask > 0] = [0, 0, 255]  # 深色细胞为红色
# composite[light_mask > 0] = [0, 255, 0]  # 浅色细胞为绿色
# ax[1, 1].imshow(composite)
# ax[1, 1].set_title('Composite (Red: Dark, Green: Light)')
# ax[1, 1].axis('off')

# plt.tight_layout()
# plt.show()

# print(f"Dark cell count: {dark_count}")
# print(f"Light cell count: {light_count}")