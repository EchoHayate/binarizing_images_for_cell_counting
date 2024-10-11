# import cv2
# import numpy as np
# image_path = '/home/xianyun/wst/Automatic-Identification-and-Counting-of-Blood-Cells/data/project_image.jpg'
# # 加载图像（这里假设是灰度图像）
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # 应用阈值来获取二值图像
# # ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
# thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                cv2.THRESH_BINARY_INV, 11, 2)

# # 创建形态学运算的核
# kernel = np.ones((5,5), np.uint8)

# # 使用开运算进行噪点清除和分离物体
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# # 查找轮廓（cv2.findContours函数在不同的OpenCV版本中返回值不同）
# contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 计算检测到的轮廓数量，即物体数量
# object_count = len(contours)

# print(f"Detected objects: {object_count}")

# # 可视化结果
# cv2.drawContours(image, contours, -1, (0,255,0), 3)
# cv2.imshow('Objects Counted', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from skimage import filters, measure, morphology
# import matplotlib.pyplot as plt


# def load_image(path):
#     """加载图像并转换为灰度图."""
#     image = cv2.imread(path)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return gray_image

# def segment_cells(gray_image, cell_type='white'):
#     """根据细胞类型进行图像分割."""
#     if cell_type == 'white':
#         # 通常白细胞比红细胞更亮
#         thresh = filters.threshold_otsu(gray_image)
#         binary_image = gray_image > thresh
#     else:
#         # 红细胞的处理可能需要不同的阈值或预处理步骤
#         thresh = filters.threshold_otsu(gray_image)
#         binary_image = gray_image < thresh
    
#     # 移除小对象，通常这些不是目标细胞
#     cleaned_image = morphology.remove_small_objects(binary_image, min_size=100)
#     return cleaned_image

# def count_cells(binary_image):
#     """计数分割后的细胞."""
#     labels = measure.label(binary_image)
#     return labels.max()  # 返回最大的标签号，即细胞数

# # 使用示例
# image_path = '/home/xianyun/wst/Automatic-Identification-and-Counting-of-Blood-Cells/data/project_image.jpg'
# gray_image = load_image(image_path)
# white_cells = segment_cells(gray_image, 'white')
# red_cells = segment_cells(gray_image, 'red')


# cfig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(white_cells, cmap='gray')
# ax[0].set_title('Binary Image of White Cells')
# ax[0].axis('off')

# ax[1].imshow(red_cells, cmap='gray')
# ax[1].set_title('Binary Image of Red Cells')
# ax[1].axis('off')

# plt.show()

# print("White Blood Cell Count:", count_cells(white_cells))
# print("Red Blood Cell Count:", count_cells(red_cells))


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image
image_path = '/home/xianyun/wst/Automatic-Identification-and-Counting-of-Blood-Cells/data/project_image.jpg'
image = Image.open(image_path)

# Convert image to RGB and numpy array
image = image.convert('RGB')
image_array = np.array(image)

# Reshape the image array for KMeans clustering (each pixel is a data point with 3 color values)
pixels = image_array.reshape((-1, 3))

# Perform KMeans clustering to classify colors
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(pixels)
clustered_pixels = kmeans.cluster_centers_[kmeans.labels_]

# Reshape back to the original image dimensions
clustered_image = clustered_pixels.reshape(image_array.shape).astype(np.uint8)

# Display the original and clustered images
plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image)

# plt.subplot(1, 2, 2)
plt.title("Clustered Image (3 clusters)")
plt.imshow(clustered_image)
plt.savefig('clustered_image.jpg', bbox_inches='tight', pad_inches=0)
plt.show()
