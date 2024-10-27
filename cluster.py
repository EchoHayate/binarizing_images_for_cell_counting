


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image
# image_path = '/home/xianyun/wst/binarizing_images_for_cell_counting/project_image.jpg'
image = cv2.imread('change/to/your/own/path')
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
