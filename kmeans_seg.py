# 图像聚类
import cv2
import numpy as np
from sklearn.cluster import KMeans


def image_clustering(image_path, num_clusters):
    # 读取图像
    image = cv2.imread(image_path)
    (height, width, p) = image.shape
    image = cv2.resize(image, (int(width/2), int(height/2)))
    # 将图像从BGR颜色空间转换为RGB颜色空间
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 将图像转换为一维数组，以便使用K-Means算法
    pixels = image.reshape(-1, 3)
    # 使用K-Means算法对像素进行聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(pixels)
    # 获取聚类中心
    centers = kmeans.cluster_centers_
    # 根据聚类结果，将每个像素替换为对应的聚类中心
    clustered_image = centers[labels].reshape(image.shape)
    # 将图像从RGB颜色空间转换回BGR颜色空间
    clustered_image = cv2.cvtColor(clustered_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return clustered_image

if __name__ == "__main__":
    # 图像路径和聚类数
    image_path = "E:/DATA/JY/ML_DATA/K/XK/mask/-1_20230827085554_d45f1a18fc011596.png"  # 替换为实际的图像路径

    num_clusters = 3  # 设置聚类数
    clustered_image = image_clustering(image_path, num_clusters)
    # 显示原始图像和聚类后的图像
    cv2.imshow("Original Image", cv2.imread(image_path))
    cv2.imshow(f"Clustered Image (K={num_clusters})", clustered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
