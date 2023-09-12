from matplotlib import pyplot as plt
import cv2
import numpy as np

# 读取图像
img = cv2.imread('axon01.tif', cv2.IMREAD_GRAYSCALE)
# 高斯滤波器滤波后的图像List，用于plot绘图
imgFilteredList = []
# 对图像应用高斯滤波器并显示结果
for sigma in [1, 2, 5, 7]:
    kernelSize = int(np.ceil(sigma * 3) * 2 + 1)
    gaussianFilter = cv2.getGaussianKernel(kernelSize, sigma)  # 生成高斯滤波器
    gaussianFilter /= np.sum(gaussianFilter)  # 归一化当前高斯滤波器
    imgFiltered = cv2.filter2D(img, -1, gaussianFilter)
    imgFilteredList.append(imgFiltered)

# 创建一个包含四个子图的Matplotlib窗口
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 9))
# 在每个子图中绘制对应的图像
axs[0, 0].imshow(imgFilteredList[0])
axs[0, 0].set_title("Gaussian Filter with sigma=1")
axs[0, 1].imshow(imgFilteredList[1])
axs[0, 1].set_title("Gaussian Filter with sigma=2")
axs[1, 0].imshow(imgFilteredList[2])
axs[1, 0].set_title("Gaussian Filter with sigma=5")
axs[1, 1].imshow(imgFilteredList[3])
axs[1, 1].set_title("Gaussian Filter with sigma=7")

# 在窗口中显示图像
plt.show()
