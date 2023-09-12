import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
axon02 = cv2.imread("axon02.tif", cv2.IMREAD_GRAYSCALE)
cell_nucleus = cv2.imread("cell_nucleus.tif", cv2.IMREAD_GRAYSCALE)


# 定义卷积计算函数
def convolve(image, kernel):
    return cv2.filter2D(image, -1, kernel)


# 定义σ值和卷积核大小
sigmaList = [1, 2, 5]

# 计算水平和垂直方向的导数图像，并显示
fig, axs = plt.subplots(nrows=len(sigmaList), ncols=6, figsize=(54, 27))

for i, sigma in enumerate(sigmaList):
    ksize = int(np.ceil(sigma * 3) * 2 + 1)
    k = cv2.getDerivKernels(1, 0, ksize)
    k1 = cv2.getDerivKernels(0, 1, ksize)
    kernelDx = cv2.getDerivKernels(1, 0, ksize)[0].T
    kernelDy = cv2.getDerivKernels(0, 1, ksize)[1]
    # 计算水平方向的导数
    axon02Dx = convolve(axon02, kernelDx)
    cellnucleusDx = convolve(cell_nucleus, kernelDx)
    # 计算竖直方向的导数
    axon02Dy = convolve(axon02, kernelDy)
    cellnucleusDy = convolve(cell_nucleus, kernelDy)

    # 绘制图像
    axs[i][0].imshow(axon02, cmap="gray")
    axs[i][0].set_title("axon02")
    axs[i][1].imshow(axon02Dx, cmap="gray")
    axs[i][1].set_title(f"dx_axon02 (σ={sigma})")
    axs[i][2].imshow(axon02Dy, cmap="gray")
    axs[i][2].set_title(f"dy_axon02 (σ={sigma})")
    axs[i][3].imshow(cell_nucleus, cmap="gray")
    axs[i][3].set_title("cell_nucleus")
    axs[i][4].imshow(cellnucleusDx, cmap="gray")
    axs[i][4].set_title(f"dx_cell_nucleus (σ={sigma})")
    axs[i][5].imshow(cellnucleusDy, cmap="gray")
    axs[i][5].set_title(f"dy_cell_nucleus (σ={sigma})")

plt.tight_layout()
plt.show()
