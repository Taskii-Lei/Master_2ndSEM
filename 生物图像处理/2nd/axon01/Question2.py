import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def show_16bit_img(img, save8bit=False, name=None, show=True):
    # 转换为 8 位图像以显示，方便可视化操作
    img_nrm = (img - np.min(img)) / (np.max(img) - np.min(img))
    tmp = np.uint8(255 * img_nrm)
    im = Image.fromarray(tmp)
    # 可以选择是否保存或显示该八位图像
    if save8bit:
        im.save(name)
    if show:
        im.show()
    return im


def draw_intensity(img, name=None,dpi=300):
    # 转成array
    tmp = np.asarray(img)
    max_pixel = np.max(tmp)
    print(tmp.dtype, max_pixel)
    # 统计各个像素的个数，然后绘制统计柱状图
    x = np.arange(0, max_pixel + 1, 1)
    intensity = np.zeros(max_pixel + 1)
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            intensity[tmp[i][j]] += 1
    plt.bar(x, intensity, width=1)
    plt.savefig(name, dpi=dpi)
    plt.show()


# 2.1	使用MATLAB或Python读取图像并显示。编写程序绘制其强度直方图。
# 请不要使用内置的图像强度直方图功能。
# 相反，请通过调用MATLAB或Python中的直方图函数来编写自己的图像强度直方图代码
if __name__ == '__main__':
    img = Image.open('axon01.tif')
    show_16bit_img(img,save8bit=True, name="Show_axon01.tiff")
    draw_intensity(img,"Intensity_of_axon1.jpg",dpi=1200)









