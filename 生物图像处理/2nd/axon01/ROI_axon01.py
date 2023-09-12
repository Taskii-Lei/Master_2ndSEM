import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import Question2 as Q2
import imageio


def ROI_Crop(img, x1, y1, x2, y2, show_crop=True, save_crop=True, name=None, validate=True):
    # 转换为 8 位以显示剪裁区域
    img_uint8 = Q2.show_16bit_img(img, show=False)
    plt.figure(figsize=(10, 6))
    # 对剪裁区域在原图用矩形框出
    draw_it = ImageDraw.Draw(img_uint8)
    draw_it.rectangle([x1, y1, x2, y2], outline="red", width=5)
    plt.subplot(2, 1, 1)
    plt.imshow(img_uint8, cmap=plt.get_cmap('gray'))
    # 剪裁
    cut_img = img.crop([x1, y1, x2, y2])
    cut_img8 = Q2.show_16bit_img(cut_img, show=False)
    plt.subplot(2, 1, 2)
    plt.imshow(cut_img8, cmap=plt.get_cmap('gray'))
    # 显示了剪裁区域在原图中的位置，方便比对
    plt.savefig("preview_"+name+".jpg", dpi=1200)
    if show_crop:
        plt.show()
    # 需要保存的是 cut_img.tif
    if save_crop:
        imageio.imsave(name+".tif", cut_img)
    # 验证保存是否正确
    if validate:
        new_img = Image.open(name+'.tif')
        re_img = np.asarray(new_img)
        print("Type of the Cropped Image:", re_img.dtype)


if __name__ == '__main__':
    img = Image.open('axon01.tif')
    [x1, y1, x2, y2] = [50, 70, 170, 200]
    # 每个剪裁图像的名字都不一样，便于保存
    # 不需要指定格式，因为会被重用
    name = "Cropped_ROI_"+str(x1)+"_"+str(y1)+"_"+str(x2)+"_"+str(y1)
    ROI_Crop(img, x1, y1, x2, y2, save_crop=True, name=name, validate=True)



