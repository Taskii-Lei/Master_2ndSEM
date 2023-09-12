from PIL import Image
import numpy as np
import os
import cv2
import math
from tqdm import tqdm

# 3.1 Calibration of dark noise
imgPath = "BIP_Project02_image_sequence/"
images = os.listdir(imgPath)
imgs = []
bg_noise = []
bg_noise_rectangular = (0, 0, 100, 10)
for img in images:
    img_path = imgPath + '/' + img
    image = Image.open(img_path)
    imgs.append(image)
    img_crop = np.array(image.crop(bg_noise_rectangular))
    bg_noise.append([np.mean(img_crop), np.std(img_crop)])

# print(bg_noise)

# 3.2 Detection of local maxima and local minima
# D = 0.61*lambda/NA = 0.61*515/1.4 nm = 224.39 nm = (224.39/65) pixels = 3.45 pixels
# sigma = D/3 = 1.15
imgs_blurred = []
for img in imgs:
    img_blurred = cv2.GaussianBlur(np.array(img), (3,3), 1.15)
    #Image.fromarray(imgs_blurred).show()
    imgs_blurred.append(img_blurred)
maxima = []
minima = []
for img in imgs_blurred:
    maximum = []
    minimum = []
    flag = False
    for i in tqdm(range(len(img)-2)):
        for j in range(len(img[i])-2):
            d = img[i:i+2, j:j+2]
            max = d.max()
            min = d.min()
            for a in range(3):
                for b in range(3):
                    if img[i+a][j+b] == max and [i+a, j+b] not in maximum:
                        maximum.append([i+a, j+b])
                    if img[i+a][j+b] == min and [i+a, j+b] not in minimum:
                        minimum.append([i+a, j+b])
    if flag:    #select the first image to show the differences of detection results
        maximum_5 = []
        minimum_5 = []
        flag = False
        for i in range(len(img)-4):
            for j in range(len(img[i])-4):
                d = img[i:i+5, j:j+5]
                max = d.max()
                min = d.min()
                for a in range(5):
                    for b in range(5):
                        if img[i+a][j+b] == max and [i+a, j+b] not in maximum_5:
                            maximum_5.append([i+a, j+b])
                        if img[i+a][j+b] == min and [i+a, j+b] not in minimum_5:
                            minimum_5.append([i+a, j+b])
        print("maxima in 3*3 but not in 5*5:")
        for i in range(len(maximum)):
            if maximum[i] in maximum_5:
                continue
            else:
                print(maximum[i])
        print("maxima in 5*5 but not in 3*3:")
        for i in range(len(maximum_5)):
            if maximum_5[i] in maximum:
                continue
            else:
                print(maximum_5[i])
    maxima.append(maximum)
    minima.append(minimum)
            
# 3.3 Establishing the local association of maxima and minima
connection = {}
for i in range(len(maxima)):
    for max in maxima[i]:
        dis = 10000
        points = []
        p = max
        for min in minima[i]:
            d = math.sqrt((max[0]-min[0])**2+(max[1]-min[1])**2)
            if dis > d:
                dis = d
                p = min
            if dis <= 3:
                points.append(min)
        if p in points:
            continue
        else:
            points.append(p)
        connection[max] = points

# 3.4 Statistical selection of local maxima

