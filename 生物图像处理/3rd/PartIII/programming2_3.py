from PIL import Image
import numpy as np
import os
import cv2
import math
from tqdm import tqdm
from scipy.spatial import Delaunay
from scipy import stats
from scipy.io import savemat

# 3.1 Calibration of dark noise
imgPath = "../DataPics/"
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

print(bg_noise)

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
            if max == min:
                maximum.append([i,j])
                minimum.append([i,j])
                continue
            for a in range(3):
                for b in range(3):
                    if img[i+a][j+b] == max:
                        maximum.append([i+a, j+b])
                    if img[i+a][j+b] == min:
                        minimum.append([i+a, j+b])
    if flag:    #select the first image to show the differences of detection results
        maximum_5 = []
        minimum_5 = []
        flag = False
        for i in tqdm(range(len(img)-4)):
            for j in range(len(img[i])-4):
                d = img[i:i+5, j:j+5]
                max = d.max()
                min = d.min()
                if max == min:
                    maximum_5.append([i,j])
                    minimum_5.append([i,j])
                    continue
                for a in range(5):
                    for b in range(5):
                        if img[i+a][j+b] == max:
                            maximum_5.append([i+a,j+b])
                        if img[i+a][j+b] == min:
                            minimum_5.append([i+a,j+b])
        print("maxima differences:")
        diff = []
        for i in tqdm(range(len(maximum))):
            for j in range(len(maximum[i])):
                if maximum[i][j] in maximum_5:
                    continue
                else:
                    diff.append(maximum[i][j])
        # print(diff)
        savemat("diff.mat", {'diff':diff})
    maxima.append(maximum)
    minima.append(minimum)
            
# 3.3 Establishing the local association of maxima and minima
connection = []
for k in range(5):
    neighbors = []
    tri = Delaunay(maxima[k]+minima[k])
    for max in tqdm(maxima[k]):
        simplex = tri.find_simplex(max)
        points = tri.simplices[simplex]
        flg = False
        p1 = []
        for p in points:
            if p > len(maxima[k]):
                p1.append(minima[k][p-len(maxima[k])])
        if len(p1) > 1:
            dis = math.sqrt((p1[0][0]-max[0])**2+(p1[0][1]-max[1])**2)
            pp = p1[0]
            for p11 in p1:
                if dis > math.sqrt((p11[0]-max[0])**2+(p11[1]-max[1])**2):
                    dis = math.sqrt((p11[0]-max[0])**2+(p11[1]-max[1])**2)
                    pp = p11
            neighbors.append(pp)
        elif len(p1) == 0:
            neighbors.append([])
        else:
            neighbors.append(p1[0])
    #print(len(neighbors), len(maxima[k]))
    connection.append(neighbors)


# 3.4 Statistical selection of local maxima
# sigma_deltai^2 = sigma_il^2 + 1/3sigma_bg^2 = 1/3*bg_noise[1]^2
for k in range(len(maxima)):
    sigma_deltaI = 1/3 * bg_noise[k][1]**2
    T = []
    for i in range(len(maxima[k])):
        if len(connection[k][i]) != 0:
            max_x, max_y = maxima[k][i]
            min_x, min_y = connection[k][i]
            #print(max_x, max_y, min_x, min_y)
            delta_I = abs(imgs_blurred[k][max_x][max_y]-imgs_blurred[k][min_x][min_y])
            T.append(delta_I/sigma_deltaI)
    # print(T)
    stat, pvalue = stats.ttest_1samp(T, 1)             #T ~ t(0,1)
    print(stat, pvalue)
    savemat(f"001_a5_002_t00{k+1}.mat", {'mean':bg_noise[k][0], 'std':bg_noise[k][1], 'maxima':maxima[k], 'minima':minima[k], 'stat':stat, 'pvalue':pvalue})

        

