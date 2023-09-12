# 2.2 Fitting of the Airy disk using a Gaussian kernel
# Based on your implementation in 3.1, expand to include an option to find the best fit of the plotted Airy disk using
# a Gaussian kernel. In particular, for each of the six listed configurations, find the standard deviation σ of the
# Gaussian kernel that gives the best fit, and compare σ to the radius of the Airy disk. What conclusion can you draw
# from this comparison? (20 points)
# 根据您在3.1中的实现，扩展到包括一个选项，以使用高斯内核找到绘制的Airy磁盘的最佳拟合。特别是，对于列出的六种配置中的每一种，找到给出最佳拟合的高斯
# 核的标准偏差σ，并将σ与艾里圆盘的半径进行比较。你能从这个比较中得出什么结论？（20分）


from PartII_21 import Airy_Disk
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting

g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
fit_g = fitting.LevMarLSQFitter()


def Gaussian_fit_AiryDisk(NA, Lambda, SaveEach=True):
    Max_NA = min(NA)
    Max_Lambda = max(Lambda)
    x_lim = 0.61 * Max_Lambda / (3 * Max_NA)
    Mean = []
    Sigma = []
    for i in range(len(NA)):
        print("="*20, " NA:", NA[i], " Lambda:", Lambda[i], " ", "="*20)
        plt.xlim([-4 * x_lim, 4 * x_lim])
        r = 0.61 * Lambda[i] / NA[i]
        x = np.arange(-4 * x_lim, 4 * x_lim, 0.0001)

        hr = Airy_Disk(NA[i], Lambda[i], x)
        plt.plot(x, hr, label="Airy_Disk")

        g = fit_g(g_init, x, hr)
        u, sigma = g.mean.value, g.stddev.value
        Mean.append(u)
        Sigma.append(sigma)
        # y = (1/(np.sqrt(2*np.pi) * sigma)) * (np.exp(-(x-u)**2/(2 * (sigma ** 2))))
        # 归一化，使得最大值为 1，方便比较
        y = np.exp(-(x - u) ** 2 / (2 * (sigma ** 2)))
        plt.plot(x, y, label="Gaussian with sigma=" + str(int(sigma)))

        print("Gaussian: mean = " + str(u) + " sigma = " + str(sigma))
        print("r / sigma = ", r / sigma)

        if SaveEach:
            plt.title("NA = " + str(NA[i]) + " ; Lambda = " + str(Lambda[i]))
            plt.legend()
            plt.savefig("2.2/Gaussian_NA=" + str(NA[i]) + "_Lambda=" + str(Lambda[i])+".jpg")
            # plt.show()
            plt.close()


if __name__ == "__main__":
    Lambda = [480, 520, 680, 520, 520, 680]
    NA = [0.5, 0.5, 0.5, 1.0, 1.4, 1.5]
    Gaussian_fit_AiryDisk(NA, Lambda, SaveEach=True)
