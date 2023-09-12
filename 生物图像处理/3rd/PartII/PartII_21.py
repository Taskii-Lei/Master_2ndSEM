import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.special as ss


def Airy_Disk(NA, LAMBDA, r):
    a = 2 * math.pi * NA / LAMBDA
    hr = (2 * ss.jv(1, a * r) / (a * r)) ** 2
    return hr


def plot_AD(NA, Lambda, x, name):
    for i in range(len(NA)):
        token = "NA=" + str(NA[i]) + "; lambda=" + str(Lambda[i])
        r = 0.61 * Lambda[i] / NA[i]
        print(token, ", and the radius is :", r)
        hr = Airy_Disk(NA[i], Lambda[i], x)
        plt.plot(x, hr, label=token)
        if name.split('_')[-1] == "zero":
            plt.scatter([-r, r], [0, 0], label="(" + str(r) + ",0)")
    plt.legend()
    plt.savefig("2.1Pics/" + name + ".jpg")
    plt.show()


if __name__ == "__main__":
    x = np.arange(-3000, 3000, 0.01)

    # 整体画图
    Lambda = [480, 520, 680, 520, 520, 680]
    NA = [0.5, 0.5, 0.5, 1.0, 1.4, 1.5]
    plot_AD(NA, Lambda, x, name="plot_Airy_Disk")

    # 整体画图 —— 带有零点
    Lambda = [480, 520, 680, 520, 520, 680]
    NA = [0.5, 0.5, 0.5, 1.0, 1.4, 1.5]
    plot_AD(NA, Lambda, x, name="plot_Airy_Disk_with_zero")

    # 控制变量：固定 NA=0.5
    Lambda = [480, 520, 680]
    NA = [0.5, 0.5, 0.5]
    plot_AD(NA, Lambda, x, name="plot_AD_set_NA=0_5")

    # 控制变量：固定 Lambda=520
    Lambda = [520, 520, 520]
    NA = [0.5, 1.0, 1.4]
    plot_AD(NA, Lambda, x, name="plot_AD_fix_Lambda=520")

    # 控制变量：固定 Lambda=680
    Lambda = [680, 680]
    NA = [0.5, 1.5]
    plot_AD(NA, Lambda, x, name="plot_AD_fix_Lambda=680.jpg")
