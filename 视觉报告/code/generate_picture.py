import matplotlib.pyplot as plt
import numpy as np


def generate_picture(train_X, train_Y, fit_polynomial, plot_title, chart_title):
    """
    生成不同拟合方法对应图像
    :param train_X: 训练集的X矩阵
    :param train_Y: 训练集的Y向量
    :param fit_polynomial: 拟合多项式次数
    :param plot_title:  拟合方法的标题
    :param chart_title:  图片标题，包括多项式次数和数据量
    :return: 无
    """
    real_X = np.linspace(0, 1, 50)
    real_Y = np.sin(2 * np.pi * real_X)
    'train data'
    plt.plot(train_X, train_Y, 's', label='train data')
    fit_Y = fit_polynomial(real_X)
    'fit data'
    plt.plot(real_X, fit_Y, 'r', label=plot_title + ' result')
    'real data'
    plt.plot(real_X, real_Y, 'b', label='real data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=1)
    plt.title(chart_title)
    plt.show()
