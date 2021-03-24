import numpy as np


def np_polyfit(train_X, train_Y, poly_degree):
    """
    调用numpy.polyfit()方法拟合数据
    :param train_X: 训练集的X矩阵
    :param train_Y: 训练集的Y向量
    :param poly_degree: 多项式次数
    :return: 拟合的信息
    """
    result = np.polyfit(train_X, train_Y, poly_degree)
    picture = np.poly1d(result)
    # print(result)
    return picture
