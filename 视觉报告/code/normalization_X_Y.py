import numpy as np


def normalization(train_X, train_Y, poly_degree):
    """
    处理数据
    :param train_X: 训练集的X矩阵
    :param train_Y: 训练集的Y向量
    :param poly_degree: 多项式次数
    :return: 处理后的X矩阵与Y向量
    """
    X = np.zeros((train_X.size, poly_degree + 1))
    for i in range(train_X.size):
        row = np.ones(poly_degree + 1) * train_X[i]
        row = np.power(row, np.arange(0, poly_degree + 1))
        X[i] = row
    Y = train_Y.reshape((train_Y.size, 1))
    return X, Y
