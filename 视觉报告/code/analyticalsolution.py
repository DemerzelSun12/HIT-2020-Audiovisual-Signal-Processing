from normalization_X_Y import *
import numpy as np


def analytical_solution_without_penalty(train_X, train_Y, poly_degree):
    """
    不加惩罚项的数值解法
    :param poly_degree: 多项式次数
    :param train_X: 训练集的X向量
    :param train_Y: 训练集的Y向量
    :return: 解向量
    """
    X, Y = normalization(train_X, train_Y,poly_degree)
    matrix = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    w_result = np.poly1d(matrix[::-1].reshape(poly_degree + 1))
    return w_result, matrix


def analytical_solution_with_penalty(train_X, train_Y, lam, poly_degree):
    """
    加惩罚项的数值解法
    :param poly_degree: 多项式次数
    :param train_X: 训练集的X矩阵
    :param train_Y: 训练集的Y向量
    :param lam: 惩罚项系数
    :return: 解向量
    """
    X, Y = normalization(train_X, train_Y,poly_degree)
    matrix = np.linalg.inv(X.T.dot(X) + lam * np.eye(X.shape[1])).dot(X.T).dot(Y)
    w_result = np.poly1d(matrix[::-1].reshape(poly_degree + 1))
    # print("w result analytical")
    # print(w_result)
    return w_result
