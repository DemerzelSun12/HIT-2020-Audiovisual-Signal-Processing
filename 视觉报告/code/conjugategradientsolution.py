from normalization_X_Y import *
import numpy as np



def conjugate_gradient_solution(train_X, train_Y, poly_degree, lam):
    """
    共轭梯度法求损失函数最小值对应的系数向量
    :param train_X: 训练集的X矩阵
    :param train_Y: 训练集的Y向量
    :param poly_degree: 拟合多项式次数
    :param lam: 惩罚项系数
    :return: 系数向量
    """
    X, Y = normalization(train_X, train_Y, poly_degree)
    # Q = (1 / data_number) * (X.T * X + lam * np.mat(np.eye(X.shape[1])))
    Q = np.dot(X.T, X) + lam * np.eye(X.shape[1])
    w = np.zeros((poly_degree + 1, 1))
    grad = np.dot(X.T, X).dot(w) - np.dot(X.T, Y) + lam * w
    r = -grad
    pre = r
    """
    Q: 矩阵X的转置与矩阵X的内积加罚项，用于计算方向向量
    grad: 梯度
    w: 初始化向量
    """
    for i in range(poly_degree + 1):
        """
        A: 下降步长
        pre: 方向向量
        r: 残差向量
        """
        A = (r.T.dot(r)) / (pre.T.dot(Q).dot(pre))
        r_pre = r
        w = w + A * pre
        r = r - (A * Q).dot(pre)
        beta = (r.T.dot(r)) / (r_pre.T.dot(r_pre))
        pre = r + beta * pre
    w_result = np.poly1d(w[::-1].reshape(poly_degree + 1))
    return w_result
