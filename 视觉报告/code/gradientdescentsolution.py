from calculate_loss_function_result import *
from normalization_X_Y import *
from global_number import *


def gradient_descent(train_X, train_Y, learning_rate, deviation, poly_degree, lam):
    """
    梯度下降法求多项式函数的系数矩阵(向量)
    :param lam: 惩罚项系数
    :param poly_degree: 多项式拟合次数
    :param train_X: 训练集的X向量
    :param train_Y: 训练集的Y向量
    :param learning_rate: 梯度下降法的学习率
    :param deviation: 梯度下降法最终允许的误差范围
    :return: 多项式函数的系数矩阵，迭代次数的list，每次迭代对应的误差
    """
    X, Y = normalization(train_X, train_Y, poly_degree)
    w = 0.1 * np.ones((poly_degree + 1, 1))
    number = 0
    grad = (1.0 / data_number) * (X.T.dot(X).dot(w) - X.T.dot(Y) + lam * w)
    loss0 = 0
    loss1 = calculate_loss_function_result(w, X, Y)
    while abs(loss1 - loss0) > deviation:
        number += 1
        w -= learning_rate * grad
        loss0 = loss1
        loss1 = calculate_loss_function_result(w, X, Y)
        if np.all(loss1 - loss0 > 0):
            learning_rate *= 0.5
        grad = (1.0 / data_number) * (X.T.dot(X).dot(w) - X.T.dot(Y) + lam * w)
    w_result = np.poly1d(w[::-1].reshape(poly_degree + 1))
    return w_result
