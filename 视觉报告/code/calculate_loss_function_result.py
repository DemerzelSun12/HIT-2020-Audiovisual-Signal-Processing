import numpy as np


def calculate_loss_function_result(w, X, Y):
    """
    计算代价函数的代价
    :param w: 多项式函数的系数矩阵(向量)
    :param X: 处理成矩阵的训练集X矩阵
    :param Y: 处理成矩阵的训练集Y矩阵
    :return: 代价函数的代价
    """
    # X, Y = normalization(train_X, train_Y)
    y_xn_w_tn = np.dot(X, w) - Y
    Ew = 1 / 2 * y_xn_w_tn.T.dot(y_xn_w_tn)
    return Ew
