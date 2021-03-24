from numpy_polyfit import *
from generate_picture import *
from analytical_solution import *
from generate_loss_picture import *
from gradient_descent_solution import *
from conjugate_gradient_solution import *


def generate_train_data(mu, sigma):
    """
    生成训练样本数据，来自函数y=sin(2 * pi * x)，并加入噪声
    :param mu: 高斯噪声的均值
    :param sigma: 高斯噪声的方差
    :return: 生成的训练数据集的X向量和Y向量
    """
    train_X = np.linspace(0, 1, data_number)
    noise = np.random.normal(mu, sigma, data_number)
    train_Y = np.sin(2 * np.pi * train_X) + noise
    # print(train_X)
    # print(np.sin(2 * np.pi * train_X))
    # print(train_Y)
    return train_X, train_Y


def generate_test_data():
    """
    生成测试样本数据，来自函数y=sin(2 * pi * x)
    :return: 生成的测试数据集的X向量和Y向量
    """
    test_X = np.linspace(0, 1, 4 * data_number)
    test_Y = np.sin(2 * np.pi * test_X)
    return test_X, test_Y


def main():
    """
    主函数
    :return: 无
    """
    train_X, train_Y = generate_train_data(mu=0, sigma=0.1)
    test_X, test_Y = generate_test_data()
    loss_train_list = []
    loss_test_list = []
    x_list = []

    for i in range(11):
        poly_degree = i
        test_X_normalize, test_Y_normalize = normalization(test_X, test_Y, poly_degree)
        train_X_normalize, train_Y_normalize = normalization(train_X, train_Y, poly_degree)

        title = 'poly degree=' + str(poly_degree) + ', data number=' + str(data_number) + '.'

        """
        numpy库的polyfit方法
        """
        numpy_result = np_polyfit(train_X, train_Y, poly_degree)
        generate_picture(train_X, train_Y, numpy_result, 'numpy.polyfit', title)
        """
        数值解法不加惩罚项
        """
        analytical_result_without_penalty, result = analytical_solution_without_penalty(train_X, train_Y, poly_degree)
        loss_train = calculate_loss_function_result(result, train_X_normalize, train_Y_normalize)
        loss_test = calculate_loss_function_result(result, test_X_normalize, test_Y_normalize)
        # print(loss_train[0])
        loss_train_list.append(loss_train[0])
        loss_test_list.append(loss_test[0])
        x_list.append(i)
        # print(loss_train)
        # print(loss_test)
        # print('-------')

        generate_picture(train_X, train_Y, analytical_result_without_penalty, 'analytical without penalty', title)

        """
        数值解法加惩罚项
        """
        coefficient_of_penalty = 0.0005
        analytical_result_with_penalty = analytical_solution_with_penalty(train_X, train_Y, coefficient_of_penalty,
                                                                          poly_degree)
        generate_picture(train_X, train_Y, analytical_result_with_penalty, 'analytical with penalty', title)

        learning_rate = 0.1
        deviation = 1e-5

        """
        梯度下降法，加惩罚项
        """
        gradient_descent_result_with_penalty = gradient_descent(train_X, train_Y, learning_rate, deviation,
                                                                poly_degree, lam=0.01)
        generate_picture(train_X, train_Y, gradient_descent_result_with_penalty, 'gradient descent without penalty', title)

        """
        梯度下降法，不加惩罚项
        """
        gradient_descent_result_without_penalty = gradient_descent(train_X, train_Y, learning_rate, deviation,
                                                                   poly_degree, lam=0)
        generate_picture(train_X, train_Y, gradient_descent_result_without_penalty, 'gradient descent with penalty',
                         title)

        """
        共轭梯度法，加惩罚项
        """
        conjugate_gradient_result_with_penalty = conjugate_gradient_solution(train_X, train_Y, poly_degree, lam=0.001
                                                                             )
        generate_picture(train_X, train_Y, conjugate_gradient_result_with_penalty, 'conjugate gradient with penalty',
                         title)
        """
        共轭梯度法，不加惩罚项
        """
        conjugate_gradient_result_without_penalty = conjugate_gradient_solution(train_X, train_Y, poly_degree, lam=0
                                                                                )
        generate_picture(train_X, train_Y, conjugate_gradient_result_without_penalty,
                         'conjugate gradient without penalty', title)

    x_array = np.array(x_list)
    loss_test_array = np.array(loss_test_list)
    loss_train_array = np.array(loss_train_list)
    generate_loss_picture(x_array, loss_train_array, loss_test_array)


if __name__ == '__main__':
    main()
