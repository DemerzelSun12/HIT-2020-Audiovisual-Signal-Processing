import matplotlib.pyplot as plt


def generate_loss_picture(x_array, loss_train_array, loss_test_array):
    """
    生成损失函数随次数变化图片
    :param x_array: 横坐标向量
    :param loss_train_array: 训练集的损失
    :param loss_test_array: 测试集的损失
    :return: 无
    """
    # print(x_array)
    # print(loss_train_array)
    # print(loss_test_array)
    plt.plot(x_array, loss_train_array, 'b', label='loss of train data')
    plt.plot(x_array, loss_test_array, 'r', label='loss of test data')
    plt.xlabel('poly degree')
    plt.ylabel('loss')
    plt.legend(loc=1)
    plt.title('loss of different poly degree')
    plt.show()
