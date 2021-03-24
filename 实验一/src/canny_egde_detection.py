import numpy as np
from src.read_write_img import *


class CannyEdgeDetection(object):
    """
    Canny边缘检测
    """
    def __init__(self):
        # self.img_rgb = read_source_img_from_file()
        self.img_gray = read_source_img_with_gray_from_file()
        self.img_matrix = np.asarray(self.img_gray)
        self.img_matrix_filter = np.pad(self.img_matrix, ((2, 2), (2, 2)), 'constant', constant_values=(0, 0))
        self.gaussian_filter = None
        self.img_gradient = None
        self.theta = None
        self.matrix_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        self.matrix_x = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        self.nms_result = None
        self.dtd_result = None

    def gaussian_filtering(self):
        """
        高斯平滑
        :return: 无
        """
        print("start to gauss filter of the image")
        sigma1 = sigma2 = 1
        sum = 0
        gaussian = np.zeros([5, 5])
        for i in range(5):
            for j in range(5):
                gaussian[i, j] = np.exp(
                    -1 / 2 * (np.square(i - 3) / np.square(sigma1) + (np.square(j - 3) / np.square(sigma2)))) / (
                                         2 * np.pi * sigma1 * sigma2)
                sum = sum + gaussian[i, j]
        gaussian = gaussian / sum
        # print(gaussian)

        img_gauss = self.img_matrix_filter.copy()
        height, width = np.shape(img_gauss)
        new_img = np.zeros((height, width))
        height_top = height - 2
        width_top = width - 2
        for i in range(2, height_top):
            for j in range(2, width_top):
                new_img[i, j] = np.sum(gaussian * img_gauss[i - 2:i + 3, j - 2:j + 3])
        # print(np.shape(new_img))
        new_img = new_img[1:height_top + 1, 1:width_top + 1]
        # print(np.shape(new_img))
        self.gaussian_filter = new_img
        print("finish gauss filtering the image")
        show_img(np.uint8(new_img), "gauss filtering the image")
        # return np.uint8(new_img)

    def calculate_gradient_and_direction(self):
        """
        计算各像素的梯度和梯度方向
        :return: 无
        """
        print("start to calculate the gradient and direction of the image")
        height, width = np.shape(self.gaussian_filter)
        # print(np.shape(self.gaussian_filter))
        img_gradient = self.gaussian_filter.copy()
        self.img_gradient = np.zeros((height, width), dtype="uint8")
        self.theta = np.zeros((height, width))
        height_top = height - 1
        width_top = width - 1
        for i in range(1, height_top):
            for j in range(1, width_top):
                gradient_y = (
                    np.dot(np.array([1, 1, 1]), (self.matrix_y * img_gradient[i - 1:i + 2, j - 1:j + 2]))).dot(
                    np.array([[1], [1], [1]]))
                gradient_x = (
                    np.dot(np.array([1, 1, 1]), (self.matrix_x * img_gradient[i - 1:i + 2, j - 1:j + 2]))).dot(
                    np.array([[1], [1], [1]]))

                if gradient_x[0] == 0:
                    self.theta[i - 1, j - 1] = 90
                    continue
                else:
                    temp_direction = (np.arctan(gradient_y[0] / gradient_x[0])) * 180 / np.pi
                if gradient_x[0] * gradient_y[0] > 0:
                    if gradient_x[0] > 0:
                        self.theta[i - 1, j - 1] = np.abs(temp_direction)
                    else:
                        self.theta[i - 1, j - 1] = np.abs(temp_direction) - 180
                if gradient_x[0] * gradient_y[0] < 0:
                    if gradient_x[0] > 0:
                        self.theta[i - 1, j - 1] = -1 * np.abs(temp_direction)
                    else:
                        self.theta[i - 1, j - 1] = 180 - np.abs(temp_direction)
                self.img_gradient[i, j] = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

        for i in range(1, height_top):
            for j in range(1, width_top):
                if ((self.theta[i, j] >= -22.5) and (self.theta[i, j] < 22.5)) or (
                        (self.theta[i, j] <= -157.5) and (self.theta[i, j] >= -180)) or (
                        (self.theta[i, j] >= 157.5) and (self.theta[i, j] < 180)):
                    self.theta[i, j] = 0.0
                elif ((self.theta[i, j] >= 22.5) and (self.theta[i, j] < 67.5)) or (
                        (self.theta[i, j] <= -112.5) and (self.theta[i, j] >= -157.5)):
                    self.theta[i, j] = 45.0
                elif ((self.theta[i, j] >= 67.5) and (self.theta[i, j] < 112.5)) or (
                        (self.theta[i, j] <= -67.5) and (self.theta[i, j] >= -112.5)):
                    self.theta[i, j] = 90.0
                elif ((self.theta[i, j] >= 112.5) and (self.theta[i, j] < 157.5)) or (
                        (self.theta[i, j] <= -22.5) and (self.theta[i, j] >= -67.5)):
                    self.theta[i, j] = -45.0
        show_img(self.img_gradient, "calculate the gradient and direction of the image")
        print("finish calculating the gradient and direction of the image")

    def non_maximum_suppression(self):
        """
        非极大值抑制
        :return: 无
        """
        print("start to nms")
        height, width = np.shape(self.img_gradient)
        img = self.img_gradient.copy()
        img_nms = np.zeros((height, width))
        height_top = height - 1
        width_top = width - 1
        for i in range(1, height_top):
            for j in range(1, width_top):
                if (self.theta[i, j] == 0.0) and (img[i, j] == np.max([img[i, j], img[i + 1, j], img[i - 1, j]])):
                    img_nms[i, j] = img[i, j]
                if (self.theta[i, j] == 45.0) and (
                        img[i, j] == np.max([img[i, j], img[i - 1, j + 1], img[i + 1, j - 1]])):
                    img_nms[i, j] = img[i, j]
                if (self.theta[i, j] == 90.0) and (img[i, j] == np.max([img[i, j], img[i, j + 1], img[i, j - 1]])):
                    img_nms[i, j] = img[i, j]
                if (self.theta[i, j] == -45.0) and (
                        img[i, j] == np.max([img[i, j], img[i - 1, j - 1], img[i + 1, j + 1]])):
                    img_nms[i, j] = img[i, j]
        self.nms_result = img_nms
        show_img(self.nms_result, "nms result")
        print("finish nms")

    def dual_threshold_detection(self):
        """
        双阈值检测
        :return: 无
        """
        print("start to dual threshold detection")
        dtd_img = np.zeros(np.shape(self.nms_result))
        TL = 0.15 * np.max(self.nms_result)
        TH = 0.3 * np.max(self.nms_result)
        nms = self.nms_result
        height, width = np.shape(self.nms_result)
        height_top = height - 1
        width_top = width - 1
        for i in range(1, height_top):
            for j in range(1, width_top):
                if nms[i, j] < TL:
                    dtd_img[i, j] = 0
                elif nms[i, j] > TH:
                    dtd_img[i, j] = 255
                elif (nms[i + 1, j] < TH) or (nms[i - 1, j] < TH) or (nms[i, j + 1] < TH) or (nms[i, j - 1] < TH) or (
                        nms[i - 1, j - 1] < TH) or (nms[i - 1, j + 1] < TH) or (nms[i + 1, j + 1] < TH) or (
                        nms[i + 1, j - 1] < TH):
                    dtd_img[i, j] = 255
        self.dtd_result = dtd_img
        show_img(self.dtd_result, "dual threshold detection")
        print("finish dual threshold detection")

    def canny_detection_gray(self):
        """
        Canny检测类的主程序
        :return:
        """
        print("start to detect the edge of image with canny operator")
        self.gaussian_filtering()
        self.calculate_gradient_and_direction()
        self.non_maximum_suppression()
        self.dual_threshold_detection()
        print("finish detecting the edge of image with canny operator")


def main():
    canny_img = CannyEdgeDetection()
    canny_img.canny_detection_gray()
    canny_edge_detection = canny_img.dtd_result
    show_img(canny_edge_detection, "canny edge detection of lena")
    write_img_to_file(canny_edge_detection, "canny edge detection of lena")


if __name__ == '__main__':
    main()
