import numpy as np
from src.read_write_img import *


class SobelEdgeDetection(object):
    def __init__(self):
        self.img_rgb = read_source_img_from_file()
        self.img_gray = read_source_img_with_gray_from_file()
        self.img_matrix = np.asarray(self.img_rgb)
        self.b, self.g, self.r = cv.split(self.img_rgb)
        self.b_matrix = np.asarray(self.b)
        self.g_matrix = np.asarray(self.g)
        self.r_matrix = np.asarray(self.r)
        self.b_matrix_filter = np.pad(self.b_matrix, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        self.g_matrix_filter = np.pad(self.g_matrix, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        self.r_matrix_filter = np.pad(self.r_matrix, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))

    @staticmethod
    def sobel_detection_gray(img_matrix):
        print("start to detection the edge of gray image with sobel operator")
        img_sobel = img_matrix.copy()
        height, width = np.shape(img_sobel)
        # print(np.shape(img_sobel))
        sobel_operator_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_operator_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        new_img = np.zeros((height, width))
        height_top = height - 1
        width_top = width - 1
        for i in range(1, height_top):
            for j in range(1, width_top):
                new_x = abs(np.sum(img_sobel[i - 1:i + 2, j - 1:j + 2] * sobel_operator_x))
                new_y = abs(np.sum(img_sobel[i - 1:i + 2, j - 1:j + 2] * sobel_operator_y))
                new_img[i, j] = np.sqrt(np.square(new_x) + np.square(new_y))
        new_img = new_img[1:height_top, 1:width_top]
        print("finish detecting the edge of gray image with sobel operator")
        return np.uint8(new_img)

    def sobel_detection_rgb(self):
        print("start to detection the edge of rgb image with sobel operator")
        b_result = self.sobel_detection_gray(self.b_matrix_filter)
        g_result = self.sobel_detection_gray(self.g_matrix_filter)
        r_result = self.sobel_detection_gray(self.r_matrix_filter)
        print("finish detecting the edge of rgb image with sobel operator")
        img = cv.merge((b_result, g_result, r_result))
        return img


def main():
    img = SobelEdgeDetection()
    # rgb_sobel_detection = img.sobel_detection_rgb()
    # show_img(rgb_sobel_detection, "sobel detection of rgb lena")
    # write_img_to_file(rgb_sobel_detection, "sobel detection of rgb lena")

    gray_sobel_detection = img.sobel_detection_gray(img.img_gray)
    show_img(gray_sobel_detection, "sobel detection of gray lena")
    write_img_to_file(gray_sobel_detection, "sobel detection of gary lena")


if __name__ == '__main__':
    main()
