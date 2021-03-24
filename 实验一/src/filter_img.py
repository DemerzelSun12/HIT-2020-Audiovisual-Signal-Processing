import numpy as np
from PIL import Image
from src.read_write_img import *


class FilterImg(object):

    def __init__(self, img):
        self.img = img
        self.img_matrix = np.asarray(self.img)
        # self.img_filter = np.pad(self.img_matrix, ((2, 2), (2, 2)), 'constant', constant_values=(0, 0))
        self.b, self.g, self.r = cv.split(self.img)
        self.b_matrix = np.asarray(self.b)
        self.g_matrix = np.asarray(self.g)
        self.r_matrix = np.asarray(self.r)
        self.b_matrix_filter = np.pad(self.b_matrix, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        self.g_matrix_filter = np.pad(self.g_matrix, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        self.r_matrix_filter = np.pad(self.r_matrix, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))

    def filter_img(self):
        self.median_filter_rgb()
        self.mean_filter_rgb()

    @staticmethod
    def median_filter_gray(img_matrix):
        img_filter = img_matrix.copy()
        height, width = np.shape(img_filter)
        # kernel = np.ones((3, 3))
        height_top = height - 1
        width_top = width - 1
        for i in range(1, height_top):
            for j in range(1, width_top):
                img_filter[i, j] = np.median(img_filter[i - 1:i + 2, j - 1:j + 2].copy().reshape((1, -1))[0])
        img_filter = img_filter[1:height_top, 1:width_top]
        return img_filter

    def median_filter_rgb(self):
        # print(self.img_matrzix)
        # print("-----------")
        # print(self.b_matrix_filter)
        # print(self.g_matrix_filter)
        # print(self.r_matrix_filter)
        print("start to median the image.")
        b_median_filter_result = self.median_filter_gray(self.b_matrix_filter)
        g_median_filter_result = self.median_filter_gray(self.g_matrix_filter)
        r_median_filter_result = self.median_filter_gray(self.r_matrix_filter)
        # print(b_median_filter_result)
        # b_img = Image.fromarray(b_median_filter_result)
        # g_img = Image.fromarray(g_median_filter_result)
        # r_img = Image.fromarray(r_median_filter_result)
        print("finish median filter image.")
        # show_img(b_median_filter_result, "blue")
        # show_img(g_median_filter_result, "green")
        # show_img(r_median_filter_result, "red")
        img = cv.merge((b_median_filter_result, g_median_filter_result, r_median_filter_result))
        return img

    @staticmethod
    def mean_filter_gray(img_matrix):
        img_filter = img_matrix.copy()
        height, width = np.shape(img_filter)
        kernel = np.ones((3, 3))
        height_top = height - 1
        width_top = width - 1
        for i in range(1, height_top):
            for j in range(1, width_top):
                img_filter[i, j] = np.sum(kernel * img_filter[i - 1:i + 2, j - 1:j + 2]) // (3 ** 2)
        img_filter = img_filter[1:height_top, 1:width_top]
        return img_filter

    def mean_filter_rgb(self):
        # print(self.img_matrix)
        # print("-----------")
        # print(b_matrix_filter)
        # print(g_matrix_filter)
        # print(r_matrix_filter)
        print("start to mean the image.")
        b_mean_filter_result = self.mean_filter_gray(self.b_matrix_filter)
        g_mean_filter_result = self.mean_filter_gray(self.g_matrix_filter)
        r_mean_filter_result = self.mean_filter_gray(self.r_matrix_filter)
        print("finish mean filter image.")
        # b_img = Image.fromarray(b_mean_filter_result)
        # g_img = Image.fromarray(g_mean_filter_result)
        # r_img = Image.fromarray(r_mean_filter_result)
        img = cv.merge((b_mean_filter_result, g_mean_filter_result, r_mean_filter_result))
        return img


def main():
    gauss_noise_img = read_gauss_noise_img_from_file()

    gauss_median_filter_img = FilterImg(gauss_noise_img)
    gauss_median_img = gauss_median_filter_img.median_filter_rgb()
    show_img(gauss_median_img, "median filter of gauss noise")
    write_img_to_file(gauss_median_img, "median filter of gauss noise")

    gauss_mean_filter_img = FilterImg(gauss_noise_img)
    gauss_mean_img = gauss_mean_filter_img.mean_filter_rgb()
    show_img(gauss_mean_img, "mean filter of gauss noise")
    write_img_to_file(gauss_mean_img, "mean filter of gauss noise")

    salt_pepper_noise_img = read_salt_pepper_noise_img_from_file()

    salt_pepper_median_filter_img = FilterImg(salt_pepper_noise_img)
    salt_pepper_median_img = salt_pepper_median_filter_img.median_filter_rgb()
    show_img(salt_pepper_median_img, "median filter of salt pepper noise")
    write_img_to_file(salt_pepper_median_img, "median filter of salt pepper noise")

    salt_pepper_mean_filter_img = FilterImg(salt_pepper_noise_img)
    salt_pepper_mean_img = salt_pepper_mean_filter_img.mean_filter_rgb()
    show_img(salt_pepper_mean_img, "mean filter of salt pepper noise")
    write_img_to_file(salt_pepper_mean_img, "mean filter of salt pepper noise")


if __name__ == '__main__':
    main()
