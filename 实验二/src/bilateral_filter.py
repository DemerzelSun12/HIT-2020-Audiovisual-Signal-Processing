import numpy as np
from src.read_write_img import *


class BilateralFilter(object):
    def __init__(self, img, radius, color_sigma, s_sigma):
        self.img = img
        self.img_b, self.img_g, self.img_r = cv.split(img)
        self.img_b_result, self.img_g_result, self.img_r_result = cv.split(img)
        self.height, self.width = np.shape(self.img_b)
        self.radius = radius
        self.color_sigma = color_sigma
        self.s_sigma = s_sigma
        self.weight_s_y = []
        self.weight_s_x = []
        self.weight_s = []
        self.color_weight = []
        self.k_max = 0
        self.generate_gauss_filter_template()

    def generate_gauss_filter_template(self):
        color_coe = -0.5 / np.square(self.color_sigma)
        for i in range(256):
            self.color_weight.append(np.exp(i ** 2 * color_coe))
        space_coe = -0.5 / np.square(self.s_sigma)
        for i in range(-self.radius, self.radius + 1):
            for j in range(-self.radius, self.radius + 1):
                r_sq = np.exp((np.square(i) + np.square(j)) * space_coe)
                self.weight_s_x.append(i)
                self.weight_s_y.append(j)
                self.weight_s.append(r_sq)
                self.k_max += 1

    def bilateral_filter_main(self, img):
        for i in range(self.height):
            for j in range(self.width):
                value = 0
                weight = 0
                for index in range(self.k_max):
                    print("i, j, index ", i, j, index)
                    temp_x = self.weight_s_x[index] + i
                    temp_y = self.weight_s_y[index] + j
                    if temp_x >= self.height or temp_y >= self.width or temp_x < 0 or temp_y < 0:
                        val = 0
                    else:
                        val = img[temp_x][temp_y]
                    temp_w = np.float32(self.weight_s[index]) * np.float32(self.color_weight[np.abs(val - img[i][j])])
                    value += val * temp_w
                    weight += temp_w
                img[i][j] = np.uint8(value / weight)
        return img

    def bilateral_filter(self):
        b_result = self.bilateral_filter_main(self.img_b_result)
        print('b_result', b_result)
        g_result = self.bilateral_filter_main(self.img_g_result)
        print('g_result', g_result)
        r_result = self.bilateral_filter_main(self.img_r_result)
        print('r_result', r_result)
        result_img = cv.merge([b_result, g_result, r_result])
        print(result_img)
        return result_img


def main():
    img = read_source_img_from_file()
    bf = BilateralFilter(img, 3, 30, 80)
    result = bf.bilateral_filter()
    show_img(result, 'bilateral filter')
    write_img_to_file(result, 'filter_result/bilateral_filter')


if __name__ == '__main__':
    main()
