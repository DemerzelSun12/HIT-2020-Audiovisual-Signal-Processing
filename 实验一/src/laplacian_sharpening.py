import numpy as np
from src.read_write_img import *


class LaplacianSharpening(object):
    def __init__(self):
        self.img = read_source_img_with_gray_from_file()
        self.img_matrix = np.asarray(self.img)
        self.img_matrix_filter = np.pad(self.img_matrix, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        self.laplace_4 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.laplace_8 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    def laplacian_sharpening(self):
        print("start laplace sharpen")
        img_matrix = self.img_matrix_filter.copy()
        height, width = np.shape(img_matrix)
        height_top = height - 1
        width_top = width - 1
        laplace_4_img = np.zeros((height, width))
        laplace_8_img = np.zeros((height, width))
        for i in range(1, height_top):
            for j in range(1, width_top):
                laplace_4_img[i, j] = np.sum(self.laplace_4 * img_matrix[i - 1:i + 2, j - 1:j + 2])
                laplace_8_img[i, j] = np.sum(self.laplace_8 * img_matrix[i - 1:i + 2, j - 1:j + 2])
                if laplace_4_img[i, j] <= 0:
                    laplace_4_img[i, j] = 0
                if laplace_8_img[i, j] <= 0:
                    laplace_8_img[i, j] = 0
        laplace_4_img = laplace_4_img[1:height_top, 1:width_top]
        laplace_8_img = laplace_8_img[1:height_top, 1:width_top]
        print("finish laplace sharpen")
        return np.uint8(laplace_4_img), np.uint8(laplace_8_img)


def main():
    laplace_img = LaplacianSharpening()
    laplace_4, laplace_8 = laplace_img.laplacian_sharpening()
    show_img(np.uint8(laplace_4), "laplace sharpening with 4 area")
    show_img(np.uint8(laplace_8), "laplace sharpening with 4 area")
    write_img_to_file(np.uint8(laplace_4), "laplace sharpening with 4 area")
    write_img_to_file(np.uint8(laplace_8), "laplace sharpening with 8 area")


if __name__ == '__main__':
    main()
