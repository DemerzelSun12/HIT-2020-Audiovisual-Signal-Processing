import numpy as np
import matplotlib.pyplot as plt
from src.read_write_img import *


class Histogram(object):
    def __init__(self, img):
        self.img = img
        self.histogram = np.zeros(256, dtype=float)

    def generate_histogram(self):
        height, width = np.shape(self.img)
        for i in range(height):
            for j in range(width):
                # print(self.img[i, j])
                self.histogram[self.img[i, j]] += 1
        self.histogram = self.histogram / np.sum(self.histogram)
        return self.histogram

    def show_histogram(self, title, file_path, file_name, color='b'):
        plt.bar(range(len(self.histogram)), self.histogram, width=1, color=color)
        plt.title(title)
        plt.savefig(file_path + file_name + '.jpg')
        plt.show()


def main():
    img = read_source_img_with_gray_from_file()
    histogram = Histogram(img)
    _ = histogram.generate_histogram()
    title = 'histogram of source image'
    file_path = '../data/histogram_result/'
    histogram.show_histogram(title, file_path, title)


if __name__ == '__main__':
    main()
