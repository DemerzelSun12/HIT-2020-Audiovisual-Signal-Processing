from src.generate_histogram import *


class HistogramEqualization(object):
    def __init__(self, img):
        self.img = img
        self.img_hist = Histogram(img)
        self.img_histogram = self.img_hist.generate_histogram()
        self.img_hist.show_histogram("histogram of source image", '../data/histogram_result/',
                                     "histogram of source image")
        self.histogram_equal = np.zeros(256)
        self.histogram_spec = np.zeros(256)

    def histogram_equalization(self):
        self.histogram_equal = np.cumsum(self.img_histogram)
        height, width = np.shape(self.img)
        self.histogram_equal = np.uint8(255 * self.histogram_equal + 0.5)
        out_image = np.uint8(np.zeros((height, width)))
        for i in range(height):
            for j in range(width):
                out_image[i, j] = self.histogram_equal[self.img[i, j]]
        cv.imshow("histogram equalization", out_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        write_img_to_file(out_image, 'histogram_result/histogram equalization of source image')
        histogram = Histogram(out_image)
        chart = histogram.generate_histogram()
        histogram.show_histogram("histogram equalization of source image", '../data/histogram_result/',
                                 "histogram equalization of source image")


def main():
    img = read_source_img_with_gray_from_file()
    ht = HistogramEqualization(img)
    ht.histogram_equalization()


if __name__ == '__main__':
    main()
