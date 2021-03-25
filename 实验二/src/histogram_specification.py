from src.read_write_img import *
from src.generate_histogram import *
import matplotlib.pyplot as plt
import os


class HistogramSpecification(object):
    def __init__(self, template_img):
        self.template_img = template_img
        self.img_b, self.img_g, self.img_r = cv.split(self.template_img)
        self.histogram_b = Histogram(self.img_b).generate_histogram()
        self.histogram_g = Histogram(self.img_g).generate_histogram()
        self.histogram_r = Histogram(self.img_r).generate_histogram()
        self.generate_histogram('../data/histogram_result/', 'histogram of template image',
                                'histogram of template image')
        self.histogram_reverse_b = self.calculate_reverse_mapping(
            np.uint8((256 - 1) * np.cumsum(self.histogram_b) + 0.5))
        self.histogram_reverse_g = self.calculate_reverse_mapping(
            np.uint8((256 - 1) * np.cumsum(self.histogram_g) + 0.5))
        self.histogram_reverse_r = self.calculate_reverse_mapping(
            np.uint8((256 - 1) * np.cumsum(self.histogram_r) + 0.5))
        self.target_b = None
        self.target_g = None
        self.target_r = None
        self.template2target_b = None
        self.template2target_g = None
        self.template2target_r = None

    @staticmethod
    def calculate_reverse_mapping(histogram):
        mapping = list(histogram)
        off_mapping = []
        pre_f = 0.0
        for i in range(256):
            try:
                temp_f = mapping.index(i)
                pre_f = temp_f
            except ValueError:
                temp_f = pre_f
            off_mapping.append(temp_f)
        off_mapping = np.array(off_mapping)
        # plt.bar(range(len(off_mapping)), off_mapping, width=1)
        # plt.title("reverse mapping")
        # plt.show()
        return off_mapping

    def generate_new_image(self, target_img, file):
        self.target_b, self.target_g, self.target_r = cv.split(target_img)

        histogram_b = Histogram(self.target_b)
        histogram_g = Histogram(self.target_g)
        histogram_r = Histogram(self.target_r)
        target_histogram_b = histogram_b.generate_histogram()
        target_histogram_g = histogram_g.generate_histogram()
        target_histogram_r = histogram_r.generate_histogram()

        plt.bar(range(len(target_histogram_b)), target_histogram_b, width=1, color='b')
        plt.bar(range(len(target_histogram_g)), target_histogram_g, width=1, color='g')
        plt.bar(range(len(target_histogram_r)), target_histogram_r, width=1, color='r')
        plt.title('histogram of test image ' + str(file))
        plt.savefig('../data/histogram_result/' + 'histogram of test image ' + str(file) + '.jpg')
        plt.show()

        target_histogram_b = np.uint8((256 - 1) * np.cumsum(target_histogram_b) + 0.5)
        target_histogram_g = np.uint8((256 - 1) * np.cumsum(target_histogram_g) + 0.5)
        target_histogram_r = np.uint8((256 - 1) * np.cumsum(target_histogram_r) + 0.5)
        self.template2target_b = self.calculate_target_mapping(target_histogram_b, self.histogram_reverse_b)
        self.template2target_g = self.calculate_target_mapping(target_histogram_g, self.histogram_reverse_g)
        self.template2target_r = self.calculate_target_mapping(target_histogram_r, self.histogram_reverse_r)
        return self.generate_new_image_from_template(file)

    @staticmethod
    def calculate_target_mapping(target_histogram, template_histogram):
        target_map = list(target_histogram)
        template_map = list(template_histogram)
        final_map = []
        for i in range(256):
            temp_tar = target_map[i]
            temp_tem = template_map[temp_tar]
            final_map.append(temp_tem)
        final_map = np.array(final_map)
        return final_map

    def generate_new_image_from_template(self, file):
        height, width = np.shape(self.target_b)
        # print(height, width)
        for i in range(height):
            for j in range(width):
                temp_b = self.target_b[i, j]
                self.target_b[i, j] = self.template2target_b[temp_b]
                temp_g = self.target_g[i, j]
                self.target_g[i, j] = self.template2target_g[temp_g]
                temp_r = self.target_r[i, j]
                self.target_r[i, j] = self.template2target_r[temp_r]
        new_img = cv.merge([self.target_b, self.target_g, self.target_r])
        histogram_b = Histogram(self.target_b)
        histogram_g = Histogram(self.target_g)
        histogram_r = Histogram(self.target_r)
        chart_b = histogram_b.generate_histogram()
        chart_g = histogram_g.generate_histogram()
        chart_r = histogram_r.generate_histogram()

        plt.bar(range(len(chart_b)), chart_b, width=1, color='b')
        plt.bar(range(len(chart_g)), chart_g, width=1, color='g')
        plt.bar(range(len(chart_r)), chart_r, width=1, color='r')
        plt.title('histogram specification of new image ' + str(file))
        plt.savefig('../data/histogram_result/' + 'histogram specification of new image ' + str(file) + '.jpg')
        plt.show()

        return new_img

    def generate_histogram(self, file_path, file_name, title):
        plt.bar(range(len(self.histogram_b)), self.histogram_b, width=1, color='b')
        plt.bar(range(len(self.histogram_g)), self.histogram_g, width=1, color='g')
        plt.bar(range(len(self.histogram_r)), self.histogram_r, width=1, color='r')
        plt.title(title)
        plt.savefig(file_path + file_name + '.jpg')
        plt.show()


def main():
    template_img = read_template_img_rgb_from_file()
    hs = HistogramSpecification(template_img)
    test_path = '../data/test/'
    output_path = '../data/histogram_result/'
    img_list = os.listdir(test_path)
    for file in img_list:
        img_path = os.path.join(test_path, file)
        print('open image ' + str(img_path))
        target_img = cv.imread(img_path)
        result_img = hs.generate_new_image(target_img, file)
        print('finish change ' + str(file))
        output_path_file = os.path.join(output_path, file)
        print(output_path_file)
        show_img(result_img, 'histogram specification of image ' + str(file))
        cv.imwrite(output_path_file, result_img)


if __name__ == '__main__':
    main()
