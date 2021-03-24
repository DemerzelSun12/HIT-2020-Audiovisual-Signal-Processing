import numpy as np
import cv2 as cv
from src.read_file_from_dic import *


class LocatePlate(object):
    def __init__(self, img):
        self.img = img
        self.open_img = None
        self.hat_img = None

    def binaryzation(self):
        maxi = float(self.hat_img.max())
        mini = float(self.hat_img.min())
        x = maxi - ((maxi - mini) / 2)
        _, thresh = cv.threshold(self.hat_img, x, 255, cv.THRESH_BINARY)
        return thresh

    @staticmethod
    def find_rectangle(contour):
        x = []
        y = []
        for p in contour:
            y.append(p[0][0])
            x.append(p[0][1])
        return [min(x), max(x), min(y), max(y)]

    def filter_img(self):
        mask = 400 * self.img.shape[0] / self.img.shape[1]
        self.img = cv.resize(self.img, (400, int(mask)), interpolation=cv.INTER_AREA)
        gray_img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        radius = 16
        height = width = radius * 2 + 1
        kernel = np.zeros((height, width), np.uint8)
        cv.circle(kernel, (radius, radius), radius, 1, -1)
        # 开运算
        open_img = cv.morphologyEx(gray_img, cv.MORPH_OPEN, kernel)
        self.open_img = open_img
        # show_img("open img", self.open_img)
        # 顶帽变换
        hat_img = cv.absdiff(gray_img, self.open_img)
        # show_img("hat img", hat_img)
        self.hat_img = hat_img
        # 二值化
        binary_img = self.binaryzation()
        # Canny边缘检测
        canny = cv.Canny(binary_img, binary_img.shape[0], binary_img.shape[1])
        # show_img("canny", canny)
        kernel = np.ones((5, 19), np.uint8)
        # 闭运算
        close_img = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
        # show_img("close img", close_img)

        # 开运算
        open_img = cv.morphologyEx(close_img, cv.MORPH_OPEN, kernel)
        # show_img("open img", open_img)

        # 开运算
        kernel = np.ones((14, 6), np.uint8)
        open_img = cv.morphologyEx(open_img, cv.MORPH_OPEN, kernel)
        self.open_img = open_img
        # show_img("open img", self.open_img)

    def detect_edge(self):
        contours, _ = cv.findContours(self.open_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        block = []
        for c in contours:
            r = self.find_rectangle(c)
            block.append(r)
        max_weight = 0
        max_index = -1
        for i in range(len(block)):
            b = self.img[block[i][0]: block[i][1], block[i][2]: block[i][3]]
            # 转化为hsv颜色空间
            hsv_img = cv.cvtColor(b, cv.COLOR_BGR2HSV)
            # 蓝色下界
            lower = np.array([100, 50, 50])
            # 蓝色上界
            upper = np.array([140, 255, 255])
            mask = cv.inRange(hsv_img, lower, upper)
            w1 = 0
            for m in mask:
                w1 += m / 255
            w2 = 0
            for n in w1:
                w2 += n
            if w2 > max_weight:
                max_index = i
                max_weight = w2
        # print(block)
        # print(max_index)
        if max_index == -1:
            return self.img
        rect = block[max_index]
        img = self.img.copy()
        cv.rectangle(img, (rect[3], rect[1]), (rect[2], rect[0]), (0, 0, 255), 2)
        # show_img("img", img)
        return img

    def locate_plate_main(self):
        self.filter_img()
        img = self.detect_edge()
        return img


def main():
    img_list = read_file_from_dic()
    for i in range(img_list.size):
        # show_img("i", img_list[i])
        print("detect img " + str(i + 1))
        locate = LocatePlate(img_list[i])
        img = locate.locate_plate_main()
        write_img_to_file(str(i + 1), img)


if __name__ == '__main__':
    main()
