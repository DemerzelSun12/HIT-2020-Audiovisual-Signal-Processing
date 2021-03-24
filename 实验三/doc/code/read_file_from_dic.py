import os
import numpy as np
import cv2 as cv


def read_file_from_dic():
    file_dir_path = '../data/'
    pic_list = []
    file_list = os.listdir(file_dir_path)
    for file in file_list:
        file_path = os.path.join(file_dir_path, file)
        print("open figure " + file_path)
        with open(file_path) as f:
            img = cv.imread(file_path)
        pic_list.append(img)
    return np.asarray(pic_list)


def show_img(title, img):
    cv.namedWindow(title, cv.WINDOW_FREERATIO)
    cv.imshow(title, img)
    cv.waitKey()
    cv.destroyAllWindows()


def write_img_to_file(file_name, img):
    file_path = '../result/' + file_name + '.jpg'
    cv.imwrite(file_path, img)
