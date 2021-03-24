import cv2 as cv
import numpy as np

if __name__ == '__main__':
    img = cv.imread("../1.jpg", cv.IMREAD_GRAYSCALE)
    cv.imshow("wer", img)
    cv.waitKey()
    img_m = np.asarray(img)
    height, width = np.shape(img_m)
    img2 = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            if img[i][j] < 125:
                img2[i][j] = 0
            else:
                img2[i][j] = 255
    cv.imwrite("2.jpg", img2)
