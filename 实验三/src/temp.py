import cv2 as cv
import numpy as np
from src.read_file_from_dic import *


# 二值化
def binaryzation(img):
    maxi = float(img.max())
    mini = float(img.min())
    x = maxi - ((maxi - mini) / 2)
    ret, thresh = cv.threshold(img, x, 255, cv.THRESH_BINARY)
    return thresh


# 找到能够包围给定区块的最小矩形的左下角坐标(min(x),min(y))与右上角坐标(max(x),max(y))
def find_rectangle(contour):
    x = []
    y = []
    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])
    return [min(y), min(x), max(y), max(x)]


# 读取图片
img = cv.imread('../data/1.jpg')
m = 400 * img.shape[0] / img.shape[1]
# 调整图像尺寸
img = cv.resize(img, (400, int(m)), interpolation=cv.INTER_AREA)
# 转化为灰度图
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 结构元素(kernel)为半径为16的圆
r = 16
h = w = r * 2 + 1
kernel = np.zeros((h, w), np.uint8)
cv.circle(kernel, (r, r), r, 1, -1)
# 开运算
open_img = cv.morphologyEx(gray_img, cv.MORPH_OPEN, kernel)
# 顶帽
hat_img = cv.absdiff(gray_img, open_img)
# 图像二值化
binary_img = binaryzation(hat_img)
# canny边缘检测
canny = cv.Canny(binary_img, binary_img.shape[0], binary_img.shape[1])
# 闭运算核
kernel = np.ones((5, 19), np.uint8)
# 闭运算
close_img = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
# 开运算
open_img = cv.morphologyEx(close_img, cv.MORPH_OPEN, kernel)
# 更换结构元素
kernel = np.ones((11, 5), np.uint8)
# 开运算
open_img = cv.morphologyEx(open_img, cv.MORPH_OPEN, kernel)

show_img("open img", open_img)

# 提取轮廓
contours, hierarchy = cv.findContours(open_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# 存储每个块的矩形坐标
block = []
for c in contours:
    r = find_rectangle(c)
    block.append(r)

max_weight = 0
max_index = -1
# 遍历分割出来的各个区域
for i in range(len(block)):
    b = img[block[i][1]: block[i][3], block[i][0]: block[i][2]]
    # 转化为hsv颜色空间
    hsv = cv.cvtColor(b, cv.COLOR_BGR2HSV)
    # 蓝色下界
    lower = np.array([100, 50, 50])
    # 蓝色上界
    upper = np.array([140, 255, 255])
    # 利用掩膜进行筛选
    mask = cv.inRange(hsv, lower, upper)
    # 计算当前区域的满足情况
    w1 = 0
    for m in mask:
        w1 += m / 255
    w2 = 0
    for n in w1:
        # print(n)
        w2 += n
    if w2 > max_weight:
        max_index = i
        max_weight = w2
# 最可能为车牌的区域对应的矩形坐标
rect = block[max_index]
# 画框
cv.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
cv.imshow('ovo', img)
cv.waitKey(0)
cv.destroyAllWindows()
