import cv2 as cv


def read_source_img_from_file():
    file_path = '../data/lena512color.tiff'
    img = cv.imread(file_path)
    show_img(img, "Source image of lena")
    return img


def read_source_img_with_gray_from_file():
    file_path = '../data/lenagray.bmp'
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    show_img(img, "Source image of lena")
    return img


def read_template_img_rgb_from_file():
    file_path = '../data/template1.jpg'
    img = cv.imread(file_path)
    show_img(img, "Template image")
    return img


def read_homomorphic_filter():
    file_path = '../data/filter2.tif'
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    show_img(img, 'Source image to filter')
    return img


def write_img_to_file(img, file_name):
    file_path = '../data/' + file_name + '.png'
    cv.imwrite(file_path, img)


def show_img(img, title):
    cv.namedWindow(title, cv.WINDOW_FREERATIO)
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    read_source_img_from_file()
    # write_img_to_file()
