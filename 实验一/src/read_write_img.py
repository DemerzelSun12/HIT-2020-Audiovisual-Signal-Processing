import cv2 as cv


def read_source_img_from_file():
    file_path = '../data/lena512color.tiff'
    img = cv.imread(file_path)
    show_img(img, "Source image of lena")
    return img


def read_source_img_with_gray_from_file():
    file_path = '../data/lena512color.tiff'
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    show_img(img, "Source image of lena")
    return img


def read_gauss_noise_img_from_file():
    file_path = '../data/add gauss noise.png'
    img = cv.imread(file_path)
    show_img(img, "Add gauss noise of lena")
    return img


def read_salt_pepper_noise_img_from_file():
    file_path = '../data/add salt pepper noise.png'
    img = cv.imread(file_path)
    show_img(img, "Add salt pepper noise of lena")
    return img


def write_img_to_file(img, file_name):
    file_path = '../data/' + file_name + '.png'
    cv.imwrite(file_path, img)


def show_img(img, title):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def change_color():
    img1 = cv.imread("../data/hitsz.jpg", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("../data/hitszname.jpg", cv.IMREAD_GRAYSCALE)
    write_img_to_file(img1, "hitsz")
    write_img_to_file(img2, "hitszname")


if __name__ == '__main__':
    # read_source_img_from_file()
    # write_img_to_file()
    change_color()
