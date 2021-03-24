import numpy as np
from src.read_write_img import *


def add_noise():
    img = read_source_img_from_file()
    gauss_noise_img = add_gauss_noise(img, 0, 0.01)
    write_img_to_file(gauss_noise_img, 'add gauss noise')
    show_img(gauss_noise_img, 'add gauss noise')
    salt_pepper_noise_img = add_salt_pepper_noise(img, 0.01)
    write_img_to_file(salt_pepper_noise_img, 'add salt pepper noise')
    show_img(salt_pepper_noise_img, 'add salt pepper noise')


def add_gauss_noise(img, mean, variance):
    img = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, variance ** 0.5, img.shape)
    out = img + noise
    out = np.clip(out, 0.0, 1.0)
    out = np.uint8(out * 255)
    return out


def add_salt_pepper_noise(img, proportion):
    length, width = img.shape[0], img.shape[1]
    out = img
    noise_number = int(length * width * proportion)
    print(noise_number)
    for i in range(noise_number):
        x = np.random.randint(0, length)
        y = np.random.randint(0, width)
        if np.random.randint(0, 2) == 0:
            out[x, y] = 0
        else:
            out[y, x] = 255
    return out


if __name__ == '__main__':
    add_noise()
