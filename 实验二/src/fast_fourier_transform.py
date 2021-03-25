import numpy as np


def my_fft(img):
    print('start to fft')
    height, width = np.shape(img)
    result_complex = img.astype(np.complex)

    def fft_one(a):
        len = a.size
        if len == 1:
            return
        a0 = np.zeros(len // 2, complex)
        a1 = np.zeros(len // 2, complex)
        for i in range(0, len, 2):
            a0[i // 2] = a[i]
            a1[i // 2] = a[i + 1]
        fft_one(a0)
        fft_one(a1)

        wn = complex(np.cos(2 * np.pi / len), np.sin(2 * np.pi / len))
        w = complex(1, 0)
        for i in range(len // 2):
            t = w * a1[i]
            a[i] = a0[i] + t
            a[i + len // 2] = a0[i] - t
            w = w * wn

    for i in range(height):
        fft_one(result_complex[i])
    for i in range(width):
        fft_one(result_complex[:, i])

    print('finish fft')
    return result_complex


def my_ifft(img):
    print('start to ifft')
    height, width = np.shape(img)

    result_complex = img.astype(np.complex)
    result = np.zeros([height, width], np.float64)

    def ifft_one(a):
        len = a.size
        if len == 1:
            return

        a0 = np.zeros(len // 2, complex)
        a1 = np.zeros(len // 2, complex)
        for i in range(0, len, 2):
            a0[i // 2] = a[i]
            a1[i // 2] = a[i + 1]
        ifft_one(a0)
        ifft_one(a1)

        wn = complex(np.cos(2 * np.pi / len), -1 * np.sin(2 * np.pi / len))  # 参数
        w = complex(1, 0)
        for i in range(len // 2):
            t = w * a1[i]
            a[i] = a0[i] + t
            a[i + len // 2] = a0[i] - t
            w = w * wn

    for i in range(height):
        ifft_one(result_complex[i])
    for i in range(width):
        ifft_one(result_complex[:, i])

    for i in range(height):
        for j in range(width):
            result[i, j] = np.abs(result_complex[i, j] / (height * width))

    print('finish ifft')
    return result
