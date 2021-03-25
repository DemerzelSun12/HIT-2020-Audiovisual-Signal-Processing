import matplotlib.pyplot as plt
from src.read_write_img import *
from src.fast_fourier_transform import *


class HomomorphicFilter(object):
    def __init__(self, img, a=1.0, b=1.5):
        self.img = img
        self.fft_img = None
        self.filter_result = None
        self.ifft_img = None
        self.a = float(a)
        self.b = float(b)
        self.height, self.width = np.shape(img)

    def homomorphic_filter(self):
        self.generate_fft_img()
        print("finish fft image")
        result = self.gaussian_filter()
        print("finish filter image")
        return result

    def gaussian_filter(self):
        M = self.height
        N = self.width
        sigma = 10
        (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
        center_x = np.ceil(N / 2)
        center_y = np.ceil(M / 2)
        gauss_filter_matrix = np.square(X - center_x) + np.square(Y - center_y)

        loss_pass = np.exp(-gauss_filter_matrix / (2 * sigma * sigma))
        high_pass = 1 - loss_pass
        loss_pass_shift = np.fft.ifftshift(loss_pass.copy())
        # show_img(loss_pass_shift, "sdgdf")
        high_pass_shift = np.fft.ifftshift(high_pass.copy())

        img_out_low = np.real(my_ifft(self.fft_img.copy() * loss_pass_shift))
        # show_img(img_out_low, "sdf")
        img_out_high = np.real(my_ifft(self.fft_img.copy() * high_pass_shift))
        # show_img(img_out_high, "sdf2")

        gamma1 = 0.3
        gamma2 = 1.5

        img_out = gamma1 * img_out_low[0:self.height, 0:self.width] + gamma2 * img_out_high[0:self.height, 0:self.width]

        self.filter_result = img_out
        # show_img(img_out, "img_out")

        img_hmf = np.expm1(img_out)
        # show_img(img_hmf, "Ihmf")
        print("min img_hmf {}, max img_hmf {}", np.min(img_hmf), np.max(img_hmf))
        img_hmf = (img_hmf - (-354) * np.min(img_hmf)) / (0.915 * np.max(img_hmf) - (-354) * np.min(img_hmf))
        img_result = np.array(255 * img_hmf, dtype="uint8")
        # show_img(img_result, "img_result")
        return img_result

    def generate_fft_img(self):
        img_log = np.log1p(np.array(self.img, dtype="float") / 255)
        self.fft_img = my_fft(img_log)
        show_img(np.real(self.fft_img), "sdfsdfd")


def main():
    img = read_homomorphic_filter()
    # img = cv.resize(img, (512, 512), interpolation=cv.INTER_NEAREST)
    my_fft = HomomorphicFilter(img)
    result = my_fft.homomorphic_filter()
    show_img(result, 'homomorphic filter')
    write_img_to_file(result, 'filter_result/' + 'homomorphic_filter_result')


if __name__ == '__main__':
    main()
