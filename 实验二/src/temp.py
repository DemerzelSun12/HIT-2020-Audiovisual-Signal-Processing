import cv2  # For OpenCV modules (For Image I/O and Contour Finding)
import numpy as np  # For general purpose array manipulation
import scipy.fftpack  # For FFT2
from src.read_write_img import *


# Read in image
img = read_homomorphic_filter()

# Number of rows and columns
rows = img.shape[0]
cols = img.shape[1]

# Remove some columns from the beginning and end
img = img[:, 59:cols - 20]

# Number of rows and columns
rows = img.shape[0]
cols = img.shape[1]

# Convert image to 0 to 1, then do log(1 + I)
imgLog = np.log1p(np.array(img, dtype="float") / 255)

# Create Gaussian mask of sigma = 10
M = rows
N = cols
sigma = 10
(X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
centerX = np.ceil(N / 2)
centerY = np.ceil(M / 2)
gaussianNumerator = (X - centerX) ** 2 + (Y - centerY) ** 2

# Low pass and high pass filters
Hlow = np.exp(-gaussianNumerator / (2 * sigma * sigma))
Hhigh = 1 - Hlow

# Move origin of filters so that it's at the top left corner to
# match with the input image
HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

# Filter the image and crop
If = scipy.fftpack.fft2(imgLog.copy(), (M, N))
show_img(np.real(If), "sddg")
Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M, N)))
show_img(Ioutlow, "sdf")
Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M, N)))
show_img(Iouthigh, "sdf2")

# Set scaling factors and add
gamma1 = 0.3
gamma2 = 1.5
Iout = gamma1 * Ioutlow[0:rows, 0:cols] + gamma2 * Iouthigh[0:rows, 0:cols]
show_img(Iout, "img_out")

# Anti-log then rescale to [0,1]
Ihmf = np.expm1(Iout)
show_img(Ihmf, "Ihmf")
# print("min img_hmf {}, max img_hmf {}", np.min(Ihmf), np.max(Ihmf))
Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
Ihmf2 = np.array(255 * Ihmf, dtype="uint8")
show_img(Ihmf2, "img_result")

# Show all images
cv2.imshow('Original Image', img)
cv2.imshow('Homomorphic Filtered Result', Ihmf2)
write_img_to_file(Ihmf2, "homomorphic_filter_result")
write_img_to_file(img, "filter")
cv2.waitKey(0)
cv2.destroyAllWindows()
