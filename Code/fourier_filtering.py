import cv2
import numpy as np

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def save_image(image, path):
    cv2.imwrite(path, image)


def fft_apply(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift

def inverse_fft_apply(dft_shift):
    unshift_fourier = np.fft.ifftshift(dft_shift)
    img_inv = cv2.idft(unshift_fourier)
    magni_image  = cv2.magnitude(img_inv[:, :, 0], img_inv[:, :, 1])
    norm_image = cv2.normalize(magni_image, None, 0, 255, cv2.NORM_MINMAX)
    norm_image = np.uint8(norm_image)
    return norm_image

def apply_lowpass_filter(img_shape, sigma=25):
    rows, cols = img_shape
    row_c, col_c = rows // 2, cols // 2

    x, y = np.arange(cols), np.arange(rows)
    xgrid, ygrid = np.meshgrid(x - col_c, y - row_c)
    dist_squared = xgrid**2 + ygrid**2
    gaussian_filter = np.exp(-dist_squared / (2 * (sigma ** 2)))

    filter = np.zeros((rows, cols, 2), np.float32)
    filter[:, :, 0], filter[:, :, 1]  = gaussian_filter, gaussian_filter
    return filter


if __name__ == '__main__':
    image = load_image('Images/Noisy_image.png')

    dft_shift = fft_apply(image)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    log_magnitude = np.log(1 + magnitude)
    norm_magnitude = cv2.normalize(log_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    norm_magnitude = np.uint8(norm_magnitude)
    save_image(norm_magnitude, 'Images/converted_fourier.png')

    gaussian_filter = apply_lowpass_filter(image.shape, sigma=25)
    filtered_dft = dft_shift * gaussian_filter

    smoothen_image = inverse_fft_apply(filtered_dft)
    save_image(smoothen_image, 'Images/guassian_fourier.png')


