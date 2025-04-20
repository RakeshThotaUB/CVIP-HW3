import cv2
import numpy as np

def load_image(path):
    return cv2.imread(path)

def save_image(image, path):
    cv2.imwrite(path, image)

def convolve(image, filter):
    padded_img = np.pad(image, filter.shape[0] // 2, mode='constant', constant_values=0)
    padded_img = padded_img.astype(np.float32)
    convolve_img = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_img[i:i + filter.shape[0], j: j + filter.shape[1]]
            convolve_img[i, j] = np.sum(region * filter)
    
    convolve_img = np.clip(convolve_img, 0, 255).astype(np.uint8)
    return convolve_img

def median_filter(image, filter_size = 5):
    pad_size = filter_size // 2
    padded_img = np.pad(image, pad_size, mode='constant', constant_values=0)
    padded_img = padded_img.astype(np.float32)
    median_img = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_img[i:i + filter_size, j:j + filter_size]
            median_img[i, j] = np.median(region)

    median_img = np.clip(median_img, 0, 255).astype(np.uint8)
    return median_img

def contrast_brightness_adjust(image):
    alpha = 1.5
    beta = 50
    output = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    return output


if __name__ == "__main__":
    
    noisy_image = cv2.imread("Images/Noisy_image.png", cv2.IMREAD_GRAYSCALE)
    unexposed_image = load_image("Images/Uexposed.png") 
    
    convolution_filter = np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]], dtype=np.float32) / 9.0
    convolved_image = convolve(noisy_image, convolution_filter)
    save_image(convolved_image, "Images/convolved_image.png")
    
    averaging_filter = np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]], dtype=np.float32) / 9.0
    average_image = convolve(noisy_image, averaging_filter)
    save_image(average_image, "Images/average_image.png")
    
    gaussian_filter = np.array([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]], dtype=np.float32) / 16.0
    gaussian_image = convolve(noisy_image, gaussian_filter)
    save_image(gaussian_image, "Images/gaussian_image.png")
    
    median_image = median_filter(noisy_image, filter_size = 5)
    save_image(median_image, "Images/median_image.png")
    
    adjusted_image = contrast_brightness_adjust(unexposed_image)
    save_image(adjusted_image, "Images/adjusted_image.png")