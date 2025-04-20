import cv2
import numpy as np
import math

def load_image(path):
    return cv2.imread(path)  

def save_image(image, path):
    cv2.imwrite(path, image)

def rgb_to_hsv1(image):
    image = image.astype(np.float32) / 255.0  
    R, G, B = image[:, :, 2], image[:, :, 1], image[:, :, 0]  

    V = np.maximum(np.maximum(R, G), B)
    S = np.where(V != 0, (V - np.minimum(np.minimum(R, G), B)) / V, 0)
    H = np.zeros_like(V)

    mask = ((V - np.minimum(np.minimum(R, G), B)) != 0)
    
    H[(V == R) & mask] = (60 * (G[(V == R) & mask] - B[(V == R) & mask]) / (V - np.minimum(np.minimum(R, G), B))[(V == R) & mask]) % 360
    H[(V == G) & mask] = (120 + 60 * (B[(V == G) & mask] - R[(V == G) & mask]) / (V - np.minimum(np.minimum(R, G), B))[(V == G) & mask]) % 360
    H[(V == B) & mask] = (240 + 60 * (R[(V == B) & mask] - G[(V == B) & mask]) / (V - np.minimum(np.minimum(R, G), B))[(V == B) & mask]) % 360

    hsv1_img = np.stack([H / 2, S * 255, V * 255], axis=-1).astype(np.uint8)
    return hsv1_img


def rgb_to_hsv2(image):
    image = image.astype(np.float32) / 255.0
    R, G, B = image[:, :, 2], image[:, :, 1], image[:, :, 0]

    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    theta = np.arccos(num / den)

    H = np.where(B <= G, theta, 360 - theta)
    H[np.isnan(H)] = 0

    min_rgb = np.minimum(np.minimum(R, G), B)
    V = np.maximum(np.maximum(R, G), B)

    # In HW3, it is asked for RBG to HSV
    # If I follow the class slides it is mentioned as HSI (I tends to average(r, g, b))
    # S = 1 - (3 * min_rgb / (R+G+B)) 

    # If question really wants to calculate HSV as mentioned (V tends to max(r, g, b))
    # S = 1 - (min_rgb / max(R, G, B))  
    # But question pdf mentions to use formula in the slides
    S = 1 - (3 * min_rgb / (R+G+B)) 
    S[(R + G + B) == 0] = 0  

    hsv2_img = (cv2.merge([H, S, V]) * 255).astype(np.uint8)
    return hsv2_img

def rgb_to_cmyk(image):
    image = image.astype(np.float32) / 255.0
    R, G, B = image[:, :, 2], image[:, :, 1], image[:, :, 0]

    C = 1 - R
    M = 1 - G
    Y = 1 - B
    K = np.minimum(np.minimum(C, M), Y)

    C[K == 1] = 0
    M[K == 1] = 0
    Y[K == 1] = 0
    C = (C - K) / (1 - K)
    M = (M - K) / (1 - K)
    Y = (Y - K) / (1 - K)

    cmyk_img = (cv2.merge([C, M, Y, K]) * 255).astype(np.uint8)
    return cmyk_img

def f(t):
    return np.where(t > 0.008856, t ** (1/3), (7.787 * t) + (16 / 116))

def rgb_to_lab(image):
    image = image.astype(np.float32) / 255.0
    R, G, B = image[..., 2], image[..., 1], image[..., 0]
   
    # Calculate XYZ from RGB 
    X = R * 0.412453 + G * 0.357580 + B * 0.180423
    Y = R * 0.212671 + G * 0.715160 + B * 0.072169
    Z = R * 0.019334 + G * 0.119193 + B * 0.950227
    Xn, Yn, Zn = 0.950456, 1.0, 1.088754
    X = X / Xn
    Y = Y / Yn
    Z = Z / Zn
    fX = f(X)
    fY = f(Y)
    fZ = f(Z)

    L = np.where(Y > 0.008856, (116 * fY - 16), 903.3 * Y)
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)

    L8bit = (L * 255 / 100).clip(0, 255)
    a8bit = (a + 128).clip(0, 255)
    b8bit = (b + 128).clip(0, 255)
    lab_image = np.stack([L8bit, a8bit, b8bit], axis=-1).astype(np.uint8)
    return lab_image


if __name__ == "__main__":
    img = load_image("Images/Lenna.png")

    hsv1 = rgb_to_hsv1(img)
    save_image(hsv1, "Images/hsv_image_1.png")

    hsv2 = rgb_to_hsv2(img)
    save_image(hsv2, "Images/hsv_image_2.png")

    cmyk = rgb_to_cmyk(img)
    save_image(cmyk, "Images/cmyk_image.png")

    lab = rgb_to_lab(img)
    save_image(lab, "Images/lab_image.png")
