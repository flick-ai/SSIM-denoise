import cv2
import numpy as np
import matlab
import matlab.engine


def create(Y, X):
    # 预处理函数
    ave_y = np.mean(Y)
    mean_y = np.std(Y)
    ave_x = np.mean(X)
    mean_x = np.std(X)
    out = mean_x / mean_y * (Y - ave_y) + ave_x
    out = out.astype(int)
    return out


def SSIM_Grey(src, sample, f=10, t=1):
    # 滤波函数
    src = create(src, sample)
    H, W = src.shape
    sum_image = np.zeros((H, W), np.uint8)
    sum_weight = np.zeros((H, W), np.uint8)
    pad_length = f + t
    padding = cv2.copyMakeBorder(src, pad_length, pad_length, pad_length, pad_length, cv2.BORDER_REFLECT)
    image = padding[t:t + H + f + 1, t:t + W + f + 1]
    eng = matlab.engine.start_matlab()
    for r in range(-t, t):
        for s in range(-t, t):
            w_image = padding[t + r:t + H + f + r + 1, t + s:t + W + f + s + 1]
            weight = cal(image, w_image, eng)
            sum_image = sum_image + weight[f:f + H, f:f + W] ** 2 * w_image[f:f + H, f:f + W]
            sum_weight = sum_weight + weight[f:f + H, f:f + W] ** 2
    out = sum_image / sum_weight
    return out


def SSIM_RGB(src, sample, i, f=10, t=2):
    src = create(src, sample)
    H, W, C = src.shape
    sum_image = np.zeros((H, W, C), np.uint8)
    sum_weight = np.zeros((H, W, 1), np.uint8)
    pad_length = f + t
    padding = cv2.copyMakeBorder(src, pad_length, pad_length, pad_length, pad_length, cv2.BORDER_REFLECT)
    image = padding[t:t + H + f + 1, t:t + W + f + 1, :]
    eng = matlab.engine.start_matlab()
    for r in range(-t, t):
        for s in range(-t, t):
            w_image = padding[t + r:t + H + f + r + 1, t + s:t + W + f + s + 1, :]
            weight_b = cal(image[:, :, 0], w_image[:, :, 0], eng)
            weight_g = cal(image[:, :, 1], w_image[:, :, 1], eng)
            weight_r = cal(image[:, :, 2], w_image[:, :, 2], eng)
            weight = (weight_b ** i + weight_g ** i + weight_r ** i) ** (1 / i)
            sum_image = sum_image + np.array([weight[f:f + H, f:f + W] ** i]).reshape((512, 512, 1)) * w_image[f:f + H, f:f + W, :]
            sum_weight = sum_weight + np.array([weight[f:f + H, f:f + W] ** i]).reshape((512, 512, 1))
    out = sum_image / sum_weight
    return out


def SSIM_HSV(src, sample, i, f=10, t=2):
    src = create(src, sample)
    H, W, C = src.shape
    sum_image = np.zeros((H, W, C), np.uint8)
    sum_weight = np.zeros((H, W, 1), np.uint8)
    pad_length = f + t
    padding = cv2.copyMakeBorder(src, pad_length, pad_length, pad_length, pad_length, cv2.BORDER_REFLECT)
    image = padding[t:t + H + f + 1, t:t + W + f + 1, :]
    eng = matlab.engine.start_matlab()
    for r in range(-t, t):
        for s in range(-t, t):
            w_image = padding[t + r:t + H + f + r + 1, t + s:t + W + f + s + 1, :]
            weight_s = cal(image[:, :, 1], w_image[:, :, 1], eng)
            weight_v = cal(image[:, :, 2], w_image[:, :, 2], eng)
            weight = (weight_s ** i + weight_v ** i) ** (1 / i)
            sum_image = sum_image + np.array([weight[f:f + H, f:f + W] ** i]).reshape((512, 512, 1)) * w_image[f:f + H, f:f + W, :]
            sum_weight = sum_weight + np.array([weight[f:f + H, f:f + W] ** i]).reshape((512, 512, 1))
    out = sum_image / sum_weight
    return out


def cal(img1, img2, eng):
    # 调用matlab计算SSIM值
    img1 = np.uint8(img1)
    img1 = matlab.uint8(img1.tolist())
    img2 = np.uint8(img2)
    img2 = matlab.uint8(img2.tolist())
    [ssim, ssim_map] = eng.ssim(img1, img2, nargout=2)
    return np.array(ssim_map)
