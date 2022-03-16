import numpy as np
import math
import matlab
import matlab.engine


def PSNR(img, after):
    img1 = np.float64(img)
    img2 = np.float64(after)
    mse = np.mean((img1 - img2) ** 2)
    max_ = 255.0
    psnr = 20 * math.log10(max_ / math.sqrt(mse))
    return [mse, psnr]


def SSIM(img, after, eng):
    img1 = np.uint8(img)
    img1 = matlab.uint8(img1.tolist())
    img2 = np.uint8(after)
    img2 = matlab.uint8(img2.tolist())
    [ssim, ssim_map] = eng.ssim(img1, img2, nargout=2)
    return ssim
