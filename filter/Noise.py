import numpy as np


def gauss(img, mean=0, var=0.0005):
    img = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out = img + noise
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out * 255)
    return out


def salt(img, prob=0.9):
    output = img.copy()
    c, h, w = img.shape
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[prob, (1 - prob) / 2., (1 - prob) / 2.])
    mask = np.repeat(mask, c, axis=0)
    output[mask == 1] = 255
    output[mask == 2] = 0
    return output
