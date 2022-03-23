import cv2
import matlab.engine
import Method
import filter

# 资料库读取地址
Address = "D:/Pycharm/Graph"
address = Address + "/Color"
lena_color = cv2.imread(address + "/lena.png")
eng = matlab.engine.start_matlab()
for i in [0.002, 0.003, 0.004, 0.006, 0.007, 0.008, 0.009]:
    print(i)
    noise = filter.gauss(lena_color, 0, i)
    RGB = Method.RGB(img=lena_color, name="lena", address=address, noise=noise, eng=eng)
    RGB.Process()
    RGB_nature = Method.RGB_nature(img=lena_color, name="lena", address=address, noise=noise, eng=eng)
    RGB_nature.Process()
    HSV = Method.HSV(img=lena_color, name="lena", address=address, noise=noise, eng=eng)
    HSV.Process()
