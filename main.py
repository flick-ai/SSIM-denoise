import cv2
import Method

# 资料库读取地址
import filter

Address = "D:/Pycharm/Graph"

# 运行范例
# address = Address + "/Grey"
# Barbara_grey = cv2.imread(address + "/Barbara.bmp", 0)
# noise = filter.gauss(Barbara_grey)
# Grey1 = Method.Grey(img=Barbara_grey, name="Barbara", address=address, noise=noise)
# Grey1.Process()

address = Address + "/Color"
lena_color = cv2.imread(address + "/lena.png")
noise = filter.gauss(lena_color)
# RGB = Method.RGB(img=lena_color, name="lena", address=address, noise=noise)
# RGB.Process()
# RGB_nature = Method.RGB_nature(img=lena_color, name="lena", address=address, noise=noise)
# RGB_nature.Process()
# HSV = Method.HSV(img=lena_color, name="lena", address=address, noise=noise)
# HSV.Process()
HSV_nature = Method.HSV_nature(img=lena_color, name="lena", address=address, noise=noise)
HSV_nature.Process()
