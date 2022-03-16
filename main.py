import cv2
import Method

# 资料库读取地址
Address = "D:/Pycharm/Graph"

#
address = Address + "/Grey"
lena_grey = cv2.imread(address + "/Barbara.bmp", 0)
Grey1 = Method.Grey(img=lena_grey, name="Barbara", address=address)
Grey1.Process()


