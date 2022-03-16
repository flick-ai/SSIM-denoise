import filter
import numpy as np
import matlab.engine
import cv2


class NLM:
    def __init__(self, img, name, address, eng=matlab.engine.start_matlab(), noise=filter.gauss):
        self.image = img
        self.name = name
        self.address = address
        self.image_noise = noise(img)
        self.image_nlm = None
        self.PSNR = None
        self.SSIM = None
        self.eng = eng

    def Process(self):
        self.image_nlm = filter.NLM(self.image_noise)
        self.PSNR = filter.PSNR(self.image_nlm, self.image)
        self.SSIM = filter.SSIM(self.image_nlm, self.image, self.eng)
        print("{} in {} (PSNR:{} SSIM:{})".format(self.name, "NLM", self.PSNR, self.SSIM))
        address = self.address + "/" + self.name + '_nlm' + ".jpg"
        cv2.imwrite(address, self.image_nlm)


class Grey(NLM):
    def __init__(self, img, name, address, eng=matlab.engine.start_matlab(), noise=filter.gauss):
        super().__init__(img, name, address, eng, noise)
        self.image_ssim = None

    def Process(self):
        super().Process()
        self.image_ssim = filter.SSIM_Grey(self.image_noise, self.image_nlm)
        self.PSNR = filter.PSNR(self.image_ssim, self.image)
        self.SSIM = filter.SSIM(self.image_ssim, self.image, self.eng)
        print("{} in {} (PSNR:{} SSIM:{})".format(self.name, "SSIM", self.PSNR, self.SSIM))
        address = self.address + "/" + self.name + '_ssim' + ".jpg"
        cv2.imwrite(address, self.image_ssim)


class RGB(NLM):
    def __init__(self, img, name, address, eng=matlab.engine.start_matlab(), noise=filter.gauss):
        super().__init__(img, name, address, eng, noise)



