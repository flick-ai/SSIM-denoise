import filter
import numpy as np
import matlab.engine
import cv2


def Save(img, name, add, PSNR, SSIM, class_name):
    print("{} in {} (PSNR:{} SSIM:{})".format(name, class_name, PSNR, SSIM))
    address = add + "/" + name + '_nlm' + ".jpg"
    cv2.imwrite(address, img)


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
        Save(self.image_nlm, self.name, self.address, self.PSNR, self.SSIM, self.__class__.__name__+'_NLM')


class Grey(NLM):
    def __init__(self, img, name, address, eng=matlab.engine.start_matlab(), noise=filter.gauss):
        super().__init__(img, name, address, eng, noise)
        self.image_ssim = None

    def Process(self):
        super().Process()
        self.image_ssim = filter.SSIM_Grey(self.image_noise, self.image_nlm)
        self.PSNR = filter.PSNR(self.image_ssim, self.image)
        self.SSIM = filter.SSIM(self.image_ssim, self.image, self.eng)
        Save(self.image_ssim, self.name, self.address, self.PSNR, self.SSIM, self.__class__.__name__+'_SSIM')


def Run(layer):
    nlm = filter.NLM(layer)
    ssim = filter.SSIM_Grey(layer, nlm)
    return [nlm, ssim]


class RGB(NLM):
    def __init__(self, img, name, address, eng=matlab.engine.start_matlab(), noise=filter.gauss):
        super().__init__(img, name, address, eng, noise)
        self.image_ssim = None

    def Process(self):
        b, g, r = cv2.split(self.image_noise)
        b_nlm, b_ssim = Run(b)
        g_nlm, g_ssim = Run(g)
        r_nlm, r_ssim = Run(r)
        self.image_nlm = cv2.merge([b_nlm, g_nlm, r_nlm])
        self.PSNR = filter.PSNR(self.image_nlm, self.image)
        self.SSIM = filter.SSIM(self.image_nlm, self.image, self.eng)
        Save(self.image_nlm, self.name, self.address, self.PSNR, self.SSIM, self.__class__.__name__+'_NLM')

        self.image_ssim = cv2.merge([b_ssim, g_ssim, r_ssim])
        self.PSNR = filter.PSNR(self.image_ssim, self.image)
        self.SSIM = filter.SSIM(self.image_ssim, self.image, self.eng)
        Save(self.image_ssim, self.name, self.address, self.PSNR, self.SSIM, self.__class__.__name__+'_SSIM')


class RGB_nature(NLM):
    def __init__(self, img, name, address, eng=matlab.engine.start_matlab(), noise=filter.gauss):
        super().__init__(img, name, address, eng, noise)
        self.image_ssim = None

    def NLM(self):
        b, g, r = cv2.split(self.image_noise)
        b_nlm = filter.NLM(b)
        g_nlm = filter.NLM(g)
        r_nlm = filter.NLM(r)
        self.image_nlm = cv2.merge([b_nlm, g_nlm, r_nlm])

    def Process(self):
        self.NLM()
        self.PSNR = filter.PSNR(self.image_nlm, self.image)
        self.SSIM = filter.SSIM(self.image_nlm, self.image, self.eng)
        Save(self.image_nlm, self.name, self.address, self.PSNR, self.SSIM, self.__class__.__name__ + '_NLM')
        self.image_ssim = filter.SSIM_RGB(self.image_noise, self.image_nlm, 2)
        self.PSNR = filter.PSNR(self.image_ssim, self.image)
        self.SSIM = filter.SSIM(self.image_ssim, self.image, self.eng)
        Save(self.image_ssim, self.name, self.address, self.PSNR, self.SSIM, self.__class__.__name__ + '_SSIM')


class HSV(NLM):
    def __init__(self, img, name, address, eng=matlab.engine.start_matlab(), noise=filter.gauss):
        super().__init__(img, name, address, eng, noise)
        self.image_ssim = None

    def Process(self):
        hsv = cv2.cvtColor(self.image_noise, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s_nlm, s_ssim = Run(s)
        v_nlm, v_ssim = Run(v)
        self.image_nlm = cv2.merge([np.array(h, dtype=float), np.array(s_nlm, dtype=float), np.array(v_nlm, dtype=float)])
        self.image_nlm = cv2.cvtColor(np.uint8(self.image_nlm), cv2.COLOR_HSV2BGR)
        self.PSNR = filter.PSNR(self.image_nlm, self.image)
        self.SSIM = filter.SSIM(self.image_nlm, self.image, self.eng)
        Save(self.image_nlm, self.name, self.address, self.PSNR, self.SSIM, self.__class__.__name__ + '_NLM')

        self.image_ssim = cv2.merge([np.array(h, dtype=float), np.array(s_ssim, dtype=float), np.array(v_ssim, dtype=float)])
        self.image_ssim = cv2.cvtColor(np.uint8(self.image_ssim), cv2.COLOR_HSV2BGR)
        self.PSNR = filter.PSNR(self.image_ssim, self.image)
        self.SSIM = filter.SSIM(self.image_ssim, self.image, self.eng)
        Save(self.image_ssim, self.name, self.address, self.PSNR, self.SSIM, self.__class__.__name__ + '_SSIM')


class HSV_nature(NLM):
    def __init__(self, img, name, address, eng=matlab.engine.start_matlab(), noise=filter.gauss):
        super().__init__(img, name, address, eng, noise)
        self.image_ssim = None

    def NLM(self, hsv):
        h, s, v = cv2.split(hsv)
        s_nlm = filter.NLM(s)
        v_nlm = filter.NLM(v)
        self.image_nlm = cv2.merge([np.array(h, dtype=float), np.array(s_nlm, dtype=float), np.array(v_nlm, dtype=float)])
        return h

    def Process(self):
        hsv = cv2.cvtColor(self.image_noise, cv2.COLOR_BGR2HSV)
        h = self.NLM(hsv)
        self.image_nlm = cv2.cvtColor(np.uint8(self.image_nlm), cv2.COLOR_HSV2BGR)
        self.PSNR = filter.PSNR(self.image_nlm, self.image)
        self.SSIM = filter.SSIM(self.image_nlm, self.image, self.eng)
        Save(self.image_nlm, self.name, self.address, self.PSNR, self.SSIM, self.__class__.__name__ + '_NLM')
        self.image_ssim = filter.SSIM_RGB(self.image_noise, self.image_nlm, 2)
        h_ssim, s_ssim,v_ssim = cv2.split(self.image_ssim)
        self.image_ssim = cv2.merge([np.array(h, dtype=float), np.array(s_ssim, dtype=float), np.array(v_ssim, dtype=float)])
        self.image_ssim = cv2.cvtColor(np.uint8(self.image_ssim), cv2.COLOR_HSV2BGR)
        self.PSNR = filter.PSNR(self.image_ssim, self.image)
        self.SSIM = filter.SSIM(self.image_ssim, self.image, self.eng)
        Save(self.image_ssim, self.name, self.address, self.PSNR, self.SSIM, self.__class__.__name__ + '_SSIM')





