# SSIM-denoise
### 使用方法：
1.下载好图片库包\
2.打开main函数，修改Address参数为图片库包所在的绝对路径\
3.修改图片参数为你想要的图片信息对应的参数（可以添加新的）\
4.运行main函数，在你运行的图片所在文件夹会出现对应的图片结果，终端将显示对应的MSE、PSNR和SSIM值信息。
### 类的调用：
在Method文件中给出了可以使用的类，六种不同的彩色图像滤波器，分别是：\
NLM滤波，SSIM滤波，SSIM-nature滤波，HSV-NLM滤波 ，HSV-SSIM滤波和HSV-SSIM-nature滤波。 他们均具有5个参数：\
img:需要去噪的原图\
name：图片名称\
address：图片所在路径\
eng：matlab引擎，已经给出默认值\
noise：噪声添加方式，默认为均值为0，方差为0.0005的高斯噪声
###噪声实现：
在filter/Noise文件在文件中给出了高斯噪声、泊松噪声和椒盐噪声的实现：
### 评价指标：
在filter/Measure文件中实现了PSNR指标和SSIM指标的计算实现。\
### 其他文件：
在filter/Measure中还保存了NLM滤波器、SSIM评价指标和我们开发的SSIM滤波器具体实现代码