from __future__ import print_function
from PIL import Image
import PIL.ImageOps
import PIL.ImageEnhance
import numpy as np
from scipy.signal import correlate2d

# load input file into numpy array
im = Image.open("data.tif")
imarray = np.array(im) / 2.
im = Image.fromarray(imarray)
im.convert('L').save("out/1_input.png")

# high pass filtering in fourier domain (edge detection)
# filter is a rectangle, while the zero mode is kept for overall brightness
ftim = np.fft.fft2(imarray)
size = 50
dc = ftim[0, 0]
ftim[0:size, 0:size] = 0
ftim[-size:, -size:] = 0
ftim[-size:, 0:size] = 0
ftim[0:size, -size:] = 0
ftim[0, 0] = dc*0.8
ftim = np.fft.fftshift(ftim)
Image.fromarray(np.abs(ftim)/1000.).convert('L').save("out/6_fourierspace.png")
ftim = np.fft.ifftshift(ftim)
im = np.real(np.fft.ifft2(ftim))
Image.fromarray(im).convert('L').save("out/2_edge_detection.png")

# normalize image
# im -= np.mean(im)
# im /= max(map(max, im)) - min(map(min, im))
# im = (im + 1) * 128
# Image.fromarray(im).convert('L').save("out/2_edge_detection_n.png")

# invert image
inv = 256 - im
inv -= np.mean(inv)
inv = np.maximum(0, inv)
Image.fromarray(inv*5).convert('L').save("out/3_inverted.png")

# cross correlate with L-shapes
corr = 0
for angle in np.linspace(0, 360, 100):
    print("{:5.1f} %".format(angle / 3.6))
    shape = Image.open("houses/L11.png").rotate(angle).convert('L')
    # scale = 1
    # shape = shape.resize((int(shape.size[0]*scale), int(shape.size[1]*scale)))
    shape = np.array(shape)
    _corr = correlate2d(inv, shape, mode="same")
    corr = np.maximum(corr, _corr)
    # _corr = corr - np.mean(corr[30:-30])
    # _corr *= 2*255 / max(map(max, _corr))
    # Image.fromarray(_corr).convert('L').save("out/4_xcorrelation.png")

# normalization and output
corr -= np.mean(corr[30:-30])
corr *= 2*255 / max(map(max, corr))
Image.fromarray(corr).convert('L').save("out/4_xcorrelation.png")


# thresholding
threshold = 0.33
corr = np.maximum(0, corr - 255 * threshold) / (1-threshold)
Image.fromarray(corr).convert('L').save("out/5_threshold.png")
