from __future__ import print_function
from PIL import Image
import numpy as np


def save(arr, path):
    # save an image
    newarr = 0.+arr
    newarr -= min(map(min, arr))
    newarr *= 255. / (max(map(max, arr)))
    Image.fromarray(newarr).convert('L').save(path)


# load input file into numpy array
orig = Image.open("original.png")
orig = np.array(orig)
save(orig, "out/1_input.png")

# load target file into numpy array
targ = Image.open("target.png")
targ = np.array(targ)
save(targ, "out/2_target.png")

# FFT original
origft = np.fft.fft2(orig)
origft = np.fft.fftshift(origft)
save(np.log(np.abs(origft)), "out/3_original_ft.png")
origft = np.fft.ifftshift(origft)

# FFT target
targft = np.fft.fft2(targ)
targft = np.fft.fftshift(targft)
save(np.log(np.abs(targft)), "out/4_target_ft.png")
targft = np.fft.ifftshift(targft)

# divide target/origin
maskft = targft / origft
maskft = np.fft.fftshift(maskft)
save(np.log(np.abs(maskft)), "out/5_mask_ft.png")
maskft = np.fft.ifftshift(maskft)
