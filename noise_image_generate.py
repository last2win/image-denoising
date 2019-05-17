import numpy as np
from scipy.misc import imsave
import imageio

def im2double(im):
    info = np.iinfo(im.dtype)
    return im.astype(np.double) / info.max


def imwrite(im, filename):
    img = np.copy(im)
    img = img.squeeze()
    if img.dtype == np.double:
        #img = np.array(img*255, dtype=np.uint8)
        img = img * np.iinfo(np.uint8).max
        img = img.astype(np.uint8)
    imageio.imwrite(filename, img)


samples = {'A': 0.6}
for imgName, noiseRatio in samples.items():
    Img = im2double(imageio.imread('{}.png'.format(imgName)))
    Img[(Img == 0)] = 0.01
    rows, cols, channels = Img.shape
    noiseMask = np.ones((rows, cols, channels))
    subNoiseNum = round(noiseRatio * cols)
    for k in range(channels):
        for i in range(rows):
            tmp = np.random.permutation(cols)
            noiseIdx = np.array(tmp[:subNoiseNum])
            noiseMask[i, noiseIdx, k] = 0
    noiseImg = Img * noiseMask
    imageio.imwrite(uri= '{}_noise.png'.format(imgName),im=noiseImg)
