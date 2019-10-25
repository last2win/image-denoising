# -*- coding:utf-8 -*-
import numpy as np
import imageio
from sklearn.linear_model import LinearRegression


def im2double(im):
    info = np.iinfo(im.dtype)
    return im.astype(np.double) / info.max


def write(im, filename):
    img = np.copy(im)
    img = img.squeeze()
    if img.dtype == np.double:
        #img = np.array(img*255, dtype=np.uint8)
        img = img * np.iinfo(np.uint8).max
        img = img.astype(np.uint8)
    imageio.imwrite(filename, img)

# get the noise mask of corrImg


def getNoiseMask(corrImg):
    return np.array(corrImg != 0, dtype='double')


def restore(img, filename):
    radius = 4
    resImg = np.copy(img)
    noiseMask = getNoiseMask(img)
    rows, cols, channel = img.shape
    count = 0
    for row in range(rows):
        for col in range(cols):
            for chan in range(channel):
                if noiseMask[row, col, chan] != 0.:
                    continue
                x_train = []
                y_train = []
                for i in range(row-radius, row+radius):
                    if i < 0 or i >= row:
                        continue
                    for j in range(col-radius, col+radius):
                        if j < 0 or j >= col:
                            continue
                        elif noiseMask[i, j, chan] == 0.:
                            continue
                        elif i == row and j == col:
                            continue
                        x_train.append([i, j])
                        y_train.append([img[i, j, chan]])
                if x_train == []:
                    continue
                Regression = LinearRegression()
                Regression.fit(x_train, y_train)
                resImg[row, col, chan] = Regression.predict([[row, col]])
            count += 1
            if count % 50000 == 0:
                print(filename+".png restored:" +
                      str(float(count)/rows/cols))
    print(filename+".png restore finish!")
    return resImg


if __name__ == '__main__':
    queue = ['A']
    for img_name in queue:
        img = im2double(imageio.imread(img_name+'_noise.png'))
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        resImg = restore(img, img_name)
        write(resImg, img_name+'_recover.png')
