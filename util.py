import cv2
import numpy as np

def averageBlur(data, boxSize):
    dst = np.zeros_like(data)
    for i in range(np.shape(data)[0]):
        size = np.random.randint(boxSize[0], boxSize[1])
        dst[i, ...] = cv2.boxFilter(
            data[i, ...],
            ddepth=-1,
            ksize=size,
            normalize=False
        )
    return dst
    

def gaussianBlur(data, sigma):
    pass

def dropout(data, dropout_rate):
    pass

def flip(data, label):
    pass

