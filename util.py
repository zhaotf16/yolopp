
import numpy as np

def averageFilter(data, size):
    input_copy = np.copy(data)
    filter_template = np.ones((size, size), dtype=np.float32)
    pad = int((size - 1) / 2)
    input_copy = np.pad(input_copy, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=0)
    input_copy = input_copy[..., 0]
    output = np.copy(input_copy)
    m, n = input_copy.shape[0], input_copy.shape[1]

    for i in range(pad, m - pad):
        for j in range(pad, n - pad):
            output[i, j] = np.sum(
                filter_template * input_copy[i - pad:i + pad + 1, j - pad:j + pad + 1]
            ) / (size ** 2)

    output = output[pad:m-pad, pad:n-pad]
    output = np.expand_dims(output, axis=-1)
    
    return output

def averageBlur(data, boxSize):
    dst = np.zeros_like(data)
    for i in range(np.shape(data)[0]):
        size = np.random.randint(boxSize[0]//2, boxSize[1]//2)
        size = 2 * size + 1
        image = data[i, ...]
        dst[i, ...] = averageFilter(image, size)
    return dst

def gaussianFilter(data, sigma, size):
    input_copy = np.copy(data)
    filter_template = np.zeros((size, size), dtype=np.float32)
    mid = (size - 1) / 2
    sigma2 = sigma ** 2
    for i in range(size):
        for j in range(size):
            x = -((i-mid)**2 + (j-mid)**2) / 2 / sigma2
            filter_template[i, j] = np.math.exp(x) / 2 / 3.14159 / sigma2  

    pad = int((size - 1) / 2)
    input_copy = np.pad(input_copy, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=0)
    input_copy = input_copy[..., 0]
    output = np.copy(input_copy)
    m, n = input_copy.shape[0], input_copy.shape[1]

    for i in range(pad, m - pad):
        for j in range(pad, n - pad):
            output[i, j] = np.sum(
                filter_template * input_copy[i - pad:i + pad + 1, j - pad:j + pad + 1]
            ) / (size ** 2)

    output = output[pad:m-pad, pad:n-pad]
    output = np.expand_dims(output, axis=-1)

    return output

def gaussianBlur(data, sigma):
    dst = np.zeros_like(data)
    for i in range(np.shape(data)[0]):
        sig = sigma[0] + (sigma[1] - sigma[0]) * np.random.rand()
        size = int(3 * sig) // 2 * 2 + 1
        image = data[i, ...]
        dst[i, ...] = gaussianFilter(image, sig, size)
    return dst

def dropout(data, dropout_rate):
    max_randint = int(dropout_rate * 100)
    dst = np.copy(data)
    for k in range(dst.shape[0]):
        mean = np.mean(dst[k, ...])
        for i in range(dst.shape[1]):
            for j in range(dst.shape[2]):
                tmp = np.random.randint(0,11)
                if tmp % 10 == 0:
                    dst[k, i, j] = mean
    return dst


def flip(data, label):
    pass

if __name__ == '__main__':
    data = np.random.rand(2,1024,1024,1)
    #blur = averageBlur(data, (3,8))
    #blur = gaussianBlur(data, (0,3))
    blur = dropout(data, dropout_rate=0.1)
    print(blur.shape)
    result = np.concatenate((data, blur))
    print(result.shape)