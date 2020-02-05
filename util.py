
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

def gaussianBlur(data, sigma):
    dst = np.zeros_like(data)
    pass

def dropout(data, dropout_rate):
    pass

def flip(data, label):
    pass

if __name__ == '__main__':
    data = np.random.rand(2,1024,1024,1)
    blur = averageBlur(data, (3,8))
    print(blur.shape)
    result = np.concatenate((data, blur))
    print(result.shape)