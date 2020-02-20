
import numpy as np

def augment(x, y):
    #online data augmentation
    #flip
    flipType = np.random.randint(0, 4)
    if flipType == 0:
        pass
    elif flipType == 1:
        pass
    elif flipType == 2:
        pass
    elif flipType == 3:
        pass
    #blurring
    blurType = np.random.randint(0, 3)
    if blurType == 0:
        x = averageBlur(x, (3, 8))
    elif blurType == 1:
        x = gaussianBlur(x, (0, 3))
    elif blurType == 2:
        pass
    #dropout
    dropoutType = np.random.randint(0, 2)
    if dropoutType == 0:
        x = dropout(x, 0.1)
    elif dropoutType == 1:
        pass
    #noise
    noiseType = np.random.randint(0, 2)
    if noiseType == 0:
        x = gaussianNoise(x)
    elif noiseType == 1:
        pass
    return x, y

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

#TODO: use vectorized operation to accelerate
def dropout(data, dropout_rate):
    max_randint = int(dropout_rate * 100)
    dst = np.copy(data)
    for k in range(dst.shape[0]):
        mean = np.mean(dst[k, ...])
        for i in range(dst.shape[1]):
            for j in range(dst.shape[2]):
                tmp = np.random.randint(0, max_randint+1)
                if tmp % 10 == 0:
                    dst[k, i, j] = mean
    return dst

def contrastNormalization(data):
    dst = np.copy(data)
    for i in range(dst.shape[0]):
        dst[i] -= np.mean(dst[i])
    return dst

def gaussianNoise(data):
    dst = np.copy(data)
    for i in range(dst.shape[0]):
        dst[i] += np.random.normal(0.0, np.std(dst[i])*0.1, dst[i].shape)
    return dst

def flip(data, label):
    dst_data = np.copy(data)
    dst_label = np.copy(label)
    for i in range(dst_data.shape[0]):
        dst_data[i] = np.fliplr(dst_data[i])
        dst_label[i,:,:,0] = 1.0 - dst_label[i,:,:,0]
        dst_label[i,:,:,0] = np.flip(dst_label[i,:,:,0])
        dst_label[i,:,:,1] = 1.0 - dst_label[i,:,:,1]
        dst_label[i,:,:,1] = np.flip(dst_label[i,:,:,1])

        dst_label[i,:,:,2] = np.flip(dst_label[i,:,:,2])
        dst_label[i,:,:,3] = np.flip(dst_label[i,:,:,3])
        dst_label[i,:,:,4] = np.flip(dst_label[i,:,:,4])

        dst_label[i,:,:,0] *= dst_label[i,:,:,4]
        dst_label[i,:,:,1] *= dst_label[i,:,:,4]
        
    return dst_data, dst_label

#TODO: use vectorized operation to accelerate
def fliplr(data, label):
    dst_data = np.copy(data)
    dst_label = np.copy(label)
    for i in range(dst_data.shape[0]):
        dst_data[i] = np.fliplr(dst_data[i])
        dst_label[i,:,:,0] = 1.0 - dst_label[i,:,:,0]
        dst_label[i,:,:,0] = np.fliplr(dst_label[i,:,:,0])
        dst_label[i,:,:,2] = np.fliplr(dst_label[i,:,:,2])
        dst_label[i,:,:,3] = np.fliplr(dst_label[i,:,:,3])
        dst_label[i,:,:,4] = np.fliplr(dst_label[i,:,:,4])
    return dst_data, dst_label

#TODO: use vectorized operation to accelerate
def flipud(data, label):
    dst_data = np.copy(data)
    dst_label = np.copy(label)
    for i in range(dst_data.shape[0]):
        dst_data[i] = np.flipud(dst_data[i])
        dst_label[i,:,:,1] = 1.0 - dst_label[i,:,:,1]
        dst_label[i,:,:,1] = np.flipud(dst_label[i,:,:,1])
        dst_label[i,:,:,2] = np.flipud(dst_label[i,:,:,2])
        dst_label[i,:,:,3] = np.flipud(dst_label[i,:,:,3])
        dst_label[i,:,:,4] = np.flipud(dst_label[i,:,:,4])
    return dst_data, dst_label

if __name__ == '__main__':
    data = np.random.rand(2,3,3,1)
    label = np.random.rand(2,3,3,5)
    print(data[0,:,:,:])
    data1, label1 = fliplr(data, label)
    print(label[0,:,:,0], label1[0,:,:,0])
    data2, label2 = flipud(data, label)
    print(label[0,:,:,1], label2[0,:,:,1])
    #dst = contrastNormalization(data)
    #print(dst)
    #blur = averageBlur(data, (3,8))
    #blur = gaussianBlur(data, (0,3))
    #blur = dropout(data, dropout_rate=0.1)
    #print(blur.shape)
    #result = np.concatenate((data, blur))
    #print(result.shape)
    pass