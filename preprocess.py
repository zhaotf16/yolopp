import os
import mrcHelper
import starHelper
import numpy as np

class MSData():
    def __init__(self, mrc_data, star_data):
        self.mrc_data = mrc_data
        self.star_data = star_data

def downsample(inputs, use_factor=False, para1=None, para2=None):
    #This method executes a downsampling on the mrc
    #Inputs is a list of MrcData
    #If use_factor is True, para1 represents factor and para2 represents shape
    #Else para1 and para2 represents the target size(y, x)
    
    for i in range(len(inputs)):
        if use_factor:
            print("Prcocessing %s ..." % ( inputs[i].name))
            inputs[i].data = mrcHelper.downsample_with_factor(
                inputs[i].data,
                factor=para1,
                shape=para2
            )
            #TODO: fix labels for the downsampled micrographs
        else:
            print("Prcocessing %s ..." % (inputs[i].name))
            inputs[i].data = mrcHelper.downsample_with_size(
                inputs[i].data,
                size1=para1,
                size2=para2
            )
            # TODO: fix labels for the downsampled micrographs

    return inputs

def star2label(inputs, image_size, grid_size=64, particle_size=220):
    #inputs is a list of StarData
    label = np.zeros(
        (len(inputs), grid_size, grid_size, 5), 
        dtype=np.float32
    )
    grid_scale = image_size // grid_size
    for i in range(len(inputs)):
        for coord in inputs[i].content:
            x_index = int(coord[0] - 1) // grid_scale
            y_index = int(coord[1] - 1) // grid_scale
            label[i, x_index, y_index, 0] = coord[0] / image_size
            label[i, x_index, y_index, 1] = coord[1] / image_size
            label[i, x_index, y_index, 2] = particle_size[0] / image_size
            label[i, x_index, y_index, 3] = particle_size[1] / image_size
            label[i, x_index, y_index, 4] = 1.0
    return label

def mrc2array(inputs, image_size):
    #inputs is a list of MrcData
    array = np.zeros(
        (len(inputs), image_size, image_size, 1),
        dtype=np.float32
    )
    for i in range(len(inputs)):
        array[i,...] = np.expand_dims(inputs[i].data.astype(np.float32), axis=-1)
    return array

if __name__ == '__main__':
    #This is a test on eml1/user/ztf
    path = "../data/EMPIAR-10025/rawdata/micrographs"
    #path = "../stack_0001_DW"
    dst = "../dataset/EMPIAR-10025/processed/micrographs"
    dst1 = "../dataset/EMPIAR-10025/processed/labels"
    data = mrcHelper.load_mrc_file(path)
    label = starHelper.read_all_star("../dataset/EMPIAR-10025/rawdata/label_for_training")
    #downsampled_data = preprocess(data, False, para1=1024, para2=1024)
    data = downsample(data, False, para1=1024, para2=1024)
    downsampled_label = []
    for i in range(len(label)):
        name = label[i].name
        content = starHelper.downsample_with_size(
            label[i].content,
            (1024 / 7420, 1024 / 7676)
        )
        downsampled_label.append(starHelper.StarData(name, content))
    
    mrcHelper.write_mrc(data, dst=dst)
    starHelper.write_star(downsampled_label, dst=dst1)