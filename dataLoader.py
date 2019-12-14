import os
import mrcHelper
import starHelper
import numpy as np

class MSData():
    def __init__(self, mrc_data, star_data):
        self.mrc_data = mrc_data
        self.star_data = star_data

def preprocess(inputs, use_factor=False, para1=None, para2=None):
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



if __name__ == '__main__':
    #This is a test on eml1/user/ztf
    path = "../data/EMPIAR-10025/rawdata/micrographs"
    #path = "../stack_0001_DW"
    #dst = "../dataset/EMPIAR-10025/processed/micrographs"
    data = mrcHelper.load_mrc_file(path)
    label = starHelper.read_all_star("../dataset/EMPIAR-10025/rawdata/label_for_training")
    downsampled_data = preprocess(data, False, para1=1024, para2=1024)
    #data = preprocess(data, False, para1=1024, para2=1024)
    downsampled_label = []
    for i in range(len(label)):
        name = label[i].name
        content = starHelper.downsample_with_size(
            label[i].content,
            (1024 / data[i].header[1], 1024 / data[i].header[0])
        )
        downsampled_label.append(starHelper.StarData(name, content))
    
    
    #label = star.downsample_with_size()
    #mrc.write_mrc(data, dst=dst)