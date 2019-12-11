import os
import mrc
import numpy as np

class MrcData():
    def __init__(self, name, data, label):
        #info is a tuple:(data, header, extend_header)
        self.name = name
        self.data = data
        #self.data = info[0]
        #self.header = info[1]
        #self.extend_header = info[2]
        self.label = label


def load_mrc_file(path):
    if not os.path.isdir(path):
        print(path, " is not a valid directory")
        return
    if not path.endswith('/'):
        path += '/'
        mrc_data = []
    for file in os.listdir(path):
        if file.endswith('.mrc'):
            print("Loading %s ..." % (file))
            with open(path+file, "rb") as f:
                content = f.read()
            data, header, extend_header = mrc.parse(content=content)
            name = file
            # TODO: load and process label according to STAR or EMAN
            mrc_data.append(MrcData(name=file, data=data, label=""))
    return mrc_data

def preprocess(inputs, use_factor=False, para1=None, para2=None):
    #This method executes a downsampling on the mrc
    #Inputs is a list of MrcData
    #If use_factor is True, para1 represents factor and para2 represents shape
    #Else para1 and para2 represents the target size(y, x)
    print(type(inputs[0]))
    for i in range(len(inputs)):
        if use_factor:
            print("Prcocessing %s ..." % ( inputs[i].name))
            inputs[i].data = mrc.downsample_with_factor(
                inputs[i].data,
                factor=para1,
                shape=para2
            )
            #TODO: fix labels for the downsampled micrographs
        else:
            print("Prcocessing %s ..." % (inputs[i].name))
            inputs[i].data = mrc.downsample_with_size(
                inputs[i].data,
                size1=para1,
                size2=para2
            )
            # TODO: fix labels for the downsampled micrographs

    return inputs

def write_mrc(inputs, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    if not dst.endswith('/'):
        dst += '/'

    #for mrc_data in inputs:
    for mrc_data in inputs:
        print(type(mrc_data))
        print("Writing %s ..." % (mrc_data.name))
        data = np.expand_dims(mrc_data.data, axis=0)
        with open(dst+mrc_data.name, "wb") as f:
            mrc.write(f, data)

if __name__ == '__main__':
    #This is a test on eml1/user/ztf
    path = "../topaz/data/EMPIAR-10025/rawdata/micrographs"
    #path = "../stack_0001_DW"
    dst = "./dataset"

    data = load_mrc_file(path)
    data = preprocess(data, False, para1=1024, para2=1024)
    print(data)
    write_mrc(data, dst=dst)