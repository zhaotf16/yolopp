import numpy as np
import tensorflow as tf

from absl import app
from absl import flags

import mrcHelper
import starHelper
import preprocess
import model.cryolo_net as cn

FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", None, "dir of input micrographs")
flags.DEFINE_string("output_dir", None, "dir of output predictions")

#Local settings
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
tf.config.experimental.set_virtual_device_configuration(
    gpus[-1],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)

def pick(argv):
    del argv

    data_path = FLAGS.input_dir
    mrc = mrcHelper.load_mrc_file(data_path)
    array = preprocess.mrc2array(mrc, image_size=1024)

    batchsize = 1
    net = cn.PhosaurusNet()

    #debug:
    #array = np.expand_dims(array[0, ...], axis=0)
    #batchsize = 1
    net.load_weights('yolopp_weights/')
    batch_num = np.shape(array)[0] // batchsize
        
    for i in range(batch_num):
        index = i * batchsize
        x = array[index:index+batchsize, ...]
        y_pred = net(x, training=True)
    
    #format of output files is STAR
    stars = []

if __name__ == '__main__':
    app.run(pick)
