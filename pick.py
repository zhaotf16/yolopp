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
    star = starHelper.read_all_star("../dataset/EMPIAR-10025/processed/test_label")
    label = preprocess.star2label(star, 1024, grid_size=64, 
        particle_size=(110/7420*1024, 110/7676*1024),
    )

    net.load_weights('yolopp_weights/')
    batch_num = np.shape(array)[0] // batchsize

    for i in range(batch_num):
        index = i * batchsize
        x = array[index:index+batchsize, ...]
        y_true = label[index:index+batchsize, ...]
        y_pred = net(x, training=False)
        #bbox, score = cn.yolo_head(y_pred)
        #boxes = cn.non_max_suppression(bbox, score, 0.5)
        #score = tf.cast(score>0.2, tf.float32)
        #print(tf.reduce_sum(score))
        xy_loss, wh_loss, obj_loss = cn.yolo_loss(y_pred, y_true)
        xy_loss = tf.reduce_mean(xy_loss)
        wh_loss = tf.reduce_mean(wh_loss)
        obj_loss = tf.reduce_mean(obj_loss)
        loss = xy_loss + wh_loss + obj_loss
        print("batch: %d\txy_loss: %f\twh_loss: %f\tobj_loss: %f\tloss: %f" % 
        (i+1, xy_loss, wh_loss, obj_loss, loss))
    #format of output files is STAR
    stars = []

if __name__ == '__main__':
    app.run(pick)
