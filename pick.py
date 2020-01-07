import numpy as np
import tensorflow as tf

from absl import app
from absl import flags

import mrcHelper
import starHelper
import preprocess
import model.cryolo_net as cn

#Local settings
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[-1],
   [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)

def pick(argv):
    del argv

    data_path = FLAGS.input_dir
    dst = FLAGS.output_dir

    mrc = mrcHelper.load_mrc_file(data_path)
    array = preprocess.mrc2array(mrc, image_size=1024)

    batchsize = 1
    net = cn.PhosaurusNet()

    #debug:
    import numpy as np
    array = np.expand_dims(array[0, ...], axis=0)
    #star = starHelper.read_all_star("../dataset/STAR/test")
    #label = preprocess.star2label(star, 1024, grid_size=64, 
    #    particle_size=(110/7420*1024, 110/7676*1024),
    #)
    weights_path = FLAGS.weights_dir
    net.load_weights(weights_path)
    batch_num = np.shape(array)[0] // batchsize
    #format of output files is STAR
    stars = []
    stars = []
    for i in range(batch_num):
        index = i * batchsize
        x = array[index:index+batchsize, ...]
        y_pred = net(x, training=False)
        _, score = cn.yolo_head(y_pred)
        for n in range(batchsize):
            confidence = tf.squeeze(score[n, ...], axis=-1)
            confidence = tf.sigmoid(confidence)
            #print(tf.shape(confidence))
            w, h = tf.shape(confidence)[0], tf.shape(confidence)[1]
            star = starHelper.StarData(str(i*batchsize+n), [])
            for a in range(w):
                for b in range(h):
                    #print("(%d, %d) true: %f, pred: %f" % (a, b, true_confidence[a, b], confidence[a, b]))
                    if confidence[a, b] > 0.7:
                        star.content.append((
                            (a+tf.sigmoid(y_pred[n,a,b,0]))*7420.0/64.0, (b+tf.sigmoid(y_pred[n,a,b,1]))*7676.0/64.0
                        ))
            stars.append(star)
    starHelper.write_star(stars, dst)
if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string("input_dir", None, "dir of input micrographs")
    flags.DEFINE_string("output_dir", None, "dir of output predictions")
    flags.DEFINE_string("weights_dir", "../yolopp_weights/", "dir or pretrained weights")
    app.run(pick)
