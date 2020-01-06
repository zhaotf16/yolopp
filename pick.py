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
    for i in range(1):
        index = i * batchsize
        x = array[index:index+batchsize, ...]
        #y_true = label[index:index+batchsize, ...]
        y_pred = net(x, training=False)
        bbox, score = cn.yolo_head(y_pred)
        #boxes = cn.non_max_suppression(bbox, score, 0.5)
        #score = tf.cast(score>0.5, tf.float32)
        #print(tf.reduce_sum(score))
        #xy_loss, wh_loss, obj_loss = cn.yolo_loss(y_pred, y_true)
        #xy_loss = tf.reduce_mean(xy_loss)
        #wh_loss = tf.reduce_mean(wh_loss)
        #obj_loss = tf.reduce_mean(obj_loss)
        #loss = xy_loss + wh_loss + obj_loss
        #print("batch: %d\txy_loss: %f\twh_loss: %f\tobj_loss: %f\tloss: %f" % 
        #(i+1, xy_loss, wh_loss, obj_loss, loss))

        for n in range(batchsize):
            box_x1y1, box_x2y2 = tf.split(bbox[n,...], (2, 2), axis=-1)
            box_xy, _ = (box_x1y1 + box_x2y2) / 2, box_x2y2 - box_x1y1
            confidence = score[n, ...]
            print(tf.shape(confidence))
            w, h = tf.shape(confidence)[0], tf.shape(confidence)[1]
            star = starHelper.StarData(str(i*batchsize+n), [])
            for a in range(w):
                for b in range(h):
                    if confidence[a, b] > 0.7:
                        #star.content.append((box_xy[a,b,0]*7420, box_xy[a,b,1]*7676))
                        star.content.append((
                           (tf.sigmoid(y_pred[n,a,b,0])+a)*7420.0/64, (tf.sigmoid(y_pred[n,a,b,1])+b)*7676.0/64
                        ))
            stars.append(star)
    starHelper.write_star(stars, dst)
if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string("input_dir", None, "dir of input micrographs")
    flags.DEFINE_string("output_dir", None, "dir of output predictions")
    flags.DEFINE_string("weights_dir", "../yolopp_weights/", "dir or pretrained weights")
    app.run(pick)
