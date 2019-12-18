import mrcHelper
import starHelper
import preprocess
import numpy as np
import tensorflow as tf
import model.cryolo_net as cn

from absl import app
from absl import flags

def train(argv):
    del argv

    data_path = FLAGS.data_path
    label_path = FLAGS.label_path

    mrc = mrcHelper.load_mrc_file(data_path)
    star = starHelper.read_all_star(label_path)
    
    array = preprocess.mrc2array(mrc, image_size=1024)
    label = preprocess.star2label(star, 1024, grid_size=64, 
        particle_size=(110/7420*1024, 110/7676*1024),
    )
    
    batchsize = FLAGS.batch_size
    epochs = FLAGS.epoch
    learning_rate = 0.001
    net = cn.PhosaurusNet()
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    #debug:
    #array = np.expand_dims(array[0, ...], axis=0)
    #batchsize = 1
    #net.load_weights('yolopp_weights/')
    for e in range(epochs):
        batch_num = np.shape(array)[0] // batchsize
        total_loss = 0
        for i in range(batch_num):
            index = i * batchsize
            x = array[index:index+batchsize, ...]
            y_true = label[index:index+batchsize, ...]
            with tf.GradientTape() as tape:
                #y_pred = net.call(x, training=True)
                y_pred = net(x, training=True)
                #loss = cn.yolo_loss(y_pred, y_true)
                #loss = tf.reduce_mean(loss)
                #debug:
                xy_loss, wh_loss, obj_loss = cn.yolo_loss(y_pred, y_true)
                xy_loss = tf.reduce_mean(xy_loss)
                wh_loss = tf.reduce_mean(wh_loss)
                obj_loss = tf.reduce_mean(obj_loss)
                loss = xy_loss + wh_loss + obj_loss
                total_loss += loss
            #print("epoch: %d\tbatch: %d\tloss: %f" % (e+1, i+1, loss))
            #average_loss = total_loss / batch_num
            #print("epoch: %d\tloss:%f" %(e, average_loss))
            print("epoch: %d\tbatch: %d\txy_loss: %f\twh_loss: %f\tobj_loss: %f\tloss: %f" % 
            (e+1, i+1, xy_loss, wh_loss, obj_loss, loss))
            grads = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, net.trainable_variables))
    net.save_weights('yolopp_weights/', save_format='tf')
    
if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string("data_path", None, "path of data(mrc, etc.)")
    flags.DEFINE_string("label_path", None, "path of labels(star, etc.)")
    flags.DEFINE_integer("batch_size", 1, "batch size of training data")
    flags.DEFINE_integer("epoch", 70, "total_epochs")
    flags.DEFINE_bool("use_limit", True, "set gpu memory limit")

    flags.mark_flag_as_required("data_path")
    flags.mark_flag_as_required("label_path")

    if FLAGS.use_limit:
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_virtual_device_configuration(
            gpus[-1],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
    app.run(train)