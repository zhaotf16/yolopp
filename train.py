import augmenter
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

    if FLAGS.use_limit:
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        #cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_virtual_device_configuration(
            gpus[-1],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )

    data_path = FLAGS.data_path
    label_path = FLAGS.label_path

    #mrc = mrcHelper.load_mrc_file(data_path)
    #star = starHelper.read_all_star(label_path)
    
    #train_data = preprocess.mrc2array(mrc, image_size=1024)
    #train_label = preprocess.star2label(star, 1024, grid_size=64, 
    #    particle_size=(220/7420*1024, 220/7676*1024),
    #)

    batchsize = FLAGS.batch_size
    epochs = FLAGS.epoch
    learning_rate = 0.001
    decay = 0.95
    decay_frequency = 20
    net = cn.PhosaurusNet()
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    data = mrcHelper.load_mrc_file("../dataset/EMPIAR-10025/processed/micrographs")
    label = starHelper.read_all_star("../dataset/EMPIAR-10025/processed/labels")
    data = preprocess.mrc2array(data, image_size=1024)
    label = preprocess.star2label(label, 1024, 64,
        (220/7420*1024, 220/7676*1024)
    )
    
    # train_label blurring
    print('train_label blurring ...')
    epi = 0.1
    K = 5
    label = label * (1 - epi) + epi / K
    # debug version
    train_data = data[0:10, ...]
    train_label = label[0:10, ...]
    valid_data = data[20:30, ...]
    valid_label = label[20:30, ...]
    valid_frequency = 10
    
    for e in range(epochs):
        #shuffle
        index = [i for i in range(train_data.shape[0])]
        np.random.shuffle(index)
        train_data = train_data[index, ...]
        train_label = train_label[index, ...]
        batch_num = np.shape(train_data)[0] // batchsize
        for i in range(batch_num):
            index = i * batchsize
            x = np.copy(train_data[index:index+batchsize, ...])
            y_true = np.copy(train_label[index:index+batchsize, ...])
            x, y_true = augmenter.augment(x, y_true)
            with tf.GradientTape() as tape:
                y_pred = net(x, training=True)
                #xy_loss, wh_loss, obj_loss, no_obj_loss = cn.yolo_loss(y_pred, y_true)
                obj_loss, no_obj_loss = cn.yolo_loss(y_pred, y_true)
                #xy_loss = tf.reduce_mean(xy_loss)
                #wh_loss = tf.reduce_mean(wh_loss)
                obj_loss = tf.reduce_mean(obj_loss)
                no_obj_loss = tf.reduce_mean(no_obj_loss)
                #loss = xy_loss + obj_loss + no_obj_loss
                loss = obj_loss + no_obj_loss
            grads = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, net.trainable_variables))

        #print("epoch: %d\txy_loss: %f\tobj_loss: %f\tno_obj_loss:%f\tloss:%f" % 
        #    (e+1, xy_loss, obj_loss, no_obj_loss, loss))
        print("epoch: %d\tobj_loss: %f\tno_obj_loss:%f\tloss:%f" % 
            (e+1, obj_loss, no_obj_loss, loss))
        if (e+1) % decay_frequency == 0:
            learning_rate *= decay
        if (e+1) % valid_frequency == 0:
            valid_num = np.shape(valid_data)[0]
            picked, miss, wrong_picked = 0, 0, 0
            for i in range(valid_num):
                x = np.expand_dims(valid_data[i, ...], axis=0)
                y_true = np.expand_dims(valid_label[i, ...], axis=0)
                y_pred = net(x, training=False)
                y_pred = tf.sigmoid(y_pred)
                print(tf.reduce_max(y_pred))
                print(tf.reduce_min(y_pred))
                for x in range(64):
                    for y in range(64):
                        if y_pred[0,x,y,0] > 0.5 and y_true[0,x,y,0] > 0.5:    
                            picked += 1
                        elif y_pred[0,x,y,0] > 0.5 and y_true[0,x,y,0] < 0.5:
                            wrong_picked += 1
                        elif y_pred[0,x,y,0] < 0.5 and y_true[0,x,y,0] > 0.5:
                            miss += 1
                #_, _, objLoss, _  = cn.yolo_loss(y_pred, y_true)
                objLoss, noObjLoss = cn.yolo_loss(y_pred, y_true)
                print(objLoss + noObjLoss)
            print(
                "Validation epoch: %d\tpicked: %d\tmiss: %d\twrong_picked:%d" %
                (e+1, picked, miss, wrong_picked)
            )
            picked, miss, wrong_picked = 0, 0, 0
            for i in range(np.shape(train_data)[0]):
                x = np.expand_dims(train_data[i, ...], axis=0)
                true = np.expand_dims(train_label[i, ...], axis=0)
                pred = net(x, training=False)
                pred = tf.sigmoid(pred)
                print(tf.reduce_max(pred))
                for x in range(64):
                    for y in range(64):
                        if pred[0,x,y,0] > 0.5 and true[0,x,y,0] > 0.5:
                            picked += 1
                        elif pred[0,x,y,0] > 0.5 and true[0,x,y,0] < 0.5:
                            wrong_picked += 1
                        elif pred[0,x,y,0] < 0.5 and true[0,x,y,0] > 0.5:
                            miss += 1
            print(
                "While on training epoch: %d\tpicked: %d\tmiss: %d\twrong_picked:%d" %
                (e+1, picked, miss, wrong_picked)
            )
    net.save_weights('yolopp_weights/', save_format='tf')

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string("data_path", None, "path of data(mrc, etc.)")
    flags.DEFINE_string("label_path", None, "path of labels(star, etc.)")
    flags.DEFINE_integer("batch_size", 1, "batch size of training data")
    flags.DEFINE_integer("epoch", 400, "total_epochs")
    flags.DEFINE_bool("use_limit", True, "set gpu memory limit") 
    flags.DEFINE_string("save_weights", "../yolopp_weights", "dir to store weights")
    
    flags.mark_flag_as_required("data_path")
    flags.mark_flag_as_required("label_path")

    app.run(train)
