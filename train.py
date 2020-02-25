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
    
    #array = preprocess.mrc2array(mrc, image_size=1024)
    #label = preprocess.star2label(star, 1024, grid_size=64, 
    #    particle_size=(220/7420*1024, 220/7676*1024),
    #)

    batchsize = FLAGS.batch_size
    epochs = FLAGS.epoch
    learning_rate = 0.0001
    net = cn.PhosaurusNet()
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    valid = mrcHelper.load_mrc_file("../dataset/EMPIAR-10025/processed/micrographs")
    valid_labels = starHelper.read_all_star("../dataset/EMPIAR-10025/processed/labels")
    valid = preprocess.mrc2array(valid, image_size=1024)
    valid_labels = preprocess.star2label(valid_labels, 1024, 64,
        (220/7420*1024, 220/7676*1024)
    )
    #preprocess.normalize_uint8(valid)
    # debug version
    array = valid[0:10, ...]
    label = valid_labels[0:10, ...]
    valid = valid[20:30, ...]
    valid_labels = valid_labels[20:30, ...]
    valid_frequency = 10

    # data augmentation
    # average blurring
    print('average blurring')
    average_blur_data = augmenter.averageBlur(array, (3,8))
    average_blur_label = np.copy(label)
    # gaussian blurring
    print('gaussian blurring')
    gaussian_blur_data = augmenter.gaussianBlur(array, (0,3))
    gaussian_blur_label = np.copy(label)
    
    array = np.concatenate((array, average_blur_data, gaussian_blur_data))
    label = np.concatenate((label, average_blur_label, gaussian_blur_label))

    # add noise
    print('adding gaussian noise')
    noisy_data = augmenter.gaussianNoise(array)
    noisy_label = np.copy(label)

    array = np.concatenate((array, noisy_data))
    label = np.concatenate((label, noisy_label))
    
    # dropout
    #print('dropout')
    #dropout_data = augmenter.dropout(array, 0.1)
    #dropout_label = np.copy(label)

    ##array = np.concatenate((array, dropout_data))
    #label = np.concatenate((label, dropout_label))

    # contrast normalization
    normalized_data = augmenter.contrastNormalization(array)
    normalized_label = np.copy(label)

    array = np.concatenate((array, normalized_data))
    label = np.concatenate((label, normalized_label))

    print(array.shape)
   # preprocess.normalize_uint8(array)
    for e in range(epochs):
        #shuffle
        index = [i for i in range(array.shape[0])]
        np.random.shuffle(index)
        array = array[index, ...]
        label = label[index, ...]
        batch_num = np.shape(array)[0] // batchsize
        for i in range(batch_num):
            index = i * batchsize
            x = np.copy(array[index:index+batchsize, ...])
            y_true = np.copy(label[index:index+batchsize, ...])
            #x, y_true = augmenter.augment(x, y_true)
            with tf.GradientTape() as tape:
                y_pred = net(x, training=True)
                xy_loss, wh_loss, obj_loss, no_obj_loss = cn.yolo_loss(y_pred, y_true)
                xy_loss = tf.reduce_mean(xy_loss)
                #wh_loss = tf.reduce_mean(wh_loss)
                obj_loss = tf.reduce_mean(obj_loss)
                no_obj_loss = tf.reduce_mean(no_obj_loss)
                loss = xy_loss + obj_loss + no_obj_loss
            grads = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, net.trainable_variables))

        print("epoch: %d\txy_loss: %f\tobj_loss: %f\tno_obj_loss:%f\tloss:%f" % 
            (e+1, xy_loss, obj_loss, no_obj_loss, loss))
        if (e+1) % valid_frequency == 0:
            valid_num = np.shape(valid)[0]
            picked, miss, wrong_picked = 0, 0, 0
            for i in range(valid_num):
                valid_data = np.expand_dims(valid[i, ...], axis=0)
                valid_true = np.expand_dims(valid_labels[i, ...], axis=0)
                valid_pred = net(valid_data, training=False)
                for x in range(64):
                    for y in range(64):
                        if tf.sigmoid(valid_pred[0,x,y,4]) > 0.5 and valid_true[0,x,y,4] == 1.0:
                            picked += 1
                        elif tf.sigmoid(valid_pred[0,x,y,4]) > 0.5 and valid_true[0,x,y,4] == 0:
                            wrong_picked += 1
                        elif tf.sigmoid(valid_pred[0,x,y,4]) < 0.5 and valid_true[0,x,y,4] == 1.0:
                            miss += 1
            _, _, objLoss, _ = cn.yolo_loss(valid_pred, valid_true)
            print(objLoss)
            print(
                "Validation epoch: %d\tpicked: %d\tmiss: %d\twrong_picked:%d" %
                (e+1, picked, miss, wrong_picked)
            )
            picked, miss, wrong_picked = 0, 0, 0
            for i in range(np.shape(array)[0]):
                data = np.expand_dims(array[i, ...], axis=0)
                true = np.expand_dims(label[i, ...], axis=0)
                pred = net(data, training=False)
                for x in range(64):
                    for y in range(64):
                        if tf.sigmoid(pred[0,x,y,4]) > 0.5 and true[0,x,y,4] == 1.0:
                            picked += 1
                        elif tf.sigmoid(pred[0,x,y,4]) > 0.5 and true[0,x,y,4] == 0:
                            wrong_picked += 1
                        elif tf.sigmoid(pred[0,x,y,4]) < 0.5 and true[0,x,y,4] == 1.0:
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
