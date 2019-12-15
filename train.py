import mrcHelper
import starHelper
import dataLoader
import numpy as np
import tensorflow as tf
import model.darknet as dn
import model.cryolo_net as cn
#Local settings
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
tf.config.experimental.set_virtual_device_configuration(
    gpus[-1],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)

def train():
    data_path = "../dataset/EMPIAR-10025/processed/micrographs"
    label_path = "../dataset/EMPIAR-10025/processed/labels"
    #path = "../stack_0001_DW"
    #dst = "../dataset/EMPIAR-10025/processed/micrographs"
    #dst1 = "../dataset/EMPIAR-10025/processed/labels"
    mrc = mrcHelper.load_mrc_file(data_path)
    star = starHelper.read_all_star(label_path)
    
    array = dataLoader.mrc2array(mrc, image_size=1024)
    array = np.expand_dims(array, axis=1)
    label = dataLoader.star2label(star, 1024)

    batchsize = 1
    epochs = 10
    learning_rate = 0.1
    net = cn.PhosaurusNet()
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    for e in range(epochs):
        batch_num = array.shape[0] // batchsize
        for i in range(batch_num):
            x = array[i+batchsize-1, ...]
            y_true = label[i+batchsize-1, ...]
            with tf.GradientTape() as tape:
                y_pred = net(x, training=True)
                loss = cn.yolo_loss(y_pred, y_true)
                loss = tf.reduce_mean(loss)
            print("epoch: %d\tbatch: %d\t loss: %f" % (e+1, i+1, loss))
            grads = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, net.trainable_variables))
    
if __name__ == '__main__':
    train()