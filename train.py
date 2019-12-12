import model.darknet as dn
import model.cryolo_net as cn
import tensorflow as tf

#Local settings
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
tf.config.experimental.set_virtual_device_configuration(
    gpus[-1],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)

def train():
    import numpy as np
    x = np.random.uniform(0, 1, [4, 224, 224])
    x = np.expand_dims(x.astype(np.float32), axis=-1)
    #x = np.expand_dims(x, axis=0)
    net = cn.PhosaurusNet()
    #y_pred = model(x, training=True)
    #print(y.shape)
    y_true = np.random.uniform(0, 1, [4, 14, 14, 5])
    y_true = y_true.astype(np.float32)

    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    for i in range(3):
        print()
    epochs = 5
    for e in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = net(x, training=True)
            loss = cn.yolo_loss(y_pred, y_true)
            loss = tf.reduce_mean(loss)
        print(loss)
        grads = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, net.trainable_variables))
    
if __name__ == '__main__':
    train()