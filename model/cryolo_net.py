import tensorflow as tf
from darknet import Darknet19, DarknetConv_BN_Leaky

class PhosaurusNet(Darknet19):
    def __init__(self):
        super().__init__()
        self.additional_conv1 = DarknetConv_BN_Leaky(
            filters=1024,
            kernel_size=3,
            strides=1
        )
        self.additional_conv2 = DarknetConv_BN_Leaky(
            filters=1024,
            kernel_size=3,
            strides=1
        )
        self.additional_conv3 = DarknetConv_BN_Leaky(
            filters=256,
            kernel_size=1,
            strides=1
        )
        self.upsample = tf.keras.layers.UpSampling2D(2)
        self.concatenate = tf.keras.layers.Concatenate()

    def call(self, input, training=False):
        x = self.conv1(input)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = y = self.conv5(x, training=training)
        x = self.conv6(x, training=training)
        x = self.additional_conv1(x, training=training)
        x = self.additional_conv2(x, training=training)
        x = self.upsample(x)
        y = self.additional_conv3(y, training=training)
        x = self.concatenate([x, y])
        if training:
            x = tf.keras.layers.Dropout(rate=0.2)(x)
        x = self.conv7(x, training=training)
        return x

if __name__ == '__main__':
    #this is a test to check if network is able to run
    import numpy as np
    x = np.random.uniform(0, 1, [1024, 1024])
    x = np.expand_dims(x.astype(np.float32), axis=-1)
    x = np.expand_dims(x, axis=0)
    model = PhosaurusNet()
    y = model(x, training=True)
    print(y.shape)
    model.summary()