import tensorflow as tf

class DarknetConv_BN_Leaky(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            use_bias=False
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)

    def call(self, input, training=False):
        x = self.conv(input)
        x = self.batch_norm(x, training=training)
        x = self.leaky_relu(x)
        return x

class DarknetConvStack(tf.keras.Model):
    def __init__(self, filters, kernel_size, layers, strides=1):
        super().__init__()
        self.conv = []
        self.pool = tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],
            strides=2
        )
        for i in range(layers):
            self.conv.append(
                DarknetConv_BN_Leaky(
                    filters=2*filters if (i%2==0) else filters,
                    kernel_size=kernel_size if(i%2==0) else kernel_size//2,
                    strides=strides
                )
            )

    def call(self, input, training=False):
        x = self.pool(input)
        for conv in self.conv:
            x = conv(x, training=training)
        return x

class Darknet19(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = DarknetConv_BN_Leaky(
            filters=32,
            kernel_size=3,
        )
        self.conv2 = DarknetConvStack(
            filters=32,
            kernel_size=3,
            layers=1,
            strides=1
        )
        self.conv3 = DarknetConvStack(
            filters=64,
            kernel_size=3,
            layers=3,
            strides=1
        )
        self.conv4 = DarknetConvStack(
            filters=128,
            kernel_size=3,
            layers=3,
            strides=1
        )
        self.conv5 = DarknetConvStack(
            filters=256,
            kernel_size=3,
            layers=5,
            strides=1
        )
        self.conv6 = DarknetConvStack(
            filters=512,
            kernel_size=3,
            layers=5,
            strides=1
        )
        self.conv7 = tf.keras.layers.Conv2D(
            filters=5,
            kernel_size=1,
            strides=1
        )

    def call(self, input, training=False):
        x = self.conv1(input)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.conv6(x, training=training)
        x = self.conv7(x, training=training)
        return x

if __name__ == '__main__':
    #this is a test to check if network is able to run
    import numpy as np
    x = np.random.uniform(0, 1, [224, 224])
    x = np.expand_dims(x.astype(np.float32), axis=-1)
    x = np.expand_dims(x, axis=0)
    model = Darknet19()
    y = model(x, training=True)
    print(y.shape)