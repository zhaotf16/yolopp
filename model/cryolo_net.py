import tensorflow as tf
from model.darknet import Darknet19, DarknetConv_BN_Leaky
from tensorflow.keras.layers import Reshape, Activation, Conv2D, Input, \
    MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, LeakyReLU, \
    concatenate, UpSampling2D
#Local settings
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
tf.config.experimental.set_virtual_device_configuration(
    gpus[-1],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)
#anchor = tf.constant([[80., 80.]], dtype=tf.float32) / 1024

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
        self.dropout = tf.keras.layers.Dropout(rate=0.2)

    @tf.function
    def call(self, input, training=False):
        x = self.conv1(input)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        y = x
        x = self.conv6(x, training=training)
        x = self.additional_conv1(x, training=training)
        x = self.additional_conv2(x, training=training)
        x = self.upsample(x)
        y = self.additional_conv3(y, training=training)
        x = self.concatenate([x, y])
        x = self.dropout(x, training=training)
        x = self.conv7(x, training=training)
        return x

def space_to_depth_x2(x):
    return tf.nn.space_to_depth(x, block_size=2)

def yolov2():
    x = inputs = tf.keras.Input(shape=(1024,1024,1))
    #Layer1
    x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(x)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    #Layer2
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    # Layer 3
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Layer 4
    x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Layer 5
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    # Layer 6
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Layer 7
    x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Layer 8
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    # Layer 9
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
    x = BatchNormalization(name='norm_9')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
    x = BatchNormalization(name='norm_10')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
    x = BatchNormalization(name='norm_11')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
    x = BatchNormalization(name='norm_12')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Layer 13
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    conv13 = x # 
    
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    # Layer 14
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 18 - darknet
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
    x = BatchNormalization(name='norm_18')(x)
    darknet = LeakyReLU(alpha=0.1)(x)

    ##################################################################################
    
    # Layer 19
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(darknet)
    x = BatchNormalization(name='norm_19')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
    x = BatchNormalization(name='norm_20')(x)
    conv20 = LeakyReLU(alpha=0.1)(x)
    
    # Layer 21
    conv21 = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(conv13)
    conv21 = BatchNormalization(name='norm_21')(conv21)
    conv21 = LeakyReLU(alpha=0.1)(conv21)
    #conv21_reshaped = Lambda(space_to_depth_x2, name='space_to_depth')(conv21) 
    conv21_reshaped = UpSampling2D(2, name='upsampling')(conv21)

    x = concatenate([conv21_reshaped, conv20])
    
    # Layer 22
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
    x = BatchNormalization(name='norm_22')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Layer 23 - output
    outputs = Conv2D(5, (1,1), strides=(1,1), padding='same', name='conv_23')(x)
    
    return tf.keras.models.Model(inputs, outputs)

def yolo_head(features):
    #the output of cnn is (tx, ty, tw, th, confidence)
    #tx, ty are relative to a single cell, thus we use sigmoid() plus offset
    #to gain their values relatvie to the whole meshgrid.(used in drawing boxes)
    grid_size = tf.shape(features)[1]
    box_xy, box_wh, confidence = tf.split(features, (2, 2, 1), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    box_wh = tf.exp(box_wh)
    confidence = tf.sigmoid(confidence)

    #meshgrid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    #meshgrid = tf.stack(meshgrid, axis=-1)
    #meshgrid = tf.expand_dims(meshgrid, axis=0)
    #debug:
    meshgrid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid, coord = meshgrid[0], meshgrid[1]
    grid, coord = tf.transpose(grid), tf.transpose(coord)
    grid, coord = tf.expand_dims(grid, axis=-1), tf.expand_dims(coord, axis=-1)
    meshgrid = tf.concat((grid, coord), axis=-1)
    meshgrid = tf.expand_dims(meshgrid, axis=0)

    box_xy = (box_xy + tf.cast(meshgrid, tf.float32)) / tf.cast(grid_size, tf.float32)
    #here anchor is a relative box to the whole page: eg. 160/1024. no need to divide a gridsize
    #box_wh = box_wh * anchor

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    pred_box = tf.concat((box_x1y1, box_x2y2), axis=-1)

    return pred_box, confidence


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                   tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                   tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    intersection = w * h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                 (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                 (box_2[..., 3] - box_2[..., 1])
    return intersection / (box_1_area + box_2_area - intersection)

def yolo_loss(y_pred, y_true, ignore_threshold=0.75):
    #y_pred: yolo_output [batch, grid, grid, (x, y, w, h, confidence)]
    #y_true: true_boxes [batch, grid, grid, (x, y, w, h, confidence)]
    #y_true is relative to the whole image, and so is anchor
    object_scale = 5.0
    coordinates_scale = 1.0
    no_object_scale = 1.0
    #pred_xy is ratio to a single cell
    pred_xy, pred_wh = y_pred[..., 0:2], y_pred[..., 2:4]
    pred_xy = tf.sigmoid(pred_xy)
    #box_xy and true_xy are ratio the whole meshgrid and input_image
    pred_box, pred_confidence = yolo_head(y_pred)
    true_xy, true_wh, true_confidence = tf.split(y_true, (2, 2, 1), axis=-1)
    true_x1y1 = true_xy - true_wh / 2
    true_x2y2 = true_xy + true_wh / 2
    true_box = tf.concat((true_x1y1, true_x2y2), axis=-1)

    grid_size = tf.shape(y_true)[1]
    meshgrid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid, coord = meshgrid[0], meshgrid[1]
    grid, coord = tf.transpose(grid), tf.transpose(coord)
    grid, coord = tf.expand_dims(grid, axis=-1), tf.expand_dims(coord, axis=-1)
    meshgrid = tf.concat((grid, coord), axis=-1)
    meshgrid = tf.expand_dims(meshgrid, axis=0)
    
    #debug:
    true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(meshgrid, tf.float32)

    #true_wh = tf.math.log(true_wh / anchor)
    true_wh = tf.math.log(true_wh)
    true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)
    
    mask = tf.squeeze(true_confidence, axis=-1)
    true_box = tf.boolean_mask(true_box, tf.cast(mask, tf.bool))
    best_iou = tf.reduce_max(broadcast_iou(pred_box, true_box), axis=-1)
    ignore_mask = tf.cast(best_iou < ignore_threshold, tf.float32)

    xy_loss = mask * coordinates_scale * \
              tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
    wh_loss = mask * coordinates_scale * \
              tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
    xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2))
    wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2))
    
    obj_loss = tf.reduce_sum(tf.square(true_confidence - pred_confidence), axis=-1)
    #obj_loss = tf.keras.losses.binary_crossentropy(true_confidence, pred_confidence)
    #obj_loss = object_scale * mask * obj_loss + no_object_scale * (1 - mask) * obj_loss
    no_obj_loss = (1 - mask) * no_object_scale * obj_loss * ignore_mask
    obj_loss = object_scale * mask * obj_loss
    no_obj_loss = tf.reduce_sum(no_obj_loss, axis=(1, 2))
    obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2))

    #debug:
    return xy_loss, wh_loss, obj_loss, no_obj_loss
    #return xy_loss + wh_loss + obj_loss

def non_max_suppression(boxes, scores, iou_threshold):
    '''
    Now the network cannot work    
    '''
    pass

def PhosaurusNet_test():
    # this is a test to check if network is able to run
    import numpy as np
    x = np.random.uniform(0, 1, [1024, 1024])
    x = np.expand_dims(x.astype(np.float32), axis=-1)
    x = np.expand_dims(x, axis=0)
    model = PhosaurusNet()
    y_pred = model(x, training=True)
    #print(y.shape)
    y_true = np.random.uniform(0, 1, [64, 64, 5])
    y_true = np.expand_dims(y_true.astype(np.float32), axis=0)

    return yolo_loss(y_pred, y_true)

if __name__ == '__main__':
   PhosaurusNet_test()