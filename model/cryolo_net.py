import tensorflow as tf
from model.darknet import Darknet19, DarknetConv_BN_Leaky
#Local settings
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
tf.config.experimental.set_virtual_device_configuration(
    gpus[-1],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)

class PhosaurusNet(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        # Layer 1
        self.conv1 = DarknetConv_BN_Leaky(32, (3,3))
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)

        # Layer 2
        self.conv2 = DarknetConv_BN_Leaky(64, (3,3))
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)

        # Layer 3 - 5
        self.conv3 = DarknetConv_BN_Leaky(128, (3,3))
        self.conv4 = DarknetConv_BN_Leaky(64, (1,1))
        self.conv5 = DarknetConv_BN_Leaky(128, (3,3))
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)

        # Layer 6 - 8
        self.conv6 = DarknetConv_BN_Leaky(256, (3,3))
        self.conv7 = DarknetConv_BN_Leaky(128, (1,1))
        self.conv8 = DarknetConv_BN_Leaky(256, (3,3))
        self.pool8 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)

        # Layer 9 - 13
        self.conv9 = DarknetConv_BN_Leaky(512, (3,3))
        self.conv10 = DarknetConv_BN_Leaky(256, (1,1))
        self.conv11 = DarknetConv_BN_Leaky(512, (3,3))
        self.conv12 = DarknetConv_BN_Leaky(256, (1,1))
        self.conv13 = DarknetConv_BN_Leaky(512, (3,3))
        self.pool13 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)
        
        # Layer 14 - 21
        self.conv14 = DarknetConv_BN_Leaky(1024, (3,3))
        self.conv15 = DarknetConv_BN_Leaky(512, (1,1))
        self.conv16 = DarknetConv_BN_Leaky(1024, (3,3))
        self.conv17 = DarknetConv_BN_Leaky(512, (1,1))
        self.conv18 = DarknetConv_BN_Leaky(1024, (3,3))
        self.conv19 = DarknetConv_BN_Leaky(1024, (3,3))
        self.conv20 = DarknetConv_BN_Leaky(1024, (3,3))
        self.conv21 = DarknetConv_BN_Leaky(256, (3,3))
        # Upsampling and concatenate
        self.upsample = tf.keras.layers.UpSampling2D(2)
        self.concatenate = tf.keras.layers.Concatenate()
        # Dropout and output
        #self.conv22 = tf.keras.layers.Conv2D(5, (1,1))
        self.conv22 = tf.keras.layers.Conv2D(1, (1,1))
 
    def call(self, input, training=True):
        # Layer 1
        x = self.conv1(input, training=training)
        x = self.pool1(x)
        # Layer 2
        x = self.conv2(x, training=training)
        x = self.pool2(x)
        # Layer 3 - 5
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.pool5(x)
        # Layer 6 - 8
        x = self.conv6(x, training=training)
        x = self.conv7(x, training=training)
        x = self.conv8(x, training=training)
        x = self.pool8(x)
        # Layer 9 - 13
        x = self.conv9(x, training=training)
        x = self.conv10(x, training=training)
        x = self.conv11(x, training=training)
        x = self.conv12(x, training=training)
        x = self.conv13(x, training=training)
        y = x
        x = self.pool13(x)
        # Layer 14 - 21
        x = self.conv14(x, training=training)
        x = self.conv15(x, training=training)
        x = self.conv16(x, training=training)
        x = self.conv17(x, training=training)
        x = self.conv18(x, training=training)
        x = self.conv19(x, training=training)
        x = self.conv20(x, training=training)
        y = self.conv21(y, training=training)
        x = self.upsample(x)
        x = self.concatenate([x, y])
        x = self.conv22(x)
        return x

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
    object_scale = 4.0
    coordinates_scale = 1.0
    no_object_scale = 1.0 
    #pred_xy is ratio to a single cell
    '''
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
    '''
    y_pred = tf.sigmoid(y_pred)
    mask = tf.squeeze(y_true, axis=-1)
    #obj_mask = tf.where(mask>0.5, tf.ones_like(mask), tf.zeros_like(mask))
    #no_obj_mask = tf.where(mask<0.5, tf.ones_like(mask), tf.zeros_like(mask))
    #obj_mask = tf.cast(obj_mask, tf.float32)
    #no_obj_mask = tf.cast(no_obj_mask, tf.float32)
    obj_loss = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    #obj_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    no_obj_loss = no_object_scale * obj_loss * (1 - mask)
    obj_loss = object_scale * mask * obj_loss
    obj_loss = tf.reduce_sum(obj_loss, axis=(1,2))
    no_obj_loss = tf.reduce_sum(no_obj_loss, axis=(1,2))
    return obj_loss, no_obj_loss
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
