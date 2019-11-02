#================================================================
# 日期       ：  2019年11月2日12:34:43
# 建模类型   ：  自定义型
# 改进人     ：  亓志国
# 模型       ：  ResNet
# 文件数量   ：  1
#================================================================
import numpy as np
import tensorflow as tf
import cv2, random
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Activation, InputLayer
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Multiply, Dropout


class SELayer(Model):
    def __init__(self, filters, reduction=16):
        super(SELayer, self).__init__()
        self.gap = GlobalAveragePooling2D()
        self.fc = Sequential([
            # use_bias???
            Dense(filters // reduction,
                  input_shape=(filters, ),
                  use_bias=False),
            Dropout(0.5),
            BatchNormalization(),
            Activation('relu'),
            Dense(filters, use_bias=False),
            Dropout(0.5),
            BatchNormalization(),
            Activation('sigmoid')
        ])
        self.mul = Multiply()

    def call(self, input_tensor):
        weights = self.gap(input_tensor)
        weights = self.fc(weights)
        return self.mul([input_tensor, weights])


def DBL(filters, ksize, strides=1):
    layers = [
        BatchNormalization(),
        LeakyReLU(),
        Conv2D(filters, (ksize, ksize),
               strides=strides,
               padding='same',
               use_bias=False)
    ]
    return Sequential(layers)


class ResUnit(Model):
    def __init__(self, filters):
        super(ResUnit, self).__init__()
        self.dbl1 = DBL(filters // 2, 1)
        self.dbl2 = DBL(filters, 3)
        self.se = SELayer(filters, 1)

    def call(self, input_tensor):
        x = self.dbl1(input_tensor)
        x = self.dbl2(x)
        x = self.se(x)
        x += input_tensor
        return x


def SENet(input_shape,
          output_filters,
          filters=[64, 128, 256, 512, 1024],
          res_n=[1, 2, 8, 8, 4]):
    layers = []
    layers += [
        Conv2D(32, (7, 7),
               input_shape=input_shape,
               padding='same',
               use_bias=False)
    ]
    for fi, f in enumerate(filters):
        layers += [DBL(f, 3, 2)] + [ResUnit(f)] * res_n[fi]
    layers += [
        Dropout(0.5),
        BatchNormalization(),
        LeakyReLU(),
        Conv2D(output_filters, (7, 7), padding='same'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ]
    return Sequential(layers)




def senet():
    model = SENet((224, 224, 3), 10)
    return model

