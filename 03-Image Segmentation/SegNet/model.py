from tensorflow.keras.layers import Activation, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


class SegNet(Model):
    def get_config(self):
        pass

    def __init__(self, n_labels=0, kernel=3, pool_size=(2, 2), output_mode="softmax"):
        super(SegNet, self).__init__(name='')
        self.n_labels = n_labels
        self.output_mode = output_mode
        self.pool_size = pool_size
        self.unit01 = Sequential([
            layers.Conv2D(64, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
            layers.Conv2D(64, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
        ])
        self.unit02 = Sequential([
            layers.Conv2D(128, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
            layers.Conv2D(128, (kernel, kernel), padding="same"),
            BatchNormalization()
        ])
        self.unit03 = Sequential([
            layers.Conv2D(256, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
            layers.Conv2D(256, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
            layers.Conv2D(256, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
        ])
        self.unit04 = Sequential([
            layers.Conv2D(512, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
            layers.Conv2D(512, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
            layers.Conv2D(512, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
        ])
        self.unit05 = Sequential([
            layers.Conv2D(512, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
            layers.Conv2D(512, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
            layers.Conv2D(512, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
        ])
        self.unit06 = Sequential([
            layers.Conv2D(512, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
            layers.Conv2D(512, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
            layers.Conv2D(512, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
        ])
        self.unit07 = Sequential([
            layers.Conv2D(512, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
            layers.Conv2D(512, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
            layers.Conv2D(256, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
        ])
        self.unit08 = Sequential([
            layers.Conv2D(256, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
            layers.Conv2D(256, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
            layers.Conv2D(128, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization()
        ])
        self.unit09 = Sequential([
            layers.Conv2D(128, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
            layers.Conv2D(64, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization()
        ])
        self.unit10 = Sequential([
            layers.Conv2D(64, (kernel, kernel), padding="same", activation="relu"),
            BatchNormalization(),
            layers.Conv2D(n_labels, (1, 1), padding="valid"),
            BatchNormalization()
        ])

    def call(self, inputs, training=None, mask=None):
        # encoder
        output_01 = self.unit01(inputs)
        pool_1, mask_1 = MaxPoolingWithArgmax2D(self.pool_size)(output_01)
        output_02 = self.unit02(pool_1)
        pool_2, mask_2 = MaxPoolingWithArgmax2D(self.pool_size)(output_02)
        output_03 = self.unit03(pool_2)
        pool_3, mask_3 = MaxPoolingWithArgmax2D(self.pool_size)(output_03)
        output_04 = self.unit04(pool_3)
        pool_4, mask_4 = MaxPoolingWithArgmax2D(self.pool_size)(output_04)
        output_05 = self.unit05(pool_4)
        pool_5, mask_5 = MaxPoolingWithArgmax2D(self.pool_size)(output_05)
        print("Build enceder done..")

        # decoder
        unpool_1 = MaxUnpooling2D(self.pool_size)([pool_5, mask_5])
        output_06 = self.unit06(unpool_1)
        unpool_2 = MaxUnpooling2D(self.pool_size)([output_06, mask_4])
        output_07 = self.unit07(unpool_2)
        unpool_3 = MaxUnpooling2D(self.pool_size)([output_07, mask_3])
        output_08 = self.unit08(unpool_3)
        unpool_4 = MaxUnpooling2D(self.pool_size)([output_08, mask_2])
        output_09 = self.unit09(unpool_4)
        unpool_5 = MaxUnpooling2D(self.pool_size)([output_09, mask_1])
        output_10 = self.unit10(unpool_5)
        output_11 = Reshape(
            (256 * 256, self.n_labels),
            input_shape=(256, 256, self.n_labels),
        )(output_10)

        outputs = Activation(self.output_mode)(output_11)
        print("Build decoder done..")
        return outputs


if __name__ == '__main__':
    import tensorflow as tf
    import numpy as np

    model = SegNet(n_labels=21, kernel=3, pool_size=(2, 2), output_mode="softmax")
    model.build(input_shape=(1, 256, 256, 3))
    data = np.ones((1, 256, 256, 3))
    result = model(data)
    print(result.shape)
