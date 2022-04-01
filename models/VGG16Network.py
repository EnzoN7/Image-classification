import tensorflow as tf
from keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

class VGG16Network(tf.keras.Model):

    def __init__(self, _nbclasses, _imagesize):
        super().__init__()
        self.vgg16 = VGG16(weights='imagenet',
                           include_top=False,
                           input_shape=(_imagesize, _imagesize, 3),
                           classes=_nbclasses,
                           classifier_activation="softmax")
        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(_nbclasses, activation='softmax')
        self.flatten = Flatten()

    def call(self, inputs):
        x = self.vgg16(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)