import tensorflow as tf
from tensorflow.keras import layers

filter_size = 64
filter_depth = 1

def make_generator_model(filter_size = filter_size, filter_depth = filter_depth):
    model = tf.keras.Sequential()
    model.add(layers.Dense((filter_size // 4) * (filter_size // 4) * filter_depth * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((filter_depth, filter_size // 4, filter_size // 4, 256)))
    assert model.output_shape == (None, filter_depth, filter_size // 4, filter_size // 4, 256)  # Note: None is the batch size

    model.add(layers.Conv3DTranspose(128, (1, 5, 5), strides=(1, 1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, filter_depth, filter_size // 4, filter_size // 4, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv3DTranspose(64, (1, 5, 5), strides=(1, 2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, filter_depth, filter_size // 2, filter_size // 2, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv3DTranspose(1, (1, 5, 5), strides=(1, 2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, filter_depth, filter_size, filter_size, 1)

    return model

def make_discriminator_model(filter_size = filter_size, filter_depth = filter_depth):
    model = tf.keras.Sequential()
    model.add(layers.Conv3D(64, (1, 5, 5), strides=(1, 2, 2), padding='same',
                                     input_shape=[filter_depth, filter_size, filter_size, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv3D(128, (1, 5, 5), strides=(1, 2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model