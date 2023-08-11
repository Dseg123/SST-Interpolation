import tensorflow as tf
from tensorflow import keras
import sys
sys.path.append("..")
from get_params import params


my_params = params()
n_t = my_params['windowSize'] * (my_params['blockSize'] // my_params['avgSize'])
tile_size = my_params['tileSize']
seed_size = 100
filter_size = 64
filter_depth = 1



def create_Zeros():
    input_data = keras.layers.Input(shape=(n_t, tile_size, tile_size, 1))
    y = tf.zeros_like(input_data)
    model = keras.models.Model(inputs = input_data, outputs = y)
    
    return model
    
def create_ConvLSTM(n_t = n_t, tile_size = tile_size):
    
    #Goes from (length, width, height, channels) --> (length, width/2, height/2, n_filters)
    def down_block(x, n_filters):
        y = keras.layers.Conv3D(filters = n_filters, kernel_size = (1, 4, 4), strides = (1, 2, 2), activation = 'relu', padding = 'same')(x)
        y = keras.layers.BatchNormalization()(y)
        return y
    
    #Goes from (length, width, height, filters_in) --> (length, width, height, filters_out)
    def res_block(x,filters_in,filters_out):
        if filters_in==filters_out:
            skip = x
            y = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,4,4), activation = 'relu', padding = 'same')(x)
            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,4,4), padding = 'same')(y)
            y = y + skip
            y = keras.layers.Activation('relu')(y)
        else:
            skip = x
            y = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,4,4), activation = 'relu', padding = 'same')(x)
            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,4,4), padding = 'same')(y)
            skip = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,1,1), padding = 'same')(skip)
            y = y + skip
            y = keras.layers.Activation('relu')(y)
        return y
    
    # (n_t, 64, 64, 1)
    input_data = keras.layers.Input(shape=(n_t, tile_size, tile_size, 1))
    
    # (n_t, 32, 32, 16)
    y = down_block(input_data, 16)
    
    # (n_t, 32, 32, 16)
    y = res_block(y, 16, 16)
    
    # (n_t, 16, 16, 32)
    y = down_block(y, 32)
    
    # (n_t, 16, 16, 32)
    y = res_block(y, 32, 32)
    
    # (n_t, 8, 8, 32)
    y = down_block(y, 32)
    
    # (n_t, 8, 8, 32)
    y = res_block(y, 32, 32)
    
    encoder = keras.models.Model(inputs = input_data, outputs = y)
    
    # (n_t, 8, 8, 32)
    y = keras.layers.Bidirectional(keras.layers.ConvLSTM2D(filters = 16, kernel_size = (4, 4), padding = 'same', return_sequences = True))(encoder.output)
    
    # (n_t, 8, 8, 32)
    y = res_block(y, 32, 32)
    
    # (n_t, 16, 16, 32)
    y = keras.layers.UpSampling3D(size = (1, 2, 2))(y)
    
    # (n_t, 16, 16, 16)
    y = res_block(y, 32, 16)
    
    # (n_t, 32, 32, 16)
    y = keras.layers.UpSampling3D(size = (1, 2, 2))(y)
    
    # (n_t, 32, 32, 8)
    y = res_block(y, 16, 8)
    
    # (n_t, 64, 64, 8)
    y = keras.layers.UpSampling3D(size = (1, 2, 2))(y)
    
    # (n_t, 64, 64, 8)
    y = keras.layers.Conv3D(filters = 8, kernel_size = (1, 4, 4), padding = 'same', activation = 'relu')(y)
    y = keras.layers.BatchNormalization()(y)
    
    # (n_t, 64, 64, 1)
    y = keras.layers.Conv3D(filters = 1, kernel_size = (1, 1, 1), padding = 'same', activation = 'linear')(y)
    
    model = keras.models.Model(inputs=input_data, outputs=y)
    
    return model
    
def create_ConvLSTM_Seeded(n_t = n_t, tile_size = tile_size, seed_size = seed_size, filter_size = filter_size, filter_depth = filter_depth):
    
    #Goes from (length, width, height, channels) --> (length, width/2, height/2, n_filters)
    def down_block(x, n_filters):
        y = keras.layers.Conv3D(filters = n_filters, kernel_size = (1, 4, 4), strides = (1, 2, 2), activation = 'relu', padding = 'same')(x)
        y = keras.layers.BatchNormalization()(y)
        return y
    
    #Goes from (length, width, height, filters_in) --> (length, width, height, filters_out)
    def res_block(x,filters_in,filters_out):
        if filters_in==filters_out:
            skip = x
            y = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,4,4), activation = 'relu', padding = 'same')(x)
            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,4,4), padding = 'same')(y)
            y = y + skip
            y = keras.layers.Activation('relu')(y)
        else:
            skip = x
            y = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,4,4), activation = 'relu', padding = 'same')(x)
            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,4,4), padding = 'same')(y)
            skip = keras.layers.Conv3D(filters = filters_out, kernel_size = (1,1,1), padding = 'same')(skip)
            y = y + skip
            y = keras.layers.Activation('relu')(y)
        return y
    
    # (n_t, 64, 64, 1)
    input_data = keras.layers.Input(shape=(n_t, tile_size, tile_size, 1))
    
    # (n_t, 32, 32, 16)
    y1 = down_block(input_data, 16)
    
    # (n_t, 32, 32, 16)
    y1 = res_block(y1, 16, 16)
    
    # (n_t, 16, 16, 32)
    y1 = down_block(y1, 32)
    
    # (n_t, 16, 16, 32)
    y1 = res_block(y1, 32, 32)
    
    # (n_t, 8, 8, 32)
    y1 = down_block(y1, 32)
    
    # (n_t, 8, 8, 32)
    y1 = res_block(y1, 32, 32)
    
    # (n_t, 8, 8, 32)
    y1 = keras.layers.Bidirectional(keras.layers.ConvLSTM2D(filters = 16, kernel_size = (4, 4), padding = 'same', return_sequences = True))(y1)
    
    data_encoder = keras.models.Model(inputs = input_data, outputs = y1)
    
    # (100)
    seed = keras.layers.Input(shape=(seed_size,))
    
    y2 = keras.layers.Dense(n_t * 8 * 8 * 32, use_bias=False)(seed)
    
    y2 = keras.layers.Reshape((n_t, 8, 8, 32))(y2)
    
    
    y2 = keras.layers.Bidirectional(keras.layers.ConvLSTM2D(filters = 16, kernel_size = (4, 4), padding = 'same', return_sequences = True))(y2)
    
    seed_encoder = keras.models.Model(inputs = seed, outputs = y2)
    
    combined = keras.layers.concatenate([data_encoder.output, seed_encoder.output])
    
    
    y = keras.layers.Bidirectional(keras.layers.ConvLSTM2D(filters = 16, kernel_size = (4,4), padding='same', return_sequences=True))(combined)
    y = keras.layers.BatchNormalization()(y)
    y = res_block(y,32,32)
    
    y = keras.layers.UpSampling3D(size = (1,2,2))(y)
    y = res_block(y,32,16)
    
    y = keras.layers.UpSampling3D(size = (1,2,2))(y)
    y = res_block(y,16,8)
    
    y = keras.layers.UpSampling3D(size = (1,2,2))(y)
    y = keras.layers.Conv3D(filters = 8, kernel_size = (1,4,4), padding = 'same', activation = 'relu')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Conv3D(filters = 1, kernel_size = (1,1,1), padding = 'same', activation = 'linear')(y)

    model = keras.models.Model(inputs=[input_data, seed], outputs=y)
    
    return model