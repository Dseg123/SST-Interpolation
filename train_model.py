import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import copy
import time
import datetime
from numba import cuda

from src.generators import *
from src.models import *
from src.losses import *
from get_params import *

experiment_name = 'convlstm_sst_attempt_' + str(datetime.datetime.now())
model_weights_dir = 'weights/'
n_epochs = 100



def get_summary_stats(tile_stats):
    sum_0 = tile_stats[:, :, 0]
    sum_1 = tile_stats[:, :, 1]
    sum_2 = tile_stats[:, :, 2]
    
    final_num = np.sum(sum_0)
    if final_num == 0:
        final_mean = 0
        final_std = 1
    else:
        final_mean = np.nansum(sum_1)/final_num
        final_std = np.sqrt(np.nansum(sum_2)/final_num - (final_mean)**2)
    
    return (final_num, final_mean, final_std)

data_dir = "training_data"

tile_stats = np.load(data_dir + '/masked_stats.npy')
stats = get_summary_stats(tile_stats)
print(stats)
time.sleep(5)
train_ids = np.load(data_dir + '/train_ids.npy')
val_ids = np.load(data_dir + '/val_ids.npy')

train_gen = get_src_dataset(data_dir, list_ids = train_ids, window_size = 2)
val_gen = get_src_dataset(data_dir, list_ids = val_ids, window_size = 2)

model = create_ConvLSTM()
model.compile(loss = mse_loss, optimizer = keras.optimizers.Adam(learning_rate = 5e-4), run_eagerly=True)

stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, restore_best_weights=True)
saving = keras.callbacks.ModelCheckpoint(model_weights_dir+experiment_name+'.h5', save_weights_only=True, monitor='val_loss', mode = 'min',save_best_only= True)
lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=8,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=1e-4,
)
history_log = keras.callbacks.CSVLogger(experiment_name+'_log'+'.csv', separator=",", append=True)
callbacks = [stopping, saving, lr, history_log]

history = model.fit(train_gen, validation_data = val_gen, epochs = n_epochs, callbacks = callbacks, use_multiprocessing=True,workers=10,max_queue_size=100)

val_loss = history.history['val_loss']
train_loss = history.history['loss']
np.save(experiment_name + '_history.npy', np.stack((train_loss, val_loss), axis = -1))
prediction = model.predict(validation_generator)
np.save(experiment_name + '_prediction.npy', prediction)