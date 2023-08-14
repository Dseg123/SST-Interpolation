import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import copy
import time
from numba import cuda

from src.generators import *
from src.models import *
from src.losses import *
from get_params import *

my_params = params()
batch_size = my_params['batchSize']

experiment_name = 'convlstm_sst_attempt_' + str(int(time.time()))
model_weights_dir = 'weights/'
n_epochs = 100

my_data_params = params()
data_dir = my_data_params['dataDir']

train_ids = np.load(data_dir + '/train_ids.npy')
val_ids = np.load(data_dir + '/val_ids.npy')

num_train_batches = len(train_ids) // batch_size
num_val_batches = len(val_ids) // batch_size
train_gen = DataGeneratorFast(num_batches = num_train_batches, task = 'train', shuffle = True)
val_gen = DataGeneratorFast(num_batches = num_val_batches, task = 'val', shuffle = False)

model = create_ConvLSTM()
model.compile(loss = mse_loss, optimizer = keras.optimizers.Adam(learning_rate = 5e-4), run_eagerly=True)

stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, restore_best_weights=True)
saving = keras.callbacks.ModelCheckpoint(model_weights_dir+experiment_name+'.h5', save_weights_only=True, monitor='loss', mode = 'min',save_best_only= True, save_freq=100)
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
