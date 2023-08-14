import tensorflow as tf
import numpy as np
import os
import time
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import datetime
import random
import pandas as pd

from IPython import display
from gan.gan_dataset import get_gan_dataset
from gan.gan_models import make_discriminator_model
from src.generators import *
from src.models import *
from src.losses import *

batch_size = 12

epochs = 100
batches_per_epoch = 3000


src_data_dir = "training_data"
data_params = pd.read_csv(src_data_dir + "/data_params.csv")
window_size = data_params['windowSize'].iloc[0]
tile_size = data_params['tileSize'].iloc[0]
avg_size = data_params['avgSize'].iloc[0]
block_size = data_params['blockSize'].iloc[0]

src_window_size = window_size
src_sample_size = window_size * (block_size // avg_size)
src_tile_size = tile_size


learning_rate = 1e-4


src_train_ids = np.load(src_data_dir + '/train_ids.npy')
src_val_ids = np.load(src_data_dir + '/val_ids.npy')
np.random.shuffle(src_train_ids)
np.random.shuffle(src_val_ids)

mean, std, src_train_dataset = get_src_dataset(src_data_dir, src_train_ids, src_window_size, batch_size = batch_size)
mean, std, src_val_dataset = get_src_dataset(src_data_dir, src_val_ids, src_window_size, batch_size = batch_size)


interpolator = create_ConvLSTM(n_t = src_sample_size, tile_size = src_tile_size)
#generator = make_generator_model(filter_size = size, filter_depth = depth)


interpolator_optimizer = tf.keras.optimizers.Adam(learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')

val_loss = tf.keras.metrics.Mean(name='val_loss')

timestamp = str(datetime.datetime.now()).replace(" ", "_")
path = "experiments/experiment_" + timestamp

os.system("mkdir " + path)

model_params = {'batch_size': [batch_size],
                'batches_per_epoch': [batches_per_epoch],
                'src_sample_size': [src_sample_size],
                'src_window_size': [src_window_size],
                'src_tile_size': [src_tile_size],
                'epochs': [epochs],
                'src_data_dir': [src_data_dir],
                'learning_rate': [learning_rate],
                'mean': [mean],
                'std': [std]
               }

params_df = pd.DataFrame.from_dict(model_params)
params_df.to_csv(path + '/model_params.csv', index=False)

checkpoint_dir = path
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(interpolator_optimizer=interpolator_optimizer,
                                 interpolator=interpolator)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(sst_inputs, sst_outputs):

    with tf.GradientTape() as interp_tape:
        pred_outputs = tf.reshape(interpolator(sst_inputs, training=True), sst_outputs.shape)
        # tf.print(real_images.shape)
        # tf.print(generated_images.shape)
        interp_loss = mse_loss(sst_outputs, pred_outputs)
    
    tf.print("interp_l", interp_loss)

    gradients_of_interpolator = interp_tape.gradient(interp_loss, interpolator.trainable_variables)

    interpolator_optimizer.apply_gradients(zip(gradients_of_interpolator, interpolator.trainable_variables))
    
    train_loss(interp_loss)

@tf.function
def val_step(sst_inputs, sst_outputs):
    pred_outputs = tf.reshape(interpolator(sst_inputs, training=False), sst_outputs.shape)

    interp_loss = mse_loss(sst_outputs, pred_outputs)
    
    tf.print("interp_l", interp_loss)
    val_loss(interp_loss)
    


def train(train_src_dataset, val_src_dataset, epochs):
    records = []
    log_dict = {'epoch': [], 'train_loss': [], 'val_loss': []}
    
    src_iter_train = iter(train_src_dataset)
    src_iter_val = iter(val_src_dataset)
    
    best_loss = 1000000000
    for epoch in range(epochs):
        train_loss.reset_states()
        val_loss.reset_states()
        
        start = time.time()
        
        
        
        
        
        
#         print("Train", epoch)
        
#         for i in range(batches_per_epoch):
#             print("Train Batch", i)

#             try:
#                 src_batch = next(src_iter_train)
#             except:
                
#                 src_iter_train = iter(train_src_dataset)
#                 src_batch = next(src_iter_train)
            
#             train_step(src_batch[0], src_batch[1])
        
#         print("Val", epoch)
        
#         for i in range(batches_per_epoch):
#             print("Val Batch", i)
            
#             try:
#                 src_batch = next(src_iter_val)
#             except:
                
#                 src_iter_val = iter(val_src_dataset)
#                 src_batch = next(src_iter_val)
            
#             val_step(src_batch[0], src_batch[1])

        counter = 0
        print("Train", epoch)
        for src_batch in train_src_dataset:
            counter += 1
            print("Train Batch", counter)

            # print(gan_batch.shape)
            # print(src_batch[0].shape)
            # print(src_batch[1].shape)
            train_step(src_batch[0], src_batch[1])
            
            # if counter == 100:
            #     break
        
        counter = 0
        print("Val", epoch)
        for src_batch in val_src_dataset:
            counter += 1
            print("Val Batch", counter)

            val_step(src_batch[0], src_batch[1])
            
            # if counter == 100:
            #     break

        # Produce images for the GIF as you go
        # display.clear_output(wait=True)
        # generate_and_save_images(generator,
        #                          epoch + 1,
        #                          seed)
    
        
        log_dict = {
            'epoch': epoch + 1,
            'train_gen_loss': train_loss.result().numpy(),
            'val_gen_loss': val_loss.result().numpy()
        }
        print(log_dict)
        records.append(log_dict)
        df = pd.DataFrame.from_records(records)
        df.to_csv(path + '/log.csv', index=False)
        
        if val_loss.result() < best_loss:
            interpolator.save_weights(path + f'/interpolator_weights_epoch_{epoch}.h5')

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))



train(src_train_dataset, src_val_dataset, epochs)





