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

src_data_dir = "training_data"
gan_data_dir = "gan/gan_data_size64_duration6_window2_4hourly/real_data_size64_stride1_depth1"

data_params = pd.read_csv(src_data_dir + "/data_params.csv")
window_size = data_params['windowSize'].iloc[0]
tile_size = data_params['tileSize'].iloc[0]
avg_size = data_params['avgSize'].iloc[0]
block_size = data_params['blockSize'].iloc[0]

src_window_size = window_size
src_sample_size = window_size * (block_size // avg_size)
src_tile_size = tile_size

batch_size = 12

gan_size = 64
gan_depth = 1
gan_train_frac = 0.8
gan_val_frac = 0.1
lambda_gan = 0.1

src_window_size = 2
src_sample_size = 12
src_tile_size = 64
src_seed_size = 100
lambda_mse = 0.9

epochs = 50




learning_rate = 1e-4




gan_stats = np.load(gan_data_dir + '/stats.npy')
indexes = np.arange(gan_stats[2])
np.random.shuffle(indexes)
gan_train_indexes = indexes[: int(gan_train_frac * len(indexes))]
gan_val_indexes = indexes[int(gan_train_frac * len(indexes)) : int((gan_train_frac + gan_val_frac) * len(indexes))]
gan_test_indexes = indexes[int((gan_train_frac + gan_val_frac) * len(indexes)) : ]
gan_train_dataset = get_gan_dataset(gan_data_dir, gan_train_indexes, batch_size = batch_size)
gan_val_dataset = get_gan_dataset(gan_data_dir, gan_val_indexes, batch_size = batch_size)

src_train_ids = np.load(src_data_dir + '/train_ids.npy')
src_val_ids = np.load(src_data_dir + '/val_ids.npy')
np.random.shuffle(src_train_ids)
np.random.shuffle(src_val_ids)

mean, std, src_train_dataset = get_src_dataset(src_data_dir, src_train_ids, src_window_size, batch_size = batch_size)
mean, std, src_val_dataset = get_src_dataset(src_data_dir, src_val_ids, src_window_size, batch_size = batch_size)

interpolator = create_ConvLSTM_Seeded(n_t = src_sample_size, tile_size = src_tile_size, seed_size = src_seed_size,
                                     filter_size = gan_size, filter_depth = gan_depth)
#generator = make_generator_model(filter_size = size, filter_depth = depth)
discriminator = make_discriminator_model(filter_size = gan_size, filter_depth = gan_depth)


interpolator_optimizer = tf.keras.optimizers.Adam(learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)

train_gen_loss = tf.keras.metrics.Mean(name='train_gen_loss')
train_disc_loss = tf.keras.metrics.Mean(name='train_disc_loss')
train_gen_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_gen_accuracy')

val_gen_loss = tf.keras.metrics.Mean(name='val_gen_loss')
val_disc_loss = tf.keras.metrics.Mean(name='val_disc_loss')
val_gen_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_gen_accuracy')

timestamp = str(datetime.datetime.now()).replace(" ", "_")
path = "experiments/gan_experiment_" + timestamp

os.system("mkdir " + path)

model_params = {'batch_size': [batch_size],
                'src_seed_size': [src_seed_size],
                'src_sample_size': [src_sample_size],
                'src_window_size': [src_window_size],
                'src_tile_size': [src_tile_size],
                'epochs': [epochs],
                'src_data_dir': [src_data_dir],
                'gan_data_dir': [gan_data_dir],
                'gan_train_frac': [gan_train_frac],
                'gan_val_frac': [gan_val_frac],
                'gan_size': [gan_size],
                'gan_depth': [gan_depth],
                'learning_rate': [learning_rate],
                'mean': [mean],
                'std': [std],
                'lambda_gan': [lambda_gan],
                'lambda_mse': [lambda_mse]
               }

params_df = pd.DataFrame.from_dict(model_params)
params_df.to_csv(path + '/model_params.csv', index=False)

checkpoint_dir = path
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=interpolator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=interpolator,
                                 discriminator=discriminator)


def get_gan_sample(full_sample):
    a = np.random.randint(src_tile_size - gan_size + 1)
    b = np.random.randint(src_sample_size - gan_depth + 1)
    return full_sample[b:b + gan_depth, a:a + gan_size, a:a + gan_size]

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(real_images, sst_inputs, sst_outputs):
    noise = tf.random.normal([batch_size, src_seed_size])
    #noise = tf.zeros([batch_size, src_seed_size])

    with tf.GradientTape() as disc_tape, tf.GradientTape() as interp_tape:
        pred_outputs = tf.reshape(interpolator([sst_inputs, noise], training=True), sst_outputs.shape)
        generated_images = tf.stack([get_gan_sample(pred_outputs[i, :, :, :]) for i in range(pred_outputs.shape[0])]) #extract random snapshots
        
        # tf.print(real_images.shape)
        # tf.print(generated_images.shape)
        real_output = discriminator(real_images, training=True)
        tf.print("Real output", real_output.shape)
        tf.print(real_output)
        fake_output = discriminator(generated_images, training=True)
        tf.print("Fake output", fake_output.shape)
        tf.print(fake_output)

        disc_loss = discriminator_loss(real_output, fake_output)
        gen_loss = generator_loss(fake_output)
        mean_loss = mse_loss(sst_outputs, pred_outputs)
        interp_loss = lambda_gan * gen_loss + lambda_mse * mean_loss
    
    tf.print("gen_l", gen_loss, "mse_l", mean_loss, "interp_l", interp_loss, "disc_l", disc_loss)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_interpolator = interp_tape.gradient(interp_loss, interpolator.trainable_variables)

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    interpolator_optimizer.apply_gradients(zip(gradients_of_interpolator, interpolator.trainable_variables))
    
    train_gen_loss(interp_loss)
    train_disc_loss(disc_loss)
    train_gen_accuracy(tf.math.round(tf.math.sigmoid(real_output)), tf.ones_like(real_output))
    train_gen_accuracy(tf.math.round(tf.math.sigmoid(fake_output)), tf.zeros_like(fake_output))

@tf.function
def val_step(real_images, sst_inputs, sst_outputs):
    noise = tf.random.normal([batch_size, src_seed_size])
    #noise = tf.zeros([batch_size, src_seed_size])
    
    pred_outputs = tf.reshape(interpolator([sst_inputs, noise], training=False), sst_outputs.shape)
    generated_images = tf.stack([get_gan_sample(pred_outputs[i, :, :, :]) for i in range(pred_outputs.shape[0])]) #extract random snapshots
    
    real_output = discriminator(real_images, training=False)
    tf.print("Real output", real_output.shape)
    tf.print(real_output)
    fake_output = discriminator(generated_images, training=False)
    tf.print("Fake output", fake_output.shape)
    tf.print(fake_output)
    
    disc_loss = discriminator_loss(real_output, fake_output)
    interp_loss = lambda_gan * generator_loss(fake_output) + lambda_mse * mse_loss(sst_outputs, pred_outputs)
    
    tf.print("interp_l", interp_loss, "disc_l", disc_loss)
    
    val_gen_loss(interp_loss)
    val_disc_loss(disc_loss)
    val_gen_accuracy(tf.math.round(tf.math.sigmoid(real_output)), tf.ones_like(real_output))
    val_gen_accuracy(tf.math.round(tf.math.sigmoid(fake_output)), tf.zeros_like(fake_output))
    


def train(train_src_dataset, train_gan_dataset, val_src_dataset, val_gan_dataset, epochs):
    records = []
    log_dict = {'epoch': [], 'train_gen_loss': [], 'train_disc_loss': [], 'train_gen_accuracy': [],
                'val_gen_loss': [], 'val_disc_loss': [], 'val_gen_accuracy': []}
    
    gan_iter_train = iter(train_gan_dataset)
    src_iter_train = iter(train_src_dataset)
    
    gan_iter_val = iter(val_gan_dataset)
    src_iter_val = iter(val_src_dataset)
    
    best_gen_loss = 1000000000
    best_disc_loss = 100000000
    for epoch in range(epochs):
        start = time.time()
        
        
        
        
        print("Train", epoch)
        
#         for i in range(3000):
#             print("Train Batch", i)
#             try:
#                 gan_batch = next(gan_iter_train)
#                 assert gan_batch[0].shape[0] == batch_size
#             except:
#                 gan_iter_train = iter(train_gan_dataset)
#                 gan_batch = next(gan_iter_train)
            
#             try:
#                 src_batch = next(src_iter_train)
#                 assert src_batch[0].shape[0] == batch_size
#             except:
#                 train_gen_loss.reset_states()
#                 train_disc_loss.reset_states()
#                 train_gen_accuracy.reset_states()
#                 src_iter_train = iter(train_src_dataset)
#                 src_batch = next(src_iter_train)
            
#             train_step(gan_batch, src_batch[0], src_batch[1])
        
        
        counter = 0
        for src_batch in train_src_dataset:
            counter += 1
            print("Train Batch", counter)
            
            if src_batch[0].shape[0] != batch_size:
                continue
        
            try:
                gan_batch = next(gan_iter_train)
                assert gan_batch[0].shape[0] == batch_size
            except:
                gan_iter_train = iter(train_gan_dataset)
                gan_batch = next(gan_iter_train)
            # print(gan_batch.shape)
            # print(src_batch[0].shape)
            # print(src_batch[1].shape)
            train_step(gan_batch, src_batch[0], src_batch[1])
            
#             # if counter == 100:
#             #     break
        

        print("Val", epoch)
        
#         for i in range(3000):
#             print("Val Batch", i)
#             try:
#                 gan_batch = next(gan_iter_val)
#                 assert gan_batch[0].shape[0] == batch_size
#             except:
#                 gan_iter_val = iter(val_gan_dataset)
#                 gan_batch = next(gan_iter_val)
            
#             try:
#                 src_batch = next(src_iter_val)
#                 assert src_batch[0].shape[0] == batch_size
#             except:
#                 val_gen_loss.reset_states()
#                 val_disc_loss.reset_states()
#                 val_gen_accuracy.reset_states()
#                 src_iter_val = iter(val_src_dataset)
#                 src_batch = next(src_iter_val)
            
#             val_step(gan_batch, src_batch[0], src_batch[1])
        
        counter = 0
        for src_batch in val_src_dataset:
            counter += 1
            print("Val Batch", counter)
            
            if src_batch[0].shape[0] != batch_size:
                continue
        
            try:
                gan_batch = next(gan_iter_val)
                assert gan_batch[0].shape[0] == batch_size
            except:
                gan_iter_val = iter(val_gan_dataset)
                gan_batch = next(gan_iter_val)
            val_step(gan_batch, src_batch[0], src_batch[1])
            
            # if counter == 100:
            #     break

        # Produce images for the GIF as you go
        # display.clear_output(wait=True)
        # generate_and_save_images(generator,
        #                          epoch + 1,
        #                          seed)
    
        
        log_dict = {
            'epoch': epoch + 1,
            'train_gen_loss': train_gen_loss.result().numpy(),
            'train_disc_loss': train_disc_loss.result().numpy(),
            'train_gen_acc': train_gen_accuracy.result().numpy(),
            'val_gen_loss': val_gen_loss.result().numpy(),
            'val_disc_loss': val_disc_loss.result().numpy(),
            'val_gen_acc': val_gen_accuracy.result().numpy()
        }
        print(log_dict)
        records.append(log_dict)
        df = pd.DataFrame.from_records(records)
        df.to_csv(path + '/log.csv', index=False)
        
        if val_gen_loss.result() < best_gen_loss:
            best_gen_loss = val_gen_loss.result()
            
            print("Best interpolator!")
        
        interpolator.save_weights(path + f'/interpolator_weights_epoch_{epoch}.h5')
        
        if val_disc_loss.result() < best_disc_loss:
            best_disc_loss = val_disc_loss.result()
            
            print("Best discriminator!")
        discriminator.save_weights(path + f'/discriminator_weights_epoch_{epoch}.h5')
        
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))



train(src_train_dataset, gan_train_dataset, src_val_dataset, gan_val_dataset, epochs)





