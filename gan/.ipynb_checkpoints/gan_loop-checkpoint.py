
import tensorflow as tf
from gan_models import make_generator_model, make_discriminator_model
import numpy as np
from gan_dataset import get_dataset
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
import pandas as pd

from IPython import display

BATCH_SIZE = 12
train_frac = 0.8

data_dir = "gan_training_data"
gan_stats = np.load(data_dir + '/stats.npy')
gan_params = pd.read_csv(data_dir + '/data_params.csv')
size = gan_params.iloc[0]['filterSize']
depth = gan_params.iloc[0]['filterDepth']


generator = make_generator_model(filter_size = size, filter_depth = depth)
discriminator = make_discriminator_model(filter_size = size, filter_depth = depth)

indexes = np.arange(gan_stats[2])
np.random.shuffle(indexes)
train_indexes = indexes[: int(train_frac * len(indexes))]
test_indexes = indexes[int(train_frac * len(indexes)) : ]

train_dataset = get_dataset(data_dir, gan_stats, train_indexes, batch_size = BATCH_SIZE)
test_dataset = get_dataset(data_dir, gan_stats, test_indexes, batch_size = BATCH_SIZE)
print("Got datasets!")

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
train_gen_loss = tf.keras.metrics.Mean(name='train_gen_loss')
train_disc_loss = tf.keras.metrics.Mean(name='train_disc_loss')
train_gen_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_gen_accuracy')

test_gen_loss = tf.keras.metrics.Mean(name='test_gen_loss')
test_disc_loss = tf.keras.metrics.Mean(name='test_disc_loss')
test_gen_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_gen_accuracy')



def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


timestamp = str(datetime.datetime.now()).replace(" ", "_")
path = "experiment_" + timestamp

os.system("mkdir " + path)
checkpoint_dir = path
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    print(images.shape)
    noise = tf.random.normal([images.shape[0], noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    tf.print(gen_loss, disc_loss)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    train_gen_loss(gen_loss)
    train_disc_loss(disc_loss)
    train_gen_accuracy(tf.math.round(real_output), tf.ones_like(real_output))
    train_gen_accuracy(tf.math.round(fake_output), tf.zeros_like(fake_output))

@tf.function
def test_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    generated_images = generator(noise, training=False)
    
    real_output = discriminator(images, training=False)
    fake_output = discriminator(generated_images, training=False)
    
    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)
    
    test_gen_loss(gen_loss)
    test_disc_loss(disc_loss)
    test_gen_accuracy(tf.math.round(real_output), tf.ones_like(real_output))
    test_gen_accuracy(tf.math.round(fake_output), tf.zeros_like(fake_output))
    


    
def train(train_dataset, test_dataset, epochs):
    records = []
    log_dict = {'epoch': [], 'train_gen_loss': [], 'train_disc_loss': [], 'train_gen_accuracy': [],
                'test_gen_loss': [], 'test_disc_loss': [], 'test_gen_accuracy': []}
    
    
    best_gen_loss = 1000000000
    best_disc_loss = 100000000
    for epoch in range(epochs):
        start = time.time()
        
        train_gen_loss.reset_states()
        train_disc_loss.reset_states()
        train_gen_accuracy.reset_states()
        test_gen_loss.reset_states()
        test_disc_loss.reset_states()
        test_gen_accuracy.reset_states()


        counter = 0
        for image_batch in train_dataset:
            counter += 1
            train_step(image_batch)
            print(counter)
            
        for image_batch in test_dataset:
            test_step(image_batch)

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
            'test_gen_loss': test_gen_loss.result().numpy(),
            'test_disc_loss': test_disc_loss.result().numpy(),
            'test_gen_acc': test_gen_accuracy.result().numpy()
        }
        print(log_dict)
        records.append(log_dict)
        df = pd.DataFrame.from_records(records)
        df.to_csv(path + '/log.csv', index=False)
        
        if test_gen_loss.result() < best_gen_loss:
            best_gen_loss = test_gen_loss.result()
            generator.save_weights(path + '/generator_weights.h5')
            print("Best generator!")
        
        
        if test_disc_loss.result() < best_disc_loss:
            best_disc_loss = test_disc_loss.result()
            discriminator.save_weights(path + '/discriminator_weights.h5')
            print("Best discriminator!")
        
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  # display.clear_output(wait=True)
  # generate_and_save_images(generator,
  #                          epochs,
  #                          seed)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

train(train_dataset, test_dataset, EPOCHS)
